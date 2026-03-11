import numpy as np
import nibabel as nib
import torch
import torchio
from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation
import time

import networks
import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
normalize_img = torchio.transforms.RescaleIntensity()

# INITIALIZE NETWORKS
def init_unet(ckpt_path):
    net = networks.UNet(
        n_input_channels=1,
        n_output_channels=3,
        n_levels=4,
        n_conv=2,
        n_feat=16,
        feat_mult=2,
        kernel_size=3,
        activation='elu',
        last_activation=None #'sigmoid'
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    keys = list(state_dict.keys())
    new_state_dict = {}
    for k in keys:
        if 'model' in k:
            new_state_dict[k.replace('model.', '')] = state_dict[k]
    net.load_state_dict(new_state_dict, strict=True)
    net.eval()
    return net

def init_e3cnn(ckpt_path):
    net = networks.E3CNN_Encoder(input_chans=1, output_chans=1, n_levels=4, k=5, last_activation=None, equivariance='O3')
    net = net.to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device))['net_state_dict'])
    net.eval()
    return net

def init_canonical(slice_pos, ori='axial'):
    assert ori in ['axial', 'sagittal', 'coronal']
    if ori == 'axial':
        R = np.array([
                [-1., 0., 0.,],
                [0., -1., 0.],
                [0., 0., 1.]
        ])
    elif ori == 'sagittal':
        R = np.array([
                [0., 0., -1.,],
                [1., 0., 0.],
                [0., -1., 0.]
        ])
    else:
        R = np.array([
                [-1., 0., 0.,],
                [0., 0., -1.],
                [0., -1., 0.]
        ])
    slice_pos_vec = np.array([0.,0.,slice_pos])
    t = R @ slice_pos_vec
    return R, t

def run_unet(vnav, net):

    # obtain brain mask
    img, preproc_aff = utils.preprocess(vnav.get_fdata(), vnav.affine)
    img = img.to(device)
    raw_pred = utils.inference(net, img)
    lab_pred = utils.postprocess(raw_pred=torch.clone(raw_pred), img=img.detach().cpu(), thresh=0.9)

    # calculate translation
    if lab_pred.sum() == 0:
        lab_pred = torch.ones_like(lab_pred)
    lab_pred_np = lab_pred.squeeze().numpy()
    brain_center_pred = np.array(center_of_mass(lab_pred_np)).astype(np.float32)
    fov_center = np.array(center_of_mass(np.ones_like(lab_pred_np))).astype(np.float32)
    trans_pred = brain_center_pred-fov_center # IN NAVIGATOR COORD SYSTEM
    trans_scanner = preproc_aff[:3,:3] @ trans_pred # IN SCANNER COORD SYSTEM
    brain_center_pred_scanner = preproc_aff[:3,:3] @ brain_center_pred + preproc_aff[:3,3]

    return trans_scanner, img, lab_pred, brain_center_pred_scanner, preproc_aff

def run_e3cnn(img, pred_seg, vnav_aff, net, slice_pos, ori, brain_center, ras2sdcs):
    slice_rot_head, slice_trans_head = init_canonical(slice_pos=slice_pos,ori=ori) # SLICE POSE IN HEAD COORD SYSTEM
    with torch.no_grad():
        # COMMENT OUT
        cropped_image = utils.preprocess_rot_final(img.detach().cpu(), pred_seg.detach().cpu())
        cropped_image = torchio.ScalarImage(tensor=torch.tensor(cropped_image).squeeze().unsqueeze(0).float())
        cropped_img_transformed = normalize_img(cropped_image)
        cropped_img_transformed = cropped_img_transformed.tensor.unsqueeze(0).to(torch.float32).to(device)
        ecnn_pred = net.forward(cropped_img_transformed)
        rot_pred = utils.postprocess_rotation(ecnn_pred) # HEAD POSE IN NAVIGATOR COORD SYSTEM
        
        # rot_pred = Rotation.from_euler('xyz', np.random.uniform(-20, 20, size=(3,)), degrees=True).as_matrix() @ utils.get_rot_from_aff(np.linalg.inv(vnav_aff[:3,:3])) # FOR DEBUGGING
        # rot_pred = utils.get_rot_from_aff(np.linalg.inv(vnav_aff[:3,:3])) # FOR DEBUGGING

    rot_pred_scanner = utils.get_rot_from_aff(vnav_aff[:3,:3]) @ rot_pred # HEAD POSE IN SCANNER COORD SYSTEM
    slice_rot_prescribe = rot_pred_scanner @ slice_rot_head # SLICE ROTATION IN SCANNER COORD SYSTEM
    
    # brain_center = np.array([0,0,0]) # FOR DEBUGGING
    slice_trans_prescribe = brain_center + rot_pred_scanner @ slice_trans_head # SLICE TRANSLATION IN SCANNER COORDINATE SYSTEM
    
    slice_rot_send, slice_trans_send = utils.scanner_to_vsend(slice_rot_prescribe, slice_trans_prescribe, ras2sdcs)

    return slice_rot_send, slice_trans_send, rot_pred

def run_qa_cnn(slice, slice_pos):
    # TODO: return network-generated slice quality assessment
    qa = 1.0
    return qa