import numpy as np
import nibabel as nib
import torch
import torchio
from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation
import torchvision
import time

import networks
import utils

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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

def init_iqa_cnn(ckpt_path):
    net = networks.IQACNN(model_name = 'resnet50')
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True) 
    net.load_state_dict(state_dict) 
    net = net.to(device)    
    net.eval()
    return net

def init_transformer(path_model):
    net = networks.MovingFrameTransformer(d_model=32, num_causal_layers=2, dim_feedforward=64, dropout=0.)
    net = net.to(device)
    net.load_state_dict(torch.load(path_model, map_location=torch.device(device))['net_state_dict'])
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
                [0., 0., 1.,],
                [-1., 0., 0.],
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

def brain_segmentation(vnav, net, measurements, seg_prev=None, volume=None, slice_normal=None, debug_mode=False):
    # obtain brain mask
    img, preproc_aff = utils.preprocess(vnav.get_fdata(), vnav.affine)
    img = img.to(device)
    raw_pred = utils.inference(net, img)
    lab_pred = utils.postprocess(raw_pred=raw_pred, img=img, thresh=0.9)

    if seg_prev is not None:
        transform_to_current = torchio.transforms.Resample(target=torchio.LabelMap(tensor=seg_prev.tensor, affine=preproc_aff))
        seg_prev = transform_to_current(seg_prev).tensor
        shift = np.array(center_of_mass(lab_pred.squeeze().numpy())).astype(np.float32)-np.array(center_of_mass(seg_prev.squeeze().numpy())).astype(np.float32)
        shift_scanner = preproc_aff[:3,:3]@shift
        shift_dir = shift_scanner/np.linalg.norm(shift_scanner)
        shift_along_slice_dir = np.abs(shift_dir@slice_normal / np.linalg.norm(slice_normal))
        if lab_pred.sum() < 0.97*volume and shift_along_slice_dir>0.5 and np.linalg.norm(shift_scanner)>1:
            lab_pred = seg_prev
            print(f'segmentation error detected: using previous segmentation')
    # calculate translation
    if debug_mode:
        lab_pred = torch.ones_like(lab_pred)

    lab_pred_np = lab_pred.squeeze().numpy()
    brain_center_pred = np.array(center_of_mass(lab_pred_np)).astype(np.float32)
    fov_center = np.array(center_of_mass(np.ones_like(lab_pred_np))).astype(np.float32)
    trans_pred = brain_center_pred-fov_center # IN NAVIGATOR COORD SYSTEM
    trans_scanner = preproc_aff[:3,:3] @ trans_pred # IN SCANNER COORD SYSTEM
    brain_center_pred_scanner = preproc_aff[:3,:3] @ brain_center_pred + preproc_aff[:3,3]
    measurements['brain_com'] = brain_center_pred_scanner
    return trans_scanner, img, torchio.LabelMap(tensor=lab_pred, affine=preproc_aff), brain_center_pred_scanner, preproc_aff

def pose_estimation(vnav_params, slice_params, measurements, pose_nets, acquisition_params, debug_mode=False):
    slice_rot_head, slice_trans_head = init_canonical(slice_pos=slice_params['position'],ori=slice_params['orientation']) # SLICE POSE IN HEAD COORD SYSTEM
    e3cnn, transformer = pose_nets
    vnav_aff = vnav_params['affine']
    img = vnav_params['image']
    pred_seg = vnav_params['mask']
    with torch.no_grad():
        
        if debug_mode:
            rot_pred = Rotation.from_euler('xyz', np.random.uniform(-20, 20, size=(3,)), degrees=True).as_matrix() @ utils.get_rot_from_aff(np.linalg.inv(vnav_aff[:3,:3])) # FOR DEBUGGING
        else:
            cropped_image = utils.preprocess_rot_final(img.detach().cpu(), pred_seg.detach().cpu())
            cropped_image = torchio.ScalarImage(tensor=torch.tensor(cropped_image).squeeze().unsqueeze(0).float())
            cropped_img_transformed = normalize_img(cropped_image)
            cropped_img_transformed = cropped_img_transformed.tensor.unsqueeze(0).to(torch.float32).to(device)
            basis_pooled, basis_distribution = e3cnn.forward(cropped_img_transformed, uncertainty=True)
            rot_pred = utils.postprocess_rotation(basis_pooled) # HEAD POSE IN NAVIGATOR COORD SYSTEM

            if measurements['rot0'] is None:
                measurements['rot0'] = rot_pred

            # print(f'before:\n{rot_pred[:,0]}')
            basis_distribution = torch.einsum('ij,xiv->xjv', torch.tensor(measurements['rot0']).float().to(device), basis_distribution)

            measurements['x'].append(torch.cat([basis_distribution[0].T, -basis_distribution[0].T], dim=0))
            measurements['y'].append(basis_distribution[1].T)
            measurements['z'].append(basis_distribution[2].T)
            t = torch.tensor([acquisition_params['TR']*len(measurements['t'])]).to(device).float()
            if len(measurements['t'])==0:
                measurements['t'] = t
            else:
                measurements['t'] = torch.cat([measurements['t'], t])

            if transformer is not None:
                transformer.inference(measurements)
                rot_pred = measurements['rot0']@measurements['rot'][-1].cpu().numpy()
                # print(f'after:\n{rot_pred[:3,0]}')

    rot_pred_scanner = utils.get_rot_from_aff(vnav_aff[:3,:3]) @ rot_pred # HEAD POSE IN SCANNER COORD SYSTEM
    slice_rot_prescribe = rot_pred_scanner @ slice_rot_head # SLICE ROTATION IN SCANNER COORD SYSTEM
    
    slice_trans_prescribe = measurements['brain_com'] + rot_pred_scanner @ slice_trans_head # SLICE TRANSLATION IN SCANNER COORDINATE SYSTEM
    
    slice_rot_send, slice_trans_send = utils.scanner_to_vsend(slice_rot_prescribe, slice_trans_prescribe, acquisition_params['patient_position'])

    return slice_rot_send, slice_trans_send, slice_rot_prescribe, slice_trans_prescribe, slice_rot_head, slice_trans_head, rot_pred

def slice_qa(slice, brain_seg, slice_pos, net):
    # TODO: return network-generated slice quality assessment
    slice_tensor = torch.tensor(
        slice.get_fdata(),
        dtype=torch.float32,
    ).float().to(device)

    resample = torchio.transforms.Resample(target=torchio.ScalarImage(tensor=slice_tensor.unsqueeze(0), affine=slice.affine))
    slice_mask = resample(brain_seg).tensor.bool().to(device)

    resize_img = torchvision.transforms.Resize((244, 244), antialias=True)
    resize_mask = torchvision.transforms.Resize(
        (244, 244),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )

    # Resize image first
    slice_resized = resize_img(slice_tensor.squeeze().unsqueeze(0)).squeeze()  # (H, W)

    # Resize mask separately with nearest-neighbor interpolation
    slice_mask_resized = resize_mask(slice_mask.squeeze().unsqueeze(0).float()).squeeze().bool()  # (H, W)

    # Stack channels: image, image, mask
    stacked = torch.stack(
        [
            slice_resized,
            slice_resized,
            slice_mask_resized.float(),
        ],
        dim=0,
    )  # (3, H, W)
    input = utils.minmax_normalize_2D(stacked, mask=slice_mask_resized).unsqueeze(0)
    
    # RUN NETWORK
    with torch.no_grad():
        logits = net(input)
        probs = torch.nn.functional.softmax(logits, dim=1)
        quality_score = probs[0, 0].item()

    return quality_score, input, slice_resized, slice_mask_resized