import torch
import torchio
import numpy as np
from skimage.measure import label

thresh = 0.9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_rot_from_aff(aff):
    rot = np.copy(aff[:3,:3])
    for j in range(3):
        rot[:,j] /= np.linalg.norm(rot[:,j])
    return rot

def get_largest_cc(mask):
    mask = mask#.numpy()
    mask_int = mask.astype(np.uint8)
    labels =label(mask_int, connectivity = 2)
    if np.max(labels) > 0:
    	return labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
    	return labels

def convert_translation(t_vox, affine, spacing):
    t_vox = t_vox * spacing
    t_scanner = np.zeros_like(t_vox).astype(np.float32)
    # CORONAL
    if affine[0,0] != 0.:
    	t_scanner[0] = -t_vox[0]
    	t_scanner[1] = t_vox[2]
    	t_scanner[2] = -t_vox[1]
    # SAGITTAL
    else:
    	t_scanner[0] = -t_vox[2]
    	t_scanner[1] = t_vox[0]
    	t_scanner[2] = -t_vox[1]
    return t_scanner
    
def preprocess(img, aff):
    # read image and corresponding info
    # lab_fov = torchio.ScalarImage(tensor=torch.tensor(img).unsqueeze(0), affine=aff)
    img = torchio.ScalarImage(tensor=torch.tensor(img).unsqueeze(0), affine=aff)
    resample = torchio.transforms.Resample(target=3.)
    croporpad = torchio.transforms.CropOrPad(target_shape=128)
    transform = torchio.transforms.Compose([
    	resample,
    	croporpad,
    	torchio.transforms.RescaleIntensity()
    ])
    img_transformed = transform(img)
    new_aff = img_transformed.affine
    img_transformed = img_transformed.tensor.unsqueeze(0).float()

    # return img_transformed, lab_fov.squeeze().numpy(), new_aff
    return img_transformed, new_aff

def crop_around_brain_scale(image, brain_bbox, scale=0.6):
    brain_center, brain_corner1, brain_corner2 = brain_bbox
    brain_extent = brain_corner2 - brain_corner1
    shape = np.array([brain_extent.mean()/scale]*3)
    brain_corner1 = np.round(brain_center-shape/2).astype(np.int32)
    brain_corner2 = np.round(brain_center+shape/2).astype(np.int32)
    brain_corner1 = np.maximum(brain_corner1, 0)
    brain_corner2 = np.minimum(brain_corner2, image.squeeze().shape)
    crop_bbox = [brain_corner1, brain_corner2]
            
    cropped_image = image[crop_bbox[0][0]:crop_bbox[1][0], crop_bbox[0][1]:crop_bbox[1][1], crop_bbox[0][2]:crop_bbox[1][2]]
    
    return cropped_image

def preprocess_rot_final(vol, label, scale=0.6, resize=[64,64,64]):
    
    vol = vol.squeeze()
    label = label.squeeze()
    # print(vol.shape)
    
    # crop out padded sections
    _, minc, maxc = get_bbox(vol != 0)
    vol_rolled = vol[minc[0]:maxc[0]+1, minc[1]:maxc[1]+1, minc[2]:maxc[2]+1]
    lab_rolled = label[minc[0]:maxc[0]+1, minc[1]:maxc[1]+1, minc[2]:maxc[2]+1]
    
    brain_bbox = get_bbox(lab_rolled)
    
    # scaling
    vol_rolled = crop_around_brain_scale(vol_rolled, brain_bbox, scale=scale)
    
    # resize to 64
    shape = np.array(vol_rolled.shape).astype(np.float32)
    shape *= np.array(resize) / np.max(shape)
    shape = np.round(np.array(shape)).astype(np.int32)
    resize = torchio.transforms.Compose([torchio.Resize(tuple(shape)), torchio.transforms.RescaleIntensity()])
    vol_rolled = resize(torchio.ScalarImage(tensor=torch.tensor(vol_rolled).unsqueeze(0))).tensor.squeeze().unsqueeze(-1).numpy()

    return vol_rolled

def postprocess(raw_pred, img, thresh=0.15):
    
    raw_pred = raw_pred[0].detach().cpu()
    # threshold posteriors
    raw_pred[0, raw_pred[0] <= thresh] = 0
    raw_pred[1, raw_pred[1] <= 0.8] = 0
    
    final_post = torch.softmax(raw_pred, dim=0)
    mask = torch.argmax(final_post, dim=0).to(torch.int32) # [H, W, D]
    mask = mask.cpu().numpy()
    processed_mask = get_largest_cc((mask==1))
    out = torch.tensor(processed_mask).unsqueeze(0)
    out[img[0] == 0] = 0
    return out.detach().cpu()

def axes_to_rotation(xax, yax, zax):
    xfm1 = torch.stack([torch.eye(3)]*xax.shape[0], dim=0)
    xfm1[:,:3,0] = xax
    xfm1[:,:3,1] = yax
    xfm1[:,:3,2] = zax
    xfm2 = torch.clone(xfm1)
    xfm2[:,:3,0] = -xax
    U1, _, Vh1 = torch.linalg.svd(xfm1)
    U2, _, Vh2 = torch.linalg.svd(xfm2)
        
    xfm1 = torch.bmm(U1, Vh1)
    xfm2 = torch.bmm(U2, Vh2)
    det = torch.det(xfm1)
    xfm = []
    for d in range(xfm1.shape[0]):
        if det[d] > 0:
            xfm.append(xfm1[d])
        else:
            xfm.append(xfm2[d])
    xfm = torch.stack(xfm, dim=0).to(xax.device)
    return xfm

def postprocess_rotation(raw_pred):
    pred = raw_pred[:,:,0,0,0].reshape(1, -1, 3).detach()
    pred = torch.nn.functional.normalize(pred,dim=2)
    xfm = axes_to_rotation(pred[:,0], pred[:,1], pred[:,2])
    return xfm.squeeze().detach().cpu().numpy()

def save_seg(mask, aff, save_path):
    seg = torchio.LabelMap(
        tensor=mask,
        affine=aff
    )
    seg.save(path=save_path, squeeze=True)

def save_img(img, aff, save_path):
    img = torchio.ScalarImage(
        tensor=img.detach().cpu(),
        affine=aff
    )
    img.save(path=save_path, squeeze=True)

def get_bbox(label):
    segmentation = np.where(label==1)
    x_min = int(np.min(segmentation[0]))
    x_max = int(np.max(segmentation[0]))
    y_min = int(np.min(segmentation[1]))
    y_max = int(np.max(segmentation[1]))
    z_min = int(np.min(segmentation[2]))
    z_max = int(np.max(segmentation[2]))
    min_corner = np.array([x_min, y_min, z_min])
    max_corner = np.array([x_max, y_max, z_max])
    center = (min_corner + max_corner)/2
    return center, min_corner, max_corner

def inference(net, image):
    with torch.no_grad():
    	pred = torch.softmax(net(image), dim=1)
    	#pred_flip = torch.softmax(torch.flip(net(torch.flip(image, dims=(2,))), dims=(2,))[:,[0,1,3,2]], dim=1)
    	#pred = (pred + pred_flip)/2
    return pred

def init_canonical_translation(slice_pos, ori='axial'):
    assert ori in ['axial', 'sagittal', 'coronal']
    if ori == 'axial':
        return np.array([0,0,slice_pos])
    elif ori == 'sagittal':
        return np.array([slice_pos,0,0])
    else:
        return np.array([0,slice_pos,0])

def scanner_to_vsend(rot_mat_scanner, slice_center_scanner, ras2sdcs):
    main_ori = np.argmax(np.abs(rot_mat_scanner[:,2]))
    sign_ori = rot_mat_scanner[main_ori,2] > 0
    sl = ras2sdcs@rot_mat_scanner[:,2]
    if main_ori == 0: # SAGITTAL
        pe = -ras2sdcs@rot_mat_scanner[:,0]
        ro = -ras2sdcs@rot_mat_scanner[:,1]
    elif main_ori == 1: # CORONAL
        pe = -ras2sdcs@rot_mat_scanner[:,0]
        ro = ras2sdcs@rot_mat_scanner[:,1]
    else: # AXIAL
        pe = ras2sdcs@rot_mat_scanner[:,1]
        ro = -ras2sdcs@rot_mat_scanner[:,0]
    if not sign_ori:
        pe *= -1
    vsend_rot = np.stack([pe, ro, sl], axis=1)
    vsend_t = vsend_rot.T@ras2sdcs@(slice_center_scanner+np.array([0.,0.,7.]))
    return vsend_rot, vsend_t

def vsend_to_scanner(vsend_rot, vsend_t, voxToWorldRot, voxel_res, ras2sdcs):
    sdcs2ras = np.linalg.inv(ras2sdcs)
    main_ori = np.argmax(np.abs(voxToWorldRot[:,2]))
    sign_ori = voxToWorldRot[main_ori,2] > 0
    Z = sdcs2ras@vsend_rot[:,2]
    pe = vsend_rot[:,0]
    ro = vsend_rot[:,1]
    
    if not sign_ori:
        pe *= -1
        dro *= -1
    if main_ori == 0: # SAGITTAL
        X = -sdcs2ras@pe
        Y = -sdcs2ras@ro

    elif main_ori == 1: # CORONAL
        X = -sdcs2ras@pe
        Y = sdcs2ras@ro
    else: # AXIAL
        X = -sdcs2ras@ro
        Y = sdcs2ras@pe
    rot_mat_scanner = np.stack([X, Y, Z], axis=1)
    slice_center_scanner = sdcs2ras@vsend_rot@vsend_t-np.array([0.,0.,7.])
    return rot_mat_scanner, slice_center_scanner

def adjust_slice_t(old_slice, new_rot, center):
    new_affine = np.eye(4)
    center_vox = np.floor(np.array(old_slice.get_fdata().shape)/2)
    new_t = center - new_rot @ center_vox
    new_affine[:3,:3] = new_rot
    new_affine[:3,3] = new_t
    return new_affine

