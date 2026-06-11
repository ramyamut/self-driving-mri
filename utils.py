import torch
import torchio
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import center_of_mass
from skimage.measure import label

thresh = 0.9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

RAS2LPS = np.diag(np.array([-1.,-1.,1.]))
LPS2SDCS = {
    "SUP": np.diag(np.array([1.,-1.,-1.])), # supine
    "LL": np.array([
        [0., -1., 0.],
        [-1., 0., 0.],
        [0., 0., -1]
    ]), # left lateral
}

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
    resize64 = torchio.transforms.Compose([torchio.Resize(64), torchio.transforms.RescaleIntensity()])
    try:
        vol_rolled = resize(torchio.ScalarImage(tensor=torch.tensor(vol_rolled).unsqueeze(0))).tensor.squeeze().unsqueeze(-1).numpy()
    except:
        print('cropping error')
        vol_rolled = resize64(torchio.ScalarImage(tensor=torch.tensor(vol).unsqueeze(0))).tensor.squeeze().unsqueeze(-1).numpy()  

    return vol_rolled

def postprocess(raw_pred, img, thresh=0.15):
    import time
    start = time.time()
    raw_pred = raw_pred[0]
    # threshold posteriors
    raw_pred[0, raw_pred[0] <= thresh] = 0
    raw_pred[1, raw_pred[1] <= 0.8] = 0
    
    final_post = torch.softmax(raw_pred, dim=0)
    mask = torch.argmax(final_post, dim=0).to(torch.int32) # [H, W, D]
    mask = mask.cpu().numpy()
    processed_mask = get_largest_cc((mask==1))
    out = torch.tensor(processed_mask).unsqueeze(0)
    padding_mask = (img[0]==0).cpu()
    out[padding_mask] = 0
    end = time.time()
    # print(f"SEGMENTATION POSTPROCESSING TIME: {end-start}")
    return out

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

def scanner_to_vsend(slice_rot_scanner, slice_center_scanner, patient_position):
    ras2sdcs = LPS2SDCS[patient_position]@RAS2LPS #SCANNER2PATIENT
    slice_rot_patient = ras2sdcs@slice_rot_scanner # SLICE TO PATIENT COORDINATE SYSTEM, ROTATION 
    slice_center_patient = ras2sdcs@slice_center_scanner # SLICE CENTER RELATIVE TO PATIENT COORDINATE SYSTEM
    sl = slice_rot_patient[:,2]

    main_ori = np.argmax(np.abs(slice_rot_patient[:,2]))
    sign_ori = slice_rot_patient[main_ori,2] > 0

    if main_ori == 0: # SAGITTAL RELATIVE TO PATIENT
        pe = slice_rot_patient[:,0]
        ro = -slice_rot_patient[:,1]
    elif main_ori == 1: # CORONAL RELATIVE TO PATIENT
        pe = -slice_rot_patient[:,0]
        ro = slice_rot_patient[:,1]
    else: # AXIAL RELATIVE TO PATIENT
        pe = -slice_rot_patient[:,1]
        ro = -slice_rot_patient[:,0]
    
    if not sign_ori:
        pe *= -1
    slice_gradients_patient = np.stack([pe, ro, sl], axis=1) # SLICE GRADIENTS RELATIVE TO PATIENT
    dgradients = slice_gradients_patient.T@slice_center_patient # ORIGIN OF GRADIENT COORDINATE SYSTEM RELATIVE TO PATIENT
    dpe = dgradients[0]
    dro = dgradients[1]
    dsl = dgradients[2]
    vsend_t = np.array([dro, dpe, dsl])
    return slice_gradients_patient, vsend_t

def scanner_to_vsend_NEW(slice_rot_scanner, slice_center_scanner, patient_position):
    ras2sdcs = LPS2SDCS[patient_position]@RAS2LPS #SCANNER2PATIENT
    slice_rot_patient = ras2sdcs@slice_rot_scanner # SLICE TO PATIENT COORDINATE SYSTEM, ROTATION 
    slice_center_patient = ras2sdcs@slice_center_scanner # SLICE CENTER RELATIVE TO PATIENT COORDINATE SYSTEM
    slice_center_slice = slice_rot_scanner.T@slice_center_scanner # SLICE CENTER RELATIVE TO SLICE COORDINATE SYSTEM
    sl = slice_rot_patient[:,2]
    shift_sl = slice_center_slice[2]

    main_ori = np.argmax(np.abs(slice_rot_patient[:,2]))
    sign_ori = slice_rot_patient[main_ori,2] > 0

    if main_ori == 0: # SAGITTAL RELATIVE TO PATIENT
        pe = slice_rot_patient[:,0]
        ro = -slice_rot_patient[:,1]
        shift_pe = slice_center_slice[0]
        shift_ro = -slice_center_slice[1]
    elif main_ori == 1: # CORONAL RELATIVE TO PATIENT
        pe = -slice_rot_patient[:,0]
        ro = -slice_rot_patient[:,1]
        shift_pe = -slice_center_slice[0]
        shift_ro = -slice_center_slice[1]
    else: # AXIAL RELATIVE TO PATIENT
        pe = -slice_rot_patient[:,1]
        ro = -slice_rot_patient[:,0]
        shift_pe = -slice_center_slice[1]
        shift_ro = -slice_center_slice[0]
    
    if not sign_ori:
        pe *= -1
        shift_pe *= -1
    slice_gradients_patient = np.stack([pe, ro, sl], axis=1) # SLICE GRADIENT DIRECTIONS RELATIVE TO PATIENT
    vsend_t = np.array([shift_ro, shift_pe, shift_sl])
    return slice_gradients_patient, vsend_t

def vsend_to_scanner(slice_gradients_patient, vsend_t, slice_prescribed, patient_position):
    slice_affine_scanner = slice_prescribed.affine
    shift_ro = vsend_t[0]
    shift_pe = vsend_t[1]
    shift_sl = vsend_t[2]
    dgradients = np.array([shift_pe, shift_ro, shift_sl])
    slice_rot_scanner = get_rot_from_aff(slice_affine_scanner)
    ras2sdcs = LPS2SDCS[patient_position]@RAS2LPS #SCANNER2PATIENT
    slice_rot_patient = ras2sdcs@slice_rot_scanner # SLICE TO PATIENT COORDINATE SYSTEM, ROTATION
    slice_center_patient = slice_gradients_patient@dgradients
    slice_center_scanner1 = np.linalg.inv(ras2sdcs)@slice_center_patient
    slice_center_slice = np.array([0.,0.,shift_sl])

    main_ori = np.argmax(np.abs(slice_rot_patient[:,2]))
    sign_ori = slice_rot_patient[main_ori,2] > 0

    if not sign_ori:
        shift_pe *= -1

    if main_ori == 0: # SAGITTAL RELATIVE TO PATIENT
        slice_center_slice[0] = shift_pe
        slice_center_slice[1] = -shift_ro
    elif main_ori == 1: # CORONAL RELATIVE TO PATIENT
        slice_center_slice[0] = -shift_pe
        slice_center_slice[1] = -shift_ro
    else: # AXIAL RELATIVE TO PATIENT
        slice_center_slice[1] = -shift_pe
        slice_center_slice[0] = -shift_ro
    
    slice_center_scanner2 = slice_rot_scanner@slice_center_slice
    new_affine = adjust_slice_t(slice_prescribed, slice_affine_scanner[:3,:3], slice_center_scanner1)
    return new_affine

def adjust_slice_t(old_slice, new_rot, center):
    new_affine = np.eye(4)
    center_vox = np.floor(np.array(old_slice.get_fdata().shape)/2)
    new_t = center - new_rot @ center_vox
    new_affine[:3,:3] = new_rot
    new_affine[:3,3] = new_t
    return new_affine

def compute_crop_margins(mask, margin=5):

    nonzero = torch.nonzero(mask, as_tuple=False)  # (N, 3)

    mins = nonzero.min(dim=0).values
    maxs = nonzero.max(dim=0).values
    D, H, W = mask.shape

    w0, w1 = max(mins[0].item() - margin, 0), min(maxs[0].item() + margin + 1, D)
    h0, h1 = max(mins[1].item() - margin, 0), min(maxs[1].item() + margin + 1, H)
    d0, d1 = max(mins[2].item() - margin, 0), min(maxs[2].item() + margin + 1, W)

    slices = (min(w0, W-w1), min(h0, H-h1), min(d0, D-d1))

    return slices

def distance_to_slice_voxelgrid(volume_shape, affine, plane_point, plane_normal):
    D, H, W = volume_shape
    d = torch.arange(D, dtype=torch.float32).to(device)
    h = torch.arange(H, dtype=torch.float32).to(device)
    w = torch.arange(W, dtype=torch.float32).to(device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing="ij")

    ones = torch.ones_like(grid_d)
    vox_coords = torch.stack([grid_d, grid_h, grid_w, ones], dim=-1)

    world_coords = vox_coords @ affine.T
    world_xyz    = world_coords[..., :3]

    diff  = world_xyz - plane_point
    dist  = (diff * plane_normal).sum(dim=-1)

    return dist
    
class AcquisitionState:
    def __init__(self, mask, rotation):
        self.rotation = rotation
        pose = np.eye(4)
        pose[:3,:3] = rotation
        pose[:3,3] = np.array(center_of_mass(mask.tensor.squeeze().cpu().numpy()))
        t = np.eye(4)
        t[:3,3] = -(np.array(mask.tensor.squeeze().shape)-1)/2

        resample = torchio.transforms.Resample((mask.tensor.squeeze().shape, mask.affine@pose@t))
        resampled = resample(mask)

        # INITIALIZE MAPS
        margins = compute_crop_margins(resampled.tensor.squeeze())
        transform = torchio.transforms.Compose([
            torchio.transforms.Crop(margins),
            torchio.transforms.Resample(target=1.)
        ])
        self.mask = transform(resampled)
        self.affine = self.mask.affine
        self.mask = self.mask.tensor.squeeze().to(device).float()

        self.coverage_map = torch.zeros_like(self.mask).to(device)
        self.spinhistory_map = torch.zeros_like(self.mask).to(device)

    def update(self, slice_normal_canonical, slice_center_canonical, slice_thickness):
        sigma = slice_thickness / 2.355
        t = np.eye(4)
        t[:3,3] = -(np.array(self.mask.shape)-1)/2
        slice_distances = distance_to_slice_voxelgrid(
            self.mask.shape,
            torch.tensor(t).to(device).float(),
            torch.tensor(slice_center_canonical).to(device).float(),
            torch.tensor(slice_normal_canonical).to(device).float()
        )
        # intersection_volumes = ((slice_distances < 0)*self.mask, (slice_distances > 0)*self.mask)
        # print(intersection_volumes[0].sum()/self.mask.tensor.sum(), intersection_volumes[1].sum()/self.mask.tensor.sum())
        slice_psf = torch.exp(-0.5 * (slice_distances/sigma)**2) #* self.mask.tensor

        self.coverage_map += slice_psf
        self.spinhistory_map = slice_psf
    
    def save_coverage(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        map = self.coverage_map #* self.mask
        coverage = nib.Nifti1Image(map.cpu().numpy(), self.affine)
        nib.save(coverage, path)
    
    def save_spinhistory(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        map = self.spinhistory_map #* self.mask
        coverage = nib.Nifti1Image(map.cpu().numpy(), self.affine)
        nib.save(coverage, path)
    
    def save_mask(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        coverage = nib.Nifti1Image(self.mask.cpu().numpy(), self.affine)
        nib.save(coverage, path)

def pose_to_affine(params):
    rx, ry, rz, tx, ty, tz = params
    Rx = torch.stack([
        torch.stack([torch.ones_like(rx),  torch.zeros_like(rx), torch.zeros_like(rx)]),
        torch.stack([torch.zeros_like(rx), torch.cos(rx),        -torch.sin(rx)      ]),
        torch.stack([torch.zeros_like(rx), torch.sin(rx),         torch.cos(rx)      ]),
    ])

    Ry = torch.stack([
        torch.stack([ torch.cos(ry), torch.zeros_like(ry), torch.sin(ry)]),
        torch.stack([ torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)]),
        torch.stack([-torch.sin(ry), torch.zeros_like(ry), torch.cos(ry)]),
    ])

    Rz = torch.stack([
        torch.stack([torch.cos(rz),  -torch.sin(rz), torch.zeros_like(rz)]),
        torch.stack([torch.sin(rz),   torch.cos(rz), torch.zeros_like(rz)]),
        torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)]),
    ])

    R = Rz @ Ry @ Rx
    t = torch.stack([tx, ty, tz])

    affine = torch.eye(4, device=params.device)
    affine[:3, :3] = R
    affine[:3,  3] = t

    return affine

def affine_to_grid(affine, shape):
    D, H, W = shape
    theta = affine[:3, :]
    theta = theta.unsqueeze(0)
    grid  = torch.nn.functional.affine_grid(theta, (1, 1, D, H, W), align_corners=True)
    return grid

def ncc_loss(x, y):
    x = x.flatten()
    y = y.flatten()
    x = x - x.mean()
    y = y - y.mean()
    ncc = (x * y).sum() / (x.norm() * y.norm() + 1e-8)
    return 1.0 - ncc

class RigidRegistration:

    def __init__(
        self,
        n_iters=50,
        lr=1e-3,
        device="cuda",
    ):
        self.n_iters = n_iters
        self.lr      = lr
        self.device  = device

    @torch.no_grad()
    def _sample(self, volume, grid):
        return torch.nn.functional.grid_sample(
            volume, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def register(self, fixed, moving, init_affine=None):
        shape  = fixed.shape
        fixed  = fixed.float().unsqueeze(0).unsqueeze(0).to(self.device)
        moving = moving.float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Residual 6-DoF initialized at zero (identity residual)
        params = torch.zeros(6, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=self.lr)

        # Precompute base grid from NN affine
        if init_affine is not None:
            base_affine = init_affine.to(self.device)
        else:
            base_affine = torch.eye(4, device=self.device)

        for _ in range(self.n_iters):
            optimizer.zero_grad()

            # Compose: residual on top of NN estimate
            residual_affine = pose_to_affine(params)
            full_affine      = residual_affine @ base_affine

            grid   = affine_to_grid(full_affine, shape)
            warped = torch.nn.functional.grid_sample(moving, grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=True)

            loss = ncc_loss(fixed, warped)
            loss.backward()
            optimizer.step()

        # Final warp without grad
        with torch.no_grad():
            residual_affine = pose_to_affine(params)
            full_affine     = residual_affine @ base_affine
            grid            = affine_to_grid(full_affine, shape)
            warped          = torch.nn.functional.grid_sample(moving, grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=True)

        return full_affine.detach(), warped.squeeze().detach()

def minmax_normalize_2D(
    img: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    img:  shape (C, H, W)
    mask: shape (H, W), boolean
    """

    if img.ndim != 3:
        raise ValueError(f"`img` must have shape (C, H, W), got {tuple(img.shape)}")

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"`mask` must be bool, got {mask.dtype}")
        if mask.shape != img.shape[1:]:
            raise ValueError(
                f"`mask` shape must match image spatial shape. "
                f"Got mask {tuple(mask.shape)} and image {tuple(img.shape)}"
            )

    channels_to_norm = img[:-1]

    if mask is not None and mask.any():
        values = channels_to_norm[:, mask]
    else:
        values = channels_to_norm[channels_to_norm > 0]

    if values.numel() == 0:
        raise ValueError("No nonzero values found for normalization")

    img_min = torch.quantile(values, 0.0)
    img_max = torch.quantile(values, 1.0)

    normalized = (channels_to_norm - img_min) / (img_max - img_min + 1e-6)
    normalized = torch.clamp(normalized, 0, 1)

    return torch.cat([normalized, img[-1:]], dim=0)