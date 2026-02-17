import glob
import nibabel as nib
import numpy as np

for f in sorted(glob.glob("outputdata/*nii*")):
    img = nib.load(f)
    img2 = nib.Nifti1Image(img.get_fdata(),np.copy(img.affine))
    nib.save(img2, f)