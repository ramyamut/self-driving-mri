from collections import namedtuple
import os
import struct
from time import sleep

import nibabel as nb
import numpy as np
import string

def mosaic(data):
    x, y, z = data.shape
    n = int(np.ceil(np.sqrt(z)))
    X = np.zeros((n*x, n*y), dtype=data.dtype)
    for idx in range(z):
        x_idx = int(np.floor(idx/n)) * x
        y_idx = int(idx % n) * y
        # print x_idx, y_idx
        # print data.shape
        X[x_idx:x_idx + x, y_idx:y_idx + y] = data[..., idx]
        #import pylab
        #pylab.imshow(X, interpolation='nearest')
    return X

def demosaic(mosaic, x, y, z):
    data = np.zeros((x, y, z), dtype=mosaic.dtype)
    x,y,z = data.shape
    n = np.ceil(np.sqrt(z))
    dim = int(np.sqrt(np.prod(mosaic.shape)))
    mosaic = mosaic.reshape(dim, dim)
    for idx in range(z):
        x_idx = int(np.floor(idx/n)) * x
        y_idx = int(idx % n) * y
        data[..., idx] = mosaic[x_idx:x_idx + x, y_idx:y_idx + y]
    return data

struct_def_vnav = [('magic', '5s'),
                  ('headerVersion', 'i'),
                  ('seriesUID','64s'),
                  ('scanType', '64s'),
                  ('imageType', '16s'),
                  ('note', '256s'),
                  ('dataType', '16s'),
                  ('isLittleEndian', '?'),
                  ('isMosaic', '?'),
                  ('pixelSpacingReadMM', 'd'),
                  ('pixelSpacingPhaseMM', 'd'),
                  ('pixelSpacingSliceMM', 'd'),
                  ('sliceGapMM', 'd'),
                  ('numPixelsRead', 'i'),
                  ('numPixelsPhase', 'i'),
                  ('numSlices', 'i'),
                  ('voxelToWorldMatrix', '16f'),
                  ('repetitionTimeMS', 'i'),
                  ('repetitionDelayMS', 'i'),
                  ('currentTR', 'i'),
                  ('totalTR', 'i'),
                  ('isMotionCorrected', '?'),
                  ('mcOrder', '5s'),
                  ('mcTranslationXMM', 'd'),
                  ('mcTranslationYMM', 'd'),
                  ('mcTranslationZMM', 'd'),
                  ('mcRotationXRAD', 'd'),
                  ('mcRotationYRAD', 'd'),
                  ('mcRotationZRAD', 'd')]

struct_def_haste = [('magic', '5s'),
                  ('headerVersion', 'i'),
                  ('seriesUID','64s'),
                  ('scanType', '64s'),
                  ('imageType', '16s'),
                  ('note', '256s'),
                  ('dataType', '16s'),
                  ('isLittleEndian', '?'),
                  ('isMosaic', '?'),
                  ('pixelSpacingReadMM', 'd'),
                  ('pixelSpacingPhaseMM', 'd'),
                  ('pixelSpacingSliceMM', 'd'),
                  ('sliceGapMM', 'd'),
                  ('numPixelsRead', 'i'),
                  ('numPixelsPhase', 'i'),
                  ('numSlices', 'i'),
                  ('voxelToWorldMatrix', '16f'),
                  ('repetitionTimeMS', 'i'),
                  ('repetitionDelayMS', 'i'),
                  ('currentTR', 'i'),
                  ('totalTR', 'i'),
                  ('isMotionCorrected', '?'),
                  ('mcOrder', '5s'),
                  ('mcTranslationXMM', 'd'),
                  ('mcTranslationYMM', 'd'),
                  ('mcTranslationZMM', 'd'),
                  ('mcRotationXRAD', 'd'),
                  ('mcRotationYRAD', 'd'),
                  ('mcRotationZRAD', 'd'),
                  ('m_bSendEachSliceFlag', '?'),
                  
                  ('iSliceAnatomicalIndex', 'i'),
                  ('iSliceChronologicalIndex', 'i'),
                  ('dummy', 'i'),]


class ExternalImage(object):

    def __init__(self, typename, image_type='vnav'):
        self.names = []
        fmts = []
        self.image_type = image_type
        if self.image_type == 'vnav':
            self.format_def = struct_def_vnav
        else:
            self.format_def = struct_def_haste
        for key, fmt in self.format_def:
            self.names.append(key)
            fmts.append(fmt)
        self.formatstr = ''.join(fmts)
        self.header_fmt = struct.Struct(self.formatstr)
        self.named_tuple_class = namedtuple(typename, self.names)
        #self.hdr = None
        #self.img = None
        #self.num_bytes = None # image size # readout x # PE

    """
    byte_str: binary str representing header data
    returns an object storing the values for each header field as defined above
    """
    def hdr_from_bytes(self, byte_str):
        alist = list(self.header_fmt.unpack(byte_str))
        values = []
        for idx, key in enumerate(self.names):
            if key != 'voxelToWorldMatrix':
                val = alist.pop(0)
                if isinstance(val, str) or isinstance(val, bytes): # python2 basestring   
                    values.append(val.split(b'\0', 1)[0])
                else:
                    values.append(val)
            else: #1d array representing the transformation matrix needed for constructing the image
                values.append([alist.pop(0) for i in range(16)])
        return self.named_tuple_class._make(tuple(values))

    def hdr_to_bytes(self, hdr_info):
        values = []
        for val in hdr_info._asdict().values():
            if isinstance(val, list):
                values.extend(val)
            else:
                values.append(val)
        return self.header_fmt.pack(*values)

    def create_header(self, img, idx, nt, mosaic, note):
        if len(img.shape) == 3:
            x ,y, z = img.shape
            sx, sy, sz = img._header.get_zooms()
            tr = 1 #
        else:
            x ,y, z, _ = img.shape
            sx, sy, sz, tr = img._header.get_zooms()
        affine = img._affine.flatten().tolist()
        EInfo = self.named_tuple_class
        infotuple = EInfo(magic=b'ERTI',
                          headerVersion=1,
                          seriesUID=b'someuid',
                          scanType=b"EPI",
                          imageType=b'3D',
                          note=note,
                          dataType=b'int16_t',
                          isLittleEndian=True,
                          isMosaic=mosaic,
                          pixelSpacingReadMM=sx,
                          pixelSpacingPhaseMM=sy,
                          pixelSpacingSliceMM=sz,
                          sliceGapMM=0,
                          numPixelsRead=x,
                          numPixelsPhase=y,
                          numSlices=z,
                          voxelToWorldMatrix=affine,
                          repetitionTimeMS=tr*1000,
                          repetitionDelayMS=0,
                          currentTR=idx,
                          totalTR=nt,
                          isMotionCorrected=True,
                          mcOrder=b'XYZT',
                          mcTranslationXMM=0.1,
                          mcTranslationYMM=0.2,
                          mcTranslationZMM=0.01,
                          mcRotationXRAD=0.001,
                          mcRotationYRAD=0.002,
                          mcRotationZRAD=0.0001,
                          bSendEachSliceFlag=True,
                          iSliceAnatomicalIndex=0,
                          iSliceChronologicalIndex=0)
        return infotuple

    def get_header_size(self):
        return self.header_fmt.size

    def get_image_size(self):
        return self.num_bytes

    def from_image(self, img, idx, nt, is_mosaic=True, note=b'some note to leave'):
        hdrinfo = self.create_header(img, idx, nt, is_mosaic, note)
        #if idx is not None:
        #    data = img.get_data()[..., idx]
        #else:
        #    data = img.get_data()
        data = img.get_fdata()
        if is_mosaic:
            data = mosaic(data)
        data = data.flatten().tolist()
        data = [int(d) for d in data]
        num_elem = len(data)
        return self.hdr_to_bytes(hdrinfo), struct.pack('%dH' % num_elem,
                                                       *data)
    """
    in_bytes: str representation of the image
              should be unsigned short integer
              should be size dictated by the self.hdr

    Outputs a Nibabel object representing the image
    """
    def make_img(self, in_bytes, h):
        #h = self.hdr
        num_bytes = len(in_bytes)

        if h.dataType != b'int16_t':    # python3 bytes string
            raise ValueError('Unsupported data type: %s' % h.dataType)

        data = struct.unpack('%dH' % (num_bytes / 2), in_bytes) # %dH
        data = np.array(data).astype(np.float64)
        # print(f"array shape: {data.shape}")
        # print(h)
        
        if h.isMosaic:
            # changed
            nrows = int(np.ceil(np.sqrt(h.numSlices)))
            data = demosaic(data, h.numPixelsRead // nrows, h.numPixelsPhase // nrows, h.numSlices)
        else:
            if self.image_type == "vnav":
                numSlices = h.numSlices
            else:
                numSlices = 1
            output = np.zeros((h.numPixelsRead, h.numPixelsPhase, numSlices))
            n_vox_per_slice = h.numPixelsRead*h.numPixelsPhase
            for s in range(numSlices):
                slice_arr = np.reshape(data[s*n_vox_per_slice:s*n_vox_per_slice+n_vox_per_slice], (h.numPixelsRead, h.numPixelsPhase), order='F')
                output[:,:,s] = slice_arr
        affine = np.array(h.voxelToWorldMatrix).reshape((4, 4))
        # print(affine)
        
        img = nb.Nifti1Image(output, affine)
        img_hdr = img.header
        # img_hdr.set_zooms((h.pixelSpacingReadMM, h.pixelSpacingPhaseMM, h.pixelSpacingSliceMM,))
        img_hdr.set_xyzt_units('mm', 'msec')
        # img_hdr.set_dim_info()
        return output,affine,img

    """
    in_bytes: represents the image header

    Reads the header to determine the image size

    Returns the image header parsed from in_bytes
    """
    def process_header(self, in_bytes):
        magic = struct.unpack('4s', in_bytes[:4])[0] # extract the first 4 byte string which is the Magic #
        #print(magic)
        if magic == b'ERTI' or magic == b'SIMU':  # python3 byte and string
            # header
            #self.hdr = self.hdr_from_bytes(in_bytes)
            #h = self.hdr
            h = self.hdr_from_bytes(in_bytes)
            #print("header received: TR=%d" % h.currentTR)
            if h.isMosaic: # vNav
                nrows = int(np.ceil(np.sqrt(h.numSlices)))
                #self.num_bytes = (2 * h.numPixelsRead * h.numPixelsPhase * nrows * nrows)
                num_bytes = (2 * h.numPixelsRead * h.numPixelsPhase)
            else: # HASTE
                #num_bytes = (2 * h.numPixelsRead * h.numPixelsPhase) if h.bSendEachSliceFlag else (2 * h.numPixelsRead * h.numPixelsPhase * h.numSlices) 
                num_bytes = (2 * h.numPixelsRead * h.numPixelsPhase * h.numSlices)
            
            return h, num_bytes
        else:
            raise ValueError("Unknown magic number %s" % magic)

    """
    in_bytes: str byte data representing image
    """
    def process_image(self, in_bytes, hdr):
        data, affine, img = self.make_img(in_bytes, hdr)
        return data, affine, img
