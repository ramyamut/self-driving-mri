#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
import argparse
from collections import namedtuple
import os
import socket
import sys
from time import sleep, time
import threading
import socketserver as SocketServer
import external_image
import nibabel as nb
import numpy as np
import struct
from queue import Queue
import warnings

import utils, realtime_utils

SocketServer.TCPServer.allow_reuse_address = True


class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):
    def __init__(self, callback, infoclient, *args, **keys):
        self.callback = callback
        self.infoclient = infoclient
        SocketServer.BaseRequestHandler.__init__(self, *args, **keys)

    def handle(self): # process incoming requests from the client
        self.callback(self.infoclient, self.request)


# ThreadingMixIn allows for asynchronous processing of requests 
class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass
        

def handler_factory(callback, infoclient):
    def createHandler(*args, **keys):
        return ThreadedTCPRequestHandler(callback, infoclient,  *args, **keys)
    return createHandler


def process_data_callback(infoclient, sock):
    infoclient.process_data(sock)

# img: nibabel object Nifti1Image
def save_nifti(img, imgtype, save_location, uid, index):
    os.makedirs(save_location, exist_ok=True)
    filename = os.path.join(save_location,
            '%s-%s-%05d.nii.gz' % (imgtype, uid.decode(), index))
    img.to_filename(filename)

def save_npy(arr, imgtype, save_location, uid, index):
    os.makedirs(save_location, exist_ok=True)
    filename = os.path.join(save_location,
            '%s-%s-%05d.npy' % (imgtype, uid.decode(), index))
    np.save(filename, arr)


"""
In this model, the computer is the server and the scanner is the client
i.e., scanner requests to send data to the computer.
Scanner makes the connection to the computer.

Input the IP/PORT of the computer "server" into the vsend configurator for the scanner
"""

class Server(object):

    def __init__(self, name, host, port):
        self.name = name
        self.host = host
        self.port = port
        self._is_running = False
        self._server = None
    
    def stop(self):
        self._server.shutdown()
        self._is_running = None
        self._server = None
        print("%s stopped" % self.name)

    def start(self):
        self._startserver()

    def _startserver(self):
        if self._is_running:
            raise RuntimeError('Server already running')
        server = ThreadedTCPServer((self.host, self.port), handler_factory(process_data_callback, self))
        ip, port = server.server_address
        print("%s running at %s on port %d" % (self.name, ip, port))
        # Start a thread with the server -- that thread will then start one more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        self._is_running = True
        self._server = server

    def process_data(self, sock):
        raise NotImplementedError('')


class ImageReceiver(Server):

    def __init__(self, args, shared_data):
        super().__init__('Image Receiver', args.host, args.port)
        self.host = args.host
        self.port = args.port
        self.debug_mode = args.debug_mode
        self.no_moco = args.no_moco

        self._is_running = None
        self._server = None
        self.imagestore = []
        self.output_dir = args.save_directory
        self.subdirs = [os.path.join(self.output_dir, sd) for sd in ["navigators", "slices", "vsend_params", "for_debugging", "acquisition_state"]]
        self.current_uid = None
        self.current_series_hdr = None
        self.save_4d = args.four_dimensional
        self.stop_after_one_series = args.single_series

        self.ei_vnav = external_image.ExternalImage("ExternalImageHeader")
        self.ei_haste = external_image.ExternalImage("ExternalImageHeader", image_type='haste')
        self.shared_data = shared_data

        self.counter_HASTE = 0
        self.counter_vNav = 0
        self.stack = None

        # init acquisition params
        self.slice_thickness = args.slice_thickness
        self.TR = args.TR
        self.ori = args.orientation
        assert(self.ori in ['axial', 'sagittal', 'coronal'])
        self.patient_position = args.position
        self.num_slices = args.n_slices
        self.slice_zs_ordered = np.arange(0,self.num_slices)*self.slice_thickness
        self.slice_zs_ordered -= self.slice_zs_ordered.mean()
        self.slice_zs = []
        self.anatomical_idxs = []
        self.n_sweeps = args.n_interleave
        self.vnav0 = None
        for sweep in range(self.n_sweeps):
            self.anatomical_idxs += list(range(sweep, self.num_slices, self.n_sweeps))
            self.slice_zs += self.slice_zs_ordered[list(range(sweep, self.num_slices, self.n_sweeps))].tolist()
        print(self.anatomical_idxs)
        print(self.slice_zs)
        self.qa = 1.0
        self.qas = np.array([0.]*self.num_slices)

        # init networks
        unet_path = args.unet_path
        self.unet = realtime_utils.init_unet(unet_path)
        e3cnn_path = args.e3cnn_path
        self.e3cnn = realtime_utils.init_e3cnn(e3cnn_path)
        transformer_path = args.transformer_path
        if transformer_path != "":
            self.transformer = realtime_utils.init_transformer(transformer_path)
        else:
            self.transformer = None
        self.pose_nets = (self.e3cnn, self.transformer)
        iqa_cnn_path = args.iqa_cnn_path
        self.iqa_cnn = realtime_utils.init_iqa_cnn(iqa_cnn_path)
        self.nav_FOV_center_send = np.array([0., 0., 0.])
        self.seg_curr = None
        self.brain_volume = None
        self.slice_normal = None
        self.tracking_measurements = {'x': [], 'y': [], 'z': [], 't': [], 'brain_com': None, 'rot0': None, 'rot': None}

        # START UP NETWORKS WITH INFERENCE ON DUMMY INPUTS
        seg = None
        volume = None
        slice_normal = None
        dummy_measurements = {'x': [], 'y': [], 'z': [], 't': [], 'brain_com': None, 'rot0': None, 'rot': None}
        for startup_run in range(32):
            start1 = time()
            # dummy_navigator = nb.Nifti1Image(np.random.uniform(size=(64,64,24)), np.diag([5.5,5.5,5.5,1]))
            import glob
            dummy_navigator = nb.load(sorted(glob.glob("experiments/in_utero/AFI2-133/30w/axial_002/navigators/vNav*"))[startup_run])
            print(f"frame {startup_run}")
            _, img_input, seg, _, _ = realtime_utils.brain_segmentation(dummy_navigator, self.unet, dummy_measurements, seg_prev=seg, volume=volume, slice_normal=slice_normal, debug_mode=self.debug_mode)
            if volume is None:
                volume = seg.tensor.sum()
            end1 = time()
            _ = realtime_utils.pose_estimation(
                vnav_params={'image': img_input, 'mask': seg.tensor, 'affine': dummy_navigator.affine},
                slice_params={'position': 0, 'orientation': 'axial'},
                measurements=dummy_measurements,
                pose_nets=self.pose_nets,
                acquisition_params={'patient_position': self.patient_position, 'TR': self.TR},
                debug_mode=self.debug_mode
            )
            end2 = time()
            print(f"Total navigator processing time: {end2-start1}, {end1-start1}, {end2-end1}")
            start1 = time()
            dummy_slice = nb.load(sorted(glob.glob("experiments/in_utero/AFI2-133/30w/axial_002/slices/HASTE*"))[startup_run])
            slice_normal = dummy_slice.affine[:3,2]
            _ = realtime_utils.slice_qa(dummy_slice, seg, 0, self.iqa_cnn)
            end1 = time()
            print(f"Total slice processing time: {end1-start1}")

    
    def stop(self):
        self._server.shutdown()
        self._is_running = None
        self._server = None

        if self.save_4d:
            self.save_imagestore()

        print("image receiver stopped")
    
    def start(self):
        self._startserver()

    def check(self):
        if not self._is_running:
            raise RuntimeError('Server is not running')
        return self.imagestore

    def _startserver(self):
        if self._is_running:
            raise RuntimeError('Server already running')

        server = ThreadedTCPServer((self.host, self.port),
                                   handler_factory(process_data_callback, self))
        ip, port = server.server_address
        print("image receiver running at %s on port %d" % (ip, port))
        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        self._is_running = True
        self._server = server
    
    def process_data(self, sock):
        # PW 2023/12/06: The first 8 bytes are the size of the header and then size of the data..
        sizes_buf = sock.recv(8)
        header_size, data_size = struct.unpack('<ii', sizes_buf)

        vnav = True
        if header_size == self.ei_vnav.get_header_size():
            ei = self.ei_vnav
            imgtype = "vNav"
        elif header_size == self.ei_haste.get_header_size():
            ei = self.ei_haste
            imgtype = "HASTE"
            vnav = False
        else:
            raise ValueError(
              "Expecting a header size of %d (vnav) or %d (haste), but vsend wants to send %d" %
              (self.ei_vnav.get_header_size(), self.ei_haste.get_header_size(), header_size)
            )
            
        in_bytes = sock.recv(header_size)
        if len(in_bytes) != header_size:
            raise ValueError(
                "Header data wrong size: expected %d bytes, got %d" %
                (header_size, len(in_bytes))
                )

        hdr = ei.process_header(in_bytes)[0]
        if self.stack is None:
            # self.num_slices =hdr.numSlices
            self.stack = [None]*self.num_slices
        self.slice_pos = self.slice_zs[self.counter_HASTE]
        self.anatomical_idx = self.anatomical_idxs[self.counter_HASTE]

        # validation
        if self.current_uid != hdr.seriesUID:
            #assert hdr.currentTR == 1
            self.current_uid = hdr.seriesUID
            self.current_series_hdr = hdr

        img_data = sock.recv(data_size)
        while len(img_data) < data_size:
            in_bytes = sock.recv(4096)
            img_data += in_bytes
        img_data = img_data[:data_size]
        
        if len(img_data) != data_size:
            raise ValueError(
                "Image data wrong size: expected %d bytes, got %d" %
                (data_size, len(img_data))
                )

        new_ei = ei.process_image(img_data, hdr)[2]
        if new_ei:
            if (isinstance(new_ei, nb.Nifti1Image) and
                new_ei not in self.imagestore):
                self.imagestore.append(new_ei)
                if vnav:
                    print(f"Received navigator #{self.counter_vNav:03}")

                    # INITIALIZATION
                    if self.vnav0 is None:
                        self.vnav0 = new_ei
                    nav_FOV_center_curr = new_ei.affine[:3,3] - self.vnav0.affine[:3,3]

                    # PREDICT BRAIN MASK
                    pred_trans, img_input, self.seg_curr, brain_center_pred_scanner, proc_aff = realtime_utils.brain_segmentation(new_ei, self.unet, self.tracking_measurements, seg_prev=self.seg_curr, volume=self.brain_volume, slice_normal=self.slice_normal, debug_mode=self.debug_mode)
                    if self.brain_volume is None:
                        self.brain_volume = self.seg_curr.tensor.sum()

                    # COMPUTE ABSOLUTE NAVIGATOR TRANSLATION TO SEND TO SCANNER
                    nav_FOV_center_prescribe = nav_FOV_center_curr + pred_trans
                    self.nav_FOV_center_send = utils.LPS2SDCS[self.patient_position]@utils.RAS2LPS@nav_FOV_center_prescribe
                    clip = 199.9
                    if np.linalg.norm(self.nav_FOV_center_send) > clip:
                        self.nav_FOV_center_send = self.nav_FOV_center_send / np.linalg.norm(self.nav_FOV_center_send) * clip
                    
                    # PREDICT ROTATION
                    self.tracking_measurements['brain_com'] = brain_center_pred_scanner
                    pose_output = realtime_utils.pose_estimation(
                        vnav_params={'image': img_input, 'mask': self.seg_curr.tensor, 'affine': new_ei.affine},
                        slice_params={'position': self.slice_pos, 'orientation': self.ori},
                        measurements=self.tracking_measurements,
                        pose_nets=self.pose_nets,
                        acquisition_params={'patient_position': self.patient_position, 'TR': self.TR},
                        debug_mode=self.debug_mode
                    )
                    self.slice_rot_send, self.slice_trans_send, self.slice_rot_scanner, self.slice_center_scanner, self.slice_rot_head, self.slice_trans_head, nav_rot_pred = pose_output
                    self.slice_normal = self.slice_rot_scanner[:3,2]

                    # SEND NAVIGATOR FOV + SLICE PRESCRIPTION PARAMETERS TO SCANNER
                    result = tuple(self.nav_FOV_center_send.tolist() + self.slice_trans_send.tolist() + self.slice_rot_send.flatten().tolist() + [self.qa])
                    self.shared_data.put("vNav", (result, self.counter_vNav))

                    # INITIALIZE ACQUISITION STATE
                    if self.counter_vNav == 0:
                        self.acquisition_state = utils.AcquisitionState(self.seg_curr, nav_rot_pred)
                        self.acquisition_state.save_mask(os.path.join(self.subdirs[4], f"mask.nii.gz"))
                    # UPDATE ACQUISITION STATE
                    self.acquisition_state.update(
                        slice_normal_canonical=self.slice_rot_head[:3,2],
                        slice_center_canonical=self.slice_trans_head,
                        slice_thickness=self.slice_thickness
                    )

                    # SAVE FILES FOR DEBUGGING
                    save_nifti(nb.Nifti1Image(img_input.detach().cpu().squeeze().numpy(), proc_aff), "INPUT", self.subdirs[3], hdr.seriesUID, self.counter_vNav)
                    save_nifti(nb.Nifti1Image(self.seg_curr.tensor.float().detach().cpu().squeeze().numpy(), proc_aff), "MASK", self.subdirs[3], hdr.seriesUID, self.counter_vNav)
                    save_npy(nav_FOV_center_prescribe, 'nav_FOV_scanner', self.subdirs[3], hdr.seriesUID, self.counter_vNav)
                    save_npy(self.nav_FOV_center_send, 'nav_FOV_send', self.subdirs[2], hdr.seriesUID, self.counter_vNav)
                    save_npy(nav_rot_pred, 'nav_rot_pred', self.subdirs[3], hdr.seriesUID, self.counter_vNav)
                    save_npy(self.slice_rot_send, 'slice_rot_send', self.subdirs[2], hdr.seriesUID, self.counter_vNav)
                    save_npy(self.slice_trans_send, 'slice_trans_send', self.subdirs[2], hdr.seriesUID, self.counter_vNav)
                    save_nifti(new_ei, imgtype, self.subdirs[0], hdr.seriesUID, self.counter_vNav)

                    # SAVE STATE OUTPUTS
                    self.acquisition_state.save_coverage(os.path.join(self.subdirs[4], f"coverage_{self.counter_vNav:03}.nii.gz"))
                    self.acquisition_state.save_spinhistory(os.path.join(self.subdirs[4], f"spin_history_{self.counter_vNav:03}.nii.gz"))

                    # UPDATE VNAV FRAME
                    self.counter_vNav += 1
                    
                else:
                    print(f"Received slice #{self.counter_HASTE:03}")
                    if hdr.iSliceAnatomicalIndex == 0:
                        self.stack_affine = new_ei.affine
                    if self.no_moco:
                        anatomical_idx = hdr.iSliceAnatomicalIndex
                    else:
                        anatomical_idx = self.anatomical_idx
                    self.stack[anatomical_idx] = new_ei.get_fdata().squeeze()

                    voxel_dims = np.linalg.norm(new_ei.affine[:3,:3], axis=0)
                    new_slice_rot = self.slice_rot_scanner @ np.diag(voxel_dims)
                    new_slice_aff = utils.adjust_slice_t(new_ei, new_slice_rot, self.slice_center_scanner)
                    
                    if not self.no_moco:
                        new_ei = nb.Nifti1Image(new_ei.get_fdata(), new_slice_aff)
                        
                    self.qa = realtime_utils.slice_qa(new_ei, self.seg_curr, self.slice_pos, self.iqa_cnn)[0]
                    self.qas[anatomical_idx] = self.qa
                    save_nifti(new_ei, imgtype, self.subdirs[1], hdr.seriesUID, self.counter_HASTE)
                    self.counter_HASTE += 1


            if len(self.stack) == sum([x is not None for x in self.stack]):
                self.stack = np.stack(self.stack, axis=-1)
                save_nifti(nb.Nifti1Image(self.stack, self.stack_affine), "STACK", self.subdirs[1], hdr.seriesUID, 0)
                save_npy(self.qas, 'slice_QA', self.subdirs[1], hdr.seriesUID, 0)
            if hdr.currentTR + 1 == hdr.totalTR:
                if self.save_4d:
                    self.save_imagestore()
                    self.imagestore = []
                if self.stop_after_one_series:
                    self.stop()
        else:
            self.stop()


class VNavDataSender(Server):

    def __init__(self, host, port, shared_data):
        super().__init__('vNav Data Sender', host, port)
        self.shared_data = shared_data

    def process_data(self, sock):
        time_start = time()
        data = sock.recv(1024)
        if not data: return    	

        res = self.shared_data.get('vNav')
        sendstr = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f' % (res[0])
       
        if len(sendstr):
            print('Sending String: ', sendstr)
            sock.send(str.encode(sendstr))
            print("time send result: %f" % (time() - time_start))
        else:
            print('empty queue')


class SharedData(object):
    def __init__(self):
        self.queues = {'HASTE': Queue(), 'vNav': Queue()}

    def put(self, qname, data):
        self.queues[qname].put(data)

    def get(self, qname):
        return self.queues[qname].get()

    def size(self, qname):
        return self.queues[qname].qsize()


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--host",
                        help="Name of the host to run the image receiver on.",
                        default='192.168.2.5')  #192.168.2.5   #18.25.23.190
    parser.add_argument("-p", "--port", type=int, default="15000",
                        help="Port to run the image receiver on.")
    parser.add_argument("-ph", "--port_send_haste", type=int, default="20248",
                        help="Port to run the data sender on.")
    parser.add_argument("-pv", "--port_send_vnav", type=int, default="25000",
                        help="Port to run the data sender on.")
    parser.add_argument("-d", "--save_directory", default="./outputdata/", #"./outputdata/"
                        help="Directory to save images and labels to.")
    parser.add_argument("-fd", "--four_dimensional", action="store_true")
    parser.add_argument("-ss", "--single_series", action="store_false")
    parser.add_argument("-dm", "--debug_mode", action="store_true")
    parser.add_argument("-nm", "--no_moco", action="store_true")
    parser.add_argument("-usn", "--use_sequence_network", action="store_true")

    # ACQUISITION PARAMETERS
    parser.add_argument("--slice_thickness", type=float, default=3., help="slice thickness in mm")
    parser.add_argument("--TR", type=float, default=2.5, help="TR in seconds (time interval between consecutive slices)")
    parser.add_argument("--orientation", type=str, default="axial", help="desired anatomical orientation (axial, sagittal, coronal)")
    parser.add_argument("--position", type=str, default="SUP", help="patient position in scanner (SUP for supine, LL for left-lateral)")
    parser.add_argument("--n_slices", type=int, default=32, help="number of slices in stack")
    parser.add_argument("--n_interleave", type=int, default=4, help="number of slices to interleave by")

    # NETWORKS
    parser.add_argument("--unet_path", type=str, default="models/unet_128.ckpt", help="path to segmentation U-Net weights")
    parser.add_argument("--e3cnn_path", type=str, default="models/e3cnn_uncertainty.pth", help="path to rotation E(3)-CNN weights")
    parser.add_argument("--transformer_path", type=str, default="", help="path to temporal transformer weights")
    parser.add_argument("--iqa_cnn_path", type=str, default="models/iqa.pth", help="path to 2D IQA CNN weights")

    return parser.parse_args()


def main(argv):
    args = parse_args(argv)

    if args.host == '':
        args.host = socket.gethostbyname(socket.gethostname())  

    shared_data = SharedData()

    servers = []
    servers.append(ImageReceiver(args, shared_data))
    #servers.append(HASTEDataSender(args.host, args.port_send_haste, shared_data))
    servers.append(VNavDataSender(args.host, args.port_send_vnav, shared_data))
    
    for s in servers:
        s.start()

    while True:
    	i = input("enter 'e' to exit\n")
    	if i == 'e':
    		break

    for s in servers:
        s.stop()


if __name__ == "__main__":
    sys.exit(main(sys.argv))