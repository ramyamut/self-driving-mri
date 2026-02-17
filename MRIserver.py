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

RAS2LPS = np.diag(np.array([-1.,-1.,1.]))
LPS2SDCS = {
    "SUP": np.diag(np.array([1.,-1.,-1.])), # supine
    "LL": np.array([
        [0., -1., 0.],
        [-1., 0., 0.],
        [0., 0., -1]
    ]), # left lateral
}


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
    filename = os.path.join(save_location,
            '%s-%s-%05d.nii.gz' % (imgtype, uid.decode(), index))
    img.to_filename(filename)

# img: nibabel object Nifti1Image
def save_npy(img_arr, imgtype, save_location, uid, index):
    filename = os.path.join(save_location,
            '%s-%s-%05d.npy' % (imgtype, uid.decode(), index))
    np.save(filename, img_arr)


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

        self._is_running = None
        self._server = None
        self.imagestore = []
        self.save_location = args.save_directory
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
        self.slice_thickness = 3.
        self.ori = 'axial'
        self.patient_position = 'SUP'
        self.num_slices = 25
        self.slice_zs_ordered = np.arange(0,self.num_slices)*self.slice_thickness # - (self.slice_thickness-1)/2
        self.slice_zs_ordered -= self.slice_zs_ordered.mean()
        self.slice_zs = []
        self.anatomical_idxs = []
        self.n_sweeps = 4
        self.vnav0 = None
        for sweep in range(self.n_sweeps):
            self.anatomical_idxs += list(range(sweep, self.num_slices, self.n_sweeps))
            self.slice_zs += self.slice_zs_ordered[list(range(sweep, self.num_slices, self.n_sweeps))].tolist()
        print(self.anatomical_idxs)
        print(self.slice_zs)

        # init networks
        unet_path = "models/unet_128.ckpt"
        self.unet = realtime_utils.init_unet(unet_path)
        e3cnn_path = "models/e3cnn.pth"
        self.e3cnn = realtime_utils.init_e3cnn(e3cnn_path)
        self.vnav_t_send = np.array([0., 0., 0.])
        for _ in range(5):
            start1 = time()
            dummy_img = nb.Nifti1Image(np.random.uniform(size=(54,54,20)), np.diag([6,6,6,1]))
            _, img_input, pred_seg, _ = realtime_utils.run_unet(dummy_img, self.unet)
            _ = realtime_utils.run_e3cnn(img_input, pred_seg, dummy_img.affine, self.e3cnn, 0, 'axial', dummy_img.affine[:3,3], LPS2SDCS[self.patient_position]@RAS2LPS)
            end2 = time()
            print(f"Total time: {end2-start1}")
    
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
                    if self.vnav0 is None:
                        self.vnav0 = new_ei
                    nav_FOV_center_curr = new_ei.affine[:3,3] - self.vnav0.affine[:3,3]
                    pred_trans, img_input, pred_seg, proc_aff = realtime_utils.run_unet(new_ei, self.unet)
                    nav_FOV_center_prescribe = nav_FOV_center_curr + pred_trans
                    self.nav_FOV_center_send = LPS2SDCS[self.patient_position]@RAS2LPS@nav_FOV_center_prescribe
                    clip = 199.9
                    if np.linalg.norm(self.nav_FOV_center_send) > clip:
                        self.nav_FOV_center_send = self.nav_FOV_center_send / np.linalg.norm(self.nav_FOV_center_send) * clip
                    
                    self.slice_rot_send, self.slice_trans_send, nav_rot_pred = realtime_utils.run_e3cnn(img_input, pred_seg, new_ei.affine, self.e3cnn, self.slice_pos, self.ori, nav_FOV_center_curr, LPS2SDCS[self.patient_position]@RAS2LPS)
                    self.qa = 1.0
                    result = tuple(self.nav_FOV_center_send.tolist() + self.slice_trans_send.tolist() + self.slice_rot_send.flatten().tolist() + [self.qa])
                    self.shared_data.put("vNav", (result, self.counter_vNav))
                    self.counter_vNav += 1
                    
                    # print(f"HASTE ROT SEND: {self.haste_rot_send}")

                    save_nifti(nb.Nifti1Image(img_input.detach().cpu().squeeze().numpy(), proc_aff), "INPUT", self.save_location, hdr.seriesUID, self.counter_vNav)
                    save_nifti(nb.Nifti1Image(pred_seg.float().detach().cpu().squeeze().numpy(), proc_aff), "MASK", self.save_location, hdr.seriesUID, self.counter_vNav)
                    save_npy(nav_FOV_center_prescribe, 'nav_FOV_scanner', self.save_location, hdr.seriesUID, self.counter_vNav)
                    save_npy(self.nav_FOV_center_send, 'nav_FOV_send', self.save_location, hdr.seriesUID, self.counter_vNav)
                    save_npy(nav_rot_pred, 'nav_rot_pred', self.save_location, hdr.seriesUID, self.counter_vNav)
                    save_npy(self.slice_rot_send, 'slice_rot_send', self.save_location, hdr.seriesUID, self.counter_vNav)
                    save_npy(self.slice_trans_send, 'slice_trans_send', self.save_location, hdr.seriesUID, self.counter_vNav)
                    
                else:
                    if hdr.iSliceAnatomicalIndex == 0:
                        self.stack_affine = new_ei.affine
                        # print(self.stack_affine)
                    self.stack[hdr.iSliceAnatomicalIndex] = new_ei.get_fdata().squeeze()
                    self.counter_HASTE += 1

                    voxel_res = np.diag([1.25,1.25,3])
                    prescribed_aff = new_ei.affine
                    new_slice_rot, new_slice_center = utils.vsend_to_scanner(self.slice_rot_send, self.slice_trans_send, np.copy(prescribed_aff[:3,:3]), voxel_res, LPS2SDCS[self.patient_position]@RAS2LPS)
                    new_slice_rot = new_slice_rot @ voxel_res
                    new_slice_aff = utils.adjust_slice_t(new_ei, new_slice_rot, new_slice_center)
                    new_ei = nb.Nifti1Image(new_ei.get_fdata(), new_slice_aff)
                save_nifti(new_ei, imgtype, self.save_location, hdr.seriesUID, self.counter_vNav)
                # save_npy(new_ei.get_fdata(), imgtype, self.save_location, hdr.seriesUID, self.counter_vNav)


            if len(self.stack) == sum([x is not None for x in self.stack]):
                self.stack = np.stack(self.stack, axis=-1)
                save_nifti(nb.Nifti1Image(self.stack, self.stack_affine), "STACK", self.save_location, hdr.seriesUID, self.counter_vNav)
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
            # print('Sending String: ', sendstr)
            sock.send(str.encode(sendstr))
            # print("time send result: %f" % (time() - time_start))
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