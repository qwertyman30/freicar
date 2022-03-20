import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import cv2
import numpy as np

from visualization_msgs.msg import MarkerArray, Marker

from model import fast_scnn_model
from dataset_helper import freicar_segreg_dataloader
from dataset_helper import color_coder, freicar_segreg_dataloader
#from dataset_helper.run_freicar_segreg_dataloader import visJetColorCoding, TensorImage1ToCV
from birdsEyeT import birdseyeTransformer
import torchvision.transforms.functional as TF

color_conv = color_coder.ColorCoder()
color_coding = color_conv.color_coding

homography_path = './dataset_helper/freicar_homography.yaml'
bst = birdseyeTransformer(homography_path, 3, 3, 200, 2)
birdseye_threshold = 200


def load_model(weights_path=None):
    model = fast_scnn_model.Fast_SCNN(3, 4)
    model = model.cuda()
    
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location='cpu')['state_dict'])
        except KeyError:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print("Weights loaded!")
 
    model.requires_grad_(False)
    model.eval()
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default=None,
                        help='whether to load weights from a checkpoint')
    args = parser.parse_args()
    return args

# added these helper functions here bc of issues with imports
def visJetColorCoding(img):
    color_img = np.zeros(img.shape, dtype=img.dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    return color_img

def TensorImage3ToCV(data):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    return cv


def TensorImage1ToCV(data):
    cv = data.cpu().byte().data.numpy().squeeze()
    return cv


def callback(msg):
    try:
        np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((720, 1280, 3))
        img = TF.to_tensor(np_img)
        img = TF.resize(img, [384, 640])
        img = img.unsqueeze(0).cuda()

        color_conv = color_coder.ColorCoder()
        with torch.no_grad():
            out = model(img.cuda().float())
            seg = out[0]
            reg = out[1].cpu().squeeze(0).squeeze(0).numpy()
            preds = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy()

        seg_color = np.empty((preds.shape[0], preds.shape[1], 3), dtype=np.int8)
        for h_idx, h in enumerate(preds):
            for w_idx, w in enumerate(h):
                seg_color[h_idx][w_idx] = color_coding[preds[h_idx][w_idx]]

        # publish segmentation image
        cv = seg_color.squeeze().astype('uint8')
        # cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
        seg_pub.publish(bridge.cv2_to_imgmsg(cv, "bgr8"))

        # publish regression image
        color_img = np.zeros(reg.shape, dtype=reg.dtype)
        cv2.normalize(reg, color_img, 0, 255, cv2.NORM_MINMAX)
        color_img = color_img.astype(np.uint8)

        birdseye_reg = bst.birdseye(TensorImage1ToCV(out[1]))
        vis_birdseye_reg = visJetColorCoding(birdseye_reg)

        birdseye_reg[birdseye_reg < birdseye_threshold] = 0

        color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
        reg_pub.publish(bridge.cv2_to_imgmsg(color_img, "bgr8"))

        # publish birdseyeview
        # reg_plot = reg.astype('uint8')
        # reg_plot = cv2.applyColorMap(reg_plot, cv2.COLORMAP_JET)
        # reg_bst_plot = bst.birdseye(reg_plot)
        # birdseye_image_pub.publish(bridge.cv2_to_imgmsg(visJetColorCoding(reg_bst_plot), "bgr8"))

        reg_img = bst.birdseye(reg)
        birdseye_reg = reg_img
        birdseye_reg[birdseye_reg < birdseye_threshold] = 0

        # show images with rosrun image_view image_view image:=/reg_image
        markers = MarkerArray()
        # markers.markers = []
        n_samples = 500

        available_pixels = torch.nonzero(torch.from_numpy(birdseye_reg))
        pixel_idx = np.random.choice(np.arange(len(available_pixels)), size=n_samples, replace=True, p=None)

        if available_pixels.shape[0] >= n_samples:
            for i, sample in enumerate(available_pixels[pixel_idx]):
                marker = Marker()
                marker.id = i
                marker.header.frame_id = "freicar_1/base_link"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.003
                marker.color.a = 1.0
                marker.color.g = 1.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.y = (sample[1] / 140.0) - (640.0 / 300.0)
                marker.pose.position.x = (sample[0] / 140.0) - 0.7
                """
                if (sample[1] / 100.0) - (640.0 / 200.0) +0.18 <0:
                    print(1)
                    marker.pose.position.y = (sample[1] / 100.0) - (640.0 / 200.0) + 0.18
                    marker.pose.position.x = (sample[0] / 100.0) - 1.5
                else:
                    print(2)
                    marker.pose.position.y = (sample[1] / 100.0) - (640.0 / 200.0) - 0.18
                    marker.pose.position.x = (sample[0] / 100.0) - 1.5
                """
                marker.pose.position.z = 0
                markers.markers.append(marker)
                # markers.markers[i] = marker

            birdseye_publisher.publish(markers)


    except CvBridgeError as e:
        print(e)


def rgb_img_listener():
    rospy.init_node('rgb_img_listener')
    rgb_img_topic = '/freicar_1/sim/camera/rgb/front/image'

    rospy.Subscriber(rgb_img_topic, Image, callback)
    rospy.spin()


if __name__ == '__main__':
    # 'logs/freicar-detection/efficientdet-d0_99_109100.pth'
    args = get_args()
    bridge = CvBridge()
    model = load_model('saved_models/model_49.pth')
    seg_pub = rospy.Publisher('seg_image', Image, queue_size=10)
    reg_pub = rospy.Publisher('reg_image', Image, queue_size=10)
    # birdseye_image_pub = rospy.Publisher('birdseye_image', Image, queue_size=10)
    birdseye_publisher = rospy.Publisher('birdseye', MarkerArray, queue_size=10)
    while not rospy.is_shutdown():
        rgb_img_listener()
