import math

import torch
from torchvision.io import write_video, VideoReader
from torchvision import transforms

import matplotlib.pyplot as plt
import itertools

import cv2 as cv

import numpy as np

import sys
sys.path.insert(0, "monodepth2/")

import os

import monodepth2.networks as networks
from monodepth2.utils import download_model_if_doesnt_exist
from monodepth2.layers import disp_to_depth

MIN_DEPTH = 0.1
MAX_DEPTH = 100

STEREO_SCALE_FACTOR = 5.4

K = np.array([[520.9, 0, 325.1],
                [0, 521, 249.7],
                [0, 0, 1]], dtype=np.float32)

crop = transforms.Compose([
    transforms.CenterCrop((480, 640)),
])


reader = VideoReader('vid.mp4', 'video')

fps = reader.get_metadata()['video']['fps'][0]
duration = 3
frames = math.ceil(duration * fps)

depths = torch.empty(frames, 480, 640, 3)


model_name = "mono+stereo_640x192"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()


def calc_depth(frame):
    data = frame['data'].float() / 255.0
    data = crop(data).unsqueeze(0).float().to('cpu')

    with torch.no_grad():
        features = encoder(data)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)[-1] * STEREO_SCALE_FACTOR

    return torch.reshape(disp[0], (480, 640))


def pixel_to_camera(point):
    return np.array([point[0] - K[0, 2] / K[0, 0],
                     point[1] - K[1, 2] / K[1, 1]])

points_3d_1 = []
points_3d_2 = []


for idx, (frame_1, frame_2) in enumerate(itertools.pairwise(itertools.islice(reader, frames))):
    print(frame_1['data'].shape)
    height_ratio = frame_1['data'].shape[1] / 480.0
    weight_ratio = frame_1['data'].shape[2] / 640.0
    print(height_ratio, weight_ratio)

    img_1 = frame_1['data'].numpy().transpose(1, 2, 0)
    img_2 = frame_2['data'].numpy().transpose(1, 2, 0)

    img_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(img_1, None)
    kp2, des2 = orb.detectAndCompute(img_2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # img3 = cv.drawMatches(img_1, kp1, img_2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv.imwrite('test.jpg', img3)


    depth = calc_depth(frame_1)
    print(depth.shape)

    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt

        d_1 = depth[int(p1[1] / height_ratio), int(p1[0] / weight_ratio)]
        d_2 = depth[int(p2[1] / height_ratio), int(p2[0] / weight_ratio)]

        p1 = d_1 * pixel_to_camera(p1)
        p2 = d_2 * pixel_to_camera(p2)

        points_3d_1.append(np.array([p1[0], p1[1], d_1]))
        points_3d_2.append(np.array([p2[0], p2[1], d_2]))

    print(points_3d_1)

    break
