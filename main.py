import math

import torch
from torchvision.io import write_video, VideoReader, read_image, ImageReadMode, write_png
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

CROP_DIMS = (192, 640)


K = np.array([[718.856, 0, 607.1928 * CROP_DIMS[-1] / 1241],
              [0, 718.856, 185.2157 * CROP_DIMS[0]  / 376],
              [0, 0, 1]])

print(K)

crop = transforms.Compose([
    transforms.CenterCrop(CROP_DIMS),
])


reader = VideoReader('vid.mp4', 'video')

fps = reader.get_metadata()['video']['fps'][0]
duration = 3
frames = math.ceil(duration * fps)

depths = torch.empty(frames, *CROP_DIMS, 3)


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
    data = frame.float() / 255.0
    data = data.unsqueeze(0).float().to('cpu')

    with torch.no_grad():
        features = encoder(data)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)[-1] * STEREO_SCALE_FACTOR

    return torch.reshape(depth[0], CROP_DIMS)


def pixel_to_camera(point):
    return np.array([(point[0] - K[0, 2]) / K[0, 0],
                     (point[1] - K[1, 2]) / K[1, 1]])


points_3d_1 = []
points_3d_2 = []


# for idx, (frame_1, frame_2) in enumerate(itertools.pairwise(itertools.islice(reader, frames))):
#     frame_1 = crop(frame_1['data'])
#     frame_2 = crop(frame_2['data'])

def t():
    frame_1 = crop(read_image("000001.png", ImageReadMode.RGB))
    frame_2 = crop(read_image("000002.png", ImageReadMode.RGB))

    img_1 = frame_1.numpy().transpose(1, 2, 0)
    img_2 = frame_2.numpy().transpose(1, 2, 0)

    img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()

    kp_1, des_1 = orb.detectAndCompute(img_1, None)
    kp_2, des_2 = orb.detectAndCompute(img_2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)

    depth_1 = calc_depth(frame_1)
    depth_2 = calc_depth(frame_2)

    for match in matches:
        p_1 = kp_1[match.queryIdx].pt
        p_2 = kp_2[match.trainIdx].pt

        d_1 = depth_1[int(p_1[1]), int(p_1[0])]
        d_2 = depth_2[int(p_2[1]), int(p_2[0])]

        p_1 = d_1 * pixel_to_camera(p_1)
        p_2 = d_2 * pixel_to_camera(p_2)

        points_3d_1.append(np.array([p_1[0], p_1[1], d_1]))
        points_3d_2.append(np.array([p_2[0], p_2[1], d_2]))

    # break

t()