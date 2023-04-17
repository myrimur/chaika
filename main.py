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

from visualization import *
from kitti import *

NUM_VISUALIZED_KEYPOINTS = 10

MIN_DEPTH = 0.1
MAX_DEPTH = 100

STEREO_SCALE_FACTOR = 5.4

CROP_DIMS = (192, 640)

K = P_rect_02
K[0] *= CROP_DIMS[1]
K[1] *= CROP_DIMS[0]

crop = transforms.Compose([
    transforms.CenterCrop(CROP_DIMS),
])

reader = VideoReader('ucu_UsuiqF6T.mp4', 'video')

fps = reader.get_metadata()['video']['fps'][0]
duration = 19
frames = math.floor(duration * fps)

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


data = KittiRaw('data/2011_09_26_drive_0009_sync/', transform=crop)


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

# """"
trajectory = [np.eye(4)]
points = {}
kp_prev, des_prev = None, None
for idx, (frame_1, frame_2) in enumerate(itertools.pairwise(itertools.islice(data, len(data)))):
    print(idx)

    # frame_1 = crop(frame_1['data'])
    # frame_2 = crop(frame_2['data'])

    points_2d_1 = []
    points_2d_2 = []

    points_3d_1 = []
    points_3d_2 = []

    # frame_1 = crop(read_image("000001.png", ImageReadMode.RGB))
    # frame_2 = crop(read_image("000002.png", ImageReadMode.RGB))

    img_1 = frame_1.numpy().transpose(1, 2, 0)
    img_2 = frame_2.numpy().transpose(1, 2, 0)

    img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()

    if kp_prev is None:
        kp_1, des_1 = orb.detectAndCompute(img_1, None)
    else:
        kp_1, des_1 = kp_prev, des_prev
    kp_2, des_2 = orb.detectAndCompute(img_2, None)
    kp_prev, des_prev = kp_2, des_2

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)

    depth_1 = calc_depth(frame_1)
    depth_2 = calc_depth(frame_2)

    for match in matches:
        p_1 = kp_1[match.queryIdx].pt
        p_2 = kp_2[match.trainIdx].pt

        points_2d_1.append(p_1)
        points_2d_2.append(p_2)

        d_1 = depth_1[int(p_1[1]), int(p_1[0])]
        d_2 = depth_2[int(p_2[1]), int(p_2[0])]

        p_1 = d_1 * pixel_to_camera(p_1)
        p_2 = d_2 * pixel_to_camera(p_2)

        points_3d_1.append(np.array([p_1[0], p_1[1], d_1]))
        points_3d_2.append(np.array([p_2[0], p_2[1], d_2]))

    _, rvec, tvec, _ = cv.solvePnPRansac(np.array(points_3d_2), np.array(points_2d_1), K, None)
    rvec = rvec.flatten()
    tvec = tvec.flatten()

    R, _ = cv.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec

    T = trajectory[idx] @ T

    trajectory.append(T)

    points_3d_2 = np.array(points_3d_2[:NUM_VISUALIZED_KEYPOINTS])

    points_hom = np.hstack((points_3d_2, np.ones((points_3d_2.shape[0], 1))))
    points_transformed_hom = np.dot(T, points_hom.T).T
    points_transformed = points_transformed_hom[:, :3] / points_transformed_hom[:, 3:]

    for i, match in enumerate(matches[:NUM_VISUALIZED_KEYPOINTS]):
        des = des_2[match.trainIdx]
        points[des.tobytes()] = points_transformed[i]

    # show_point_cloud(points_3d_2 + points_transformed.tolist())

# plot_trajectory(trajectory)
show_point_cloud_and_trajectory(points.values(), trajectory)
# """
# cloud = PointCloud()
# cloud.add_points([[1, 1, 1]])
# cloud.run()
# cloud.add_points([[10, 10, 10]])
