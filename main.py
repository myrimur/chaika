import math

import torch
from torchvision.io import write_video, VideoReader
from torchvision import transforms

import matplotlib.pyplot as plt
import itertools

import cv2 as cv



reader = VideoReader('vid.mp4', 'video')

fps = reader.get_metadata()['video']['fps'][0]
duration = 3
frames = math.ceil(duration * fps)

depths = torch.empty(frames, 480, 640, 3)



for idx, (frame_1, frame_2) in enumerate(itertools.pairwise(itertools.islice(reader, frames))):
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
    img3 = cv.drawMatches(img_1, kp1, img_2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imwrite('test.jpg', img3)


    break
