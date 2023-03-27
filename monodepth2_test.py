import math

import torch
from torchvision.io import write_video, VideoReader
from torchvision import transforms

import matplotlib.pyplot as plt
import itertools

import sys
sys.path.insert(0, "monodepth2/")

import os

import monodepth2.networks as networks
from monodepth2.utils import download_model_if_doesnt_exist
from monodepth2.layers import disp_to_depth


MIN_DEPTH = 0.1
MAX_DEPTH = 100

STEREO_SCALE_FACTOR = 5.4

cmap = plt.get_cmap('plasma')
norm = plt.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH, clip=True)


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


# crop = torch.jit.script(
#     torch.nn.Sequential(
#         transforms.CenterCrop((480, 640)),
#     )
# )

crop = transforms.Compose([
    transforms.CenterCrop((480, 640)),
])

reader = VideoReader('vid.mp4', 'video')

fps = reader.get_metadata()['video']['fps'][0]
duration = 3
frames = duration * math.ceil(fps)

depths = torch.empty(frames, 480, 640, 3)

for idx, frame in enumerate(itertools.islice(reader, frames)):
    print(idx)
    data = frame['data'].float() / 255.0
    data = crop(data).unsqueeze(0).float().to('cpu')

    with torch.no_grad():
        features = encoder(data)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)[-1] * STEREO_SCALE_FACTOR
    print(torch.max(disp), torch.min(disp))

    colors = cmap(norm(disp.detach().numpy().flatten()))[:,:3] * 255

    frame_depth = torch.tensor(colors, dtype=torch.uint8)
    frame_depth = torch.reshape(frame_depth, (480, 640, 3))

    depths[idx] = frame_depth

write_video('monodepth2_result.mp4', depths, fps=fps)
