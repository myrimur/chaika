import math

import torch
from torchvision.io import write_video, VideoReader
from torchvision import transforms

import matplotlib.pyplot as plt
import itertools

from AdaBins.models import UnetAdaptiveBins
import AdaBins.model_io as model_io

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

N_BINS = 256

cmap = plt.get_cmap('plasma')
norm = plt.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH)

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
# duration = reader.get_metadata()['video']['duration'][0]
duration = 3
frames = math.ceil(duration * fps)

model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH)
pretrained_path = "AdaBins/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

depths = torch.empty(frames, 240, 320, 3)

for idx, frame in enumerate(itertools.takewhile(lambda x: x['pts'] <= duration, reader)):
    print(idx)

    data = frame['data']
    data = crop(data)
    example_rgb_batch = data.unsqueeze(0).float().to('cpu')
    _, predicted_depth = model(example_rgb_batch)

    colors = cmap(norm(predicted_depth.detach().numpy().flatten()))[:,:3] * 255

    frame_depth = torch.tensor(colors, dtype=torch.uint8)
    frame_depth = torch.reshape(frame_depth, (240, 320, 3))

    depths[idx] = frame_depth

write_video('ada_bins_result.mp4', depths, fps=fps)
