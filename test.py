import torch
from torchvision.io import read_video, write_video
from torchvision import transforms

import matplotlib as mpl

from AdaBins.models import UnetAdaptiveBins
import AdaBins.model_io as model_io

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256

cmap = mpl.cm.coolwarm
norm = mpl.colors.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH_KITTI)

# crop = torch.jit.script(
#     torch.nn.Sequential(
#         transforms.CenterCrop((480, 640)),
#     )
# )

crop = transforms.Compose([
    transforms.CenterCrop((480, 640)),
])

normalize = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5)
])

frames, _, _ = read_video('vid.mp4', start_pts=0, end_pts=0.5, pts_unit='sec', output_format="TCHW")
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "AdaBins/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

FRAMES = 3

depths = torch.empty(FRAMES, 240, 320, 3)
for idx, frame in enumerate(frames[:FRAMES]):
    frame = crop(frame)
    example_rgb_batch = frame.unsqueeze(0).float().to('cpu')
    _, predicted_depth = model(example_rgb_batch)
    print(idx)

    # print(torch.clamp(predicted_depth, min=0, max=1))
    # print(normalize(predicted_depth))

    # flattened = predicted_depth.view(predicted_depth.shape[0], -1, 1, 1)
    # min_depth, _ = torch.min(flattened, dim=1, keepdim=True)
    # max_depth, _ = torch.max(flattened, dim=1, keepdim=True)

    unit = (predicted_depth - MIN_DEPTH) / (MAX_DEPTH_KITTI - predicted_depth)
    a = torch.reshape(unit, (1, 240, 320, 1)).repeat(1, 1, 1, 3)

    # print(a.shape)
    print(a)
    #
    # print(a[0].shape)
    depths[idx] = a[0]

    # break

write_video('result.mp4', depths, fps=1)

    # TODO: normalize predicted depth and convert to tensor of uint8 to save it as a video

