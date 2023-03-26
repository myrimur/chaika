import torch
from torchvision.io import read_video, write_video
from torchvision import transforms

import matplotlib.pyplot as plt

from AdaBins.models import UnetAdaptiveBins
import AdaBins.model_io as model_io

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256

cmap = plt.get_cmap('coolwarm')
norm = plt.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH_KITTI)

# crop = torch.jit.script(
#     torch.nn.Sequential(
#         transforms.CenterCrop((480, 640)),
#     )
# )

crop = transforms.Compose([
    transforms.CenterCrop((480, 640)),
])

# normalize = transforms.Compose([
#     transforms.Normalize(mean=0.5, std=0.5)
# ])

frames, _, _ = read_video('vid.mp4', start_pts=0, end_pts=5, pts_unit='sec', output_format="TCHW")
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "AdaBins/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

depths = torch.empty(len(frames), 240, 320, 3)
for idx, frame in enumerate(frames):
    print(idx)
    frame = crop(frame)
    example_rgb_batch = frame.unsqueeze(0).float().to('cpu')
    _, predicted_depth = model(example_rgb_batch)

    # print(torch.clamp(predicted_depth, min=0, max=1))
    # print(normalize(predicted_depth))

    colors = cmap(norm(predicted_depth.detach().numpy().flatten()))[:,:3] * 255

    frame_depth = torch.tensor(colors, dtype=torch.uint8)
    frame_depth = torch.reshape(frame_depth, (240, 320, 3))

    depths[idx] = frame_depth


write_video('result.mp4', depths, fps=20)
