import torch
from torchvision.io import read_video, write_video
from torchvision import transforms

from AdaBins.models import UnetAdaptiveBins
import AdaBins.model_io as model_io

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256

crop = transforms.Compose([
    transforms.CenterCrop((480, 640)),
])

normalize = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5)
])

frames, _, _ = read_video(str('vid.mp4'), output_format="TCHW")
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "AdaBins/pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

depths = torch.Tensor(len(frames), 1, 480, 640)
for frame in frames:
    frame = crop(frame)
    example_rgb_batch = frame.unsqueeze(0).float().to('cpu')
    _, predicted_depth = model(example_rgb_batch)

    # TODO: normalize predicted depth and convert to tensor of uint8 to save it as a video
