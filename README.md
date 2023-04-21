# Chaika
Is it a plane? Is it a bird? Actually, yes, its chaika.

Pull submodules: `git submodule update --init --recursive`

You have to [build](https://github.com/stevenlovegrove/Pangolin#building) Pangolin and generate bindings for python

Monodepth2 downloads weights automatically, but if you use AdaBins, you need to download pretrained model [AdaBins_kitti.pt](https://drive.google.com/file/d/1HMgff-FV6qw1L0ywQZJ7ECa9VPq1bIoj/view?usp=sharing) and put it in the folder `AdaBins/pretrained`

| AdaBins                       | Monodepth2                      |
|-------------------------------|---------------------------------|
| ![](demo/ada_bins_result.gif) | ![](demo/monodepth2_result.gif) |

![demo.gif](demo/demo.gif)
