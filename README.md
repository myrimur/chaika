# Chaika
Is it a plane? Is it a bird? Actually, yes, its chaika.

Pull submodules: `git submodule update --init --recursive`

You have to [build](https://github.com/stevenlovegrove/Pangolin#building) Pangolin and generate bindings for python

Monodepth2 downloads weights automatically, but if you use AdaBins, you need to download pretrained model [AdaBins_kitti.pt](https://drive.google.com/file/d/1HMgff-FV6qw1L0ywQZJ7ECa9VPq1bIoj/view?usp=sharing) and put it in the folder `AdaBins/pretrained`

| AdaBins                                                        | Monodepth2                      |
|----------------------------------------------------------------|---------------------------------|
| <img src="demo/ada_bins_result.gif" width="420" height="340"/> | <img src="demo/monodepth2_result.gif" width="420" height="340"/> |

[//]: # (<img src="demo/demo.gif" width="1000" height="800"/>)

To run, download KITTY dataset [here](https://drive.google.com/file/d/1MO8aG4itpotpIy_VgWVToQPDYXCi57Fu/view?usp=sharing), unzip and put it in the folder `data/` without changing the folders structure.

![](demo/main.gif)