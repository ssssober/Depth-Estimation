# StereoNet-pytorch
> This work only provides the source code implementation of the paper **[StereoNet:Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction(ECCV2018)](https://www.researchgate.net/publication/326495585_StereoNet_Guided_Hierarchical_Refinement_for_Real-Time_Edge-Aware_Depth_Prediction)** using pytorch. Basically followed the algorithm structure of the original paper, because of the limited ability of myself. I suggest you'd better not use `edge_refinement`, if you do experiments on speckle images.
 

## Dependencies
+ PyTorch (0.4.0+)
+ Python (3.5.0+)
+ cuda-toolkit(9.0+)
+ torchvision (0.2.0+)
+ tensorboard (1.6.0)
+ pillow


## Acknowledgement
This work is mainly inspired by papers: **[StereoNet](https://www.researchgate.net/publication/326495585_StereoNet_Guided_Hierarchical_Refinement_for_Real-Time_Edge-Aware_Depth_Prediction)**, **[Connecting_the_Dots](https://github.com/autonomousvision/connecting_the_dots)**, **[ActiveStereoNet](http://asn.cs.princeton.edu/)**, **[PSMNet](https://github.com/JiaRenChang/PSMNet)** and **[GwcNet](https://github.com/xy-guo/GwcNet)** and excellent open source project: **[StereoNet-ActiveStereoNet](https://github.com/meteorshowers/StereoNet-ActiveStereoNet)**.
