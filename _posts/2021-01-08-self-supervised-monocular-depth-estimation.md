---
layout: post
title: "Self-Supervised Monocular Depth Estimation"
date: 2021-01-08 03:32:44
image: '/assets/img/monocular-depth-estimation/final_demo.png'
description: 'Explorations on self-supervised monocular depth estimation'
tags:
- Computer Vision
- Machine Learning
- PyTorch
- Monocular Depth Estimation
- CS484
categories:
- Computer Vision
---

In October of 2020, I learned of strategy to use monocular depth estimation as a cheaper low-accuracy alternative to LIDAR. Essentially, you can use a camera and a self-supervised monocular depth estimation network to produce colour images with depth information ([RGBD images](http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html)). Since we know camera's exact position and orientation, each pixel in the RGBD image can be projected into 3d to form a pseudo-pointcloud with RGB+XY information. This is similar to how the [Xbox Kinect](https://en.wikipedia.org/wiki/Kinect) obtains 3d information, though the Kinect uses infrared sensors to gauge depth instead of a RGBD image. With this pseudo-pointcloud, we can use Simultaneous Localization and Mapping (SLAM) or Occupancy Grid algorithms to produce a 3d map of our environment with colour information. 

![RGBD mapping with the XBOX Kinect](https://www-cse-managed-files.s3.amazonaws.com/research_projects/map_full_overview_1-320.jpg)
[Above: RGBD mapping with the XBOX Kinect](https://www.cs.washington.edu/research-projects/robotics/rgbd-mapping)

At the same time, I was taking [CS484 (Computational Vision)](https://cs.uwaterloo.ca/~yboykov/Courses/cs484/) taught by Professor Yuri Boykov. CS484 provides a very in-depth look into classical computer vision and focuses on camera geometry (pinhole camera model, projections and transformations, and epipolar geometry). Conveniently, just a couple weeks after I heard of the concept, we learned the core concepts behind self-supervised monocular depth estimation. The main paper we focused on was Godard et al.'s 2017 paper titled [*Unsupervised Monocular Depth Estimation with Left-Right Consistency*](https://arxiv.org/pdf/1609.03677v3.pdf). For my CS484 final project, I wanted to minimally reproduce the results in the paper while clearly explaining in a step-by-step manner exactly how the model functions. You can find my project [on my github](https://github.com/charlesyz/MonocularDepthEstimation). 

## Dense Stereo Reconstruction

First, some context. It's pretty easy to compute depth information without a neural network given two camera images where the cameras have known position. This is called [Stereo Reconstruction](https://cs.uwaterloo.ca/~yboykov/Courses/cs484_2018/Lectures/lec08_stereo_u.pdf). Assuming either the cameras are perfectly parallel or the images have been [rectified](https://en.wikipedia.org/wiki/Image_rectification), points along a horizontal scan-line in the left image will correspond to points along the horizontal scan-line in the right image. Since we have two cameras at varying positions, near objects will have a larger disparity between the left and right images than far objects. The depth of a certain pixel in question is then directly proportional to this disparity value. If your cameras are parallel (no rectification) and `F = focal length (m), B = baseline distance (m), d = disparity (pixels)`, then `depth = (b * f) / d`   ([source](https://arxiv.org/pdf/1609.03677v3.pdf)). 

![Scanline matching in rectified images](/assets/img/monocular-depth-estimation/scanline.png)

![Disparity of elements in the scene](/assets/img/monocular-depth-estimation/disparity.png)

But, there are some issues with this approach. First of all, some pixels are visible in one image but occluded in the other. This is a loss of information. Secondly, disparity values are calculated on a scanline-by-scanline basis, so there is no guarantee that scanlines next to each other will have consistent depths. This leads to "Streaking" artifacts in the final disparity map. Thirdly, it's really hard to tell the disparity level of areas with uniform texture. Some of these issues (such as the streaking) can be resolved by enforcing multi-scanline consistency using [regularization and graph-cuts](https://www.researchgate.net/publication/221787297_Stereo_Matching_and_Graph_Cuts).

![Scan line vs graph cuts](/assets/img/monocular-depth-estimation/graph-cuts.png)

## Self-supervised monocular depth estimation

Machine learning can do *Much* better. By leveraging past experiences, humans can extract pretty good depth information using only one eye or a single image. Deep learning can do the same thing. However, humans have years and years of experience to learn these depth queues. in contrast, it's very difficult to get datasets with depth information large enough to train a supervised depth estimation network. Each pixel in each image needs to be classified with the correct disparity, which is really expensive. To solve this issue, we remove the requirement for accurate ground truth, and instead create a self-supervised network. 

![Training pipeline](https://raw.githubusercontent.com/charlesyz/MonocularDepthEstimation/master/images/monodepth.png)

[Godard et al.](https://arxiv.org/pdf/1609.03677v3.pdf)'s primary contribution is a novel loss-function for self-supervised monocular depth estimation that enforces left-right photo consistency. During training, we use left-right image pair taken by a set of parallel cameras with a known base-length. [The KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php) has over 29,000 such left-right image pairs. The goal of the network is to use one of the images (in this case, the left image) to estimate two disparity maps. One disparity map contains the disparity values from the left image to the right image, and the other contains the disparity values from the right image to the left image. Note that the left-to-right disparity map allows us to reconstruct the right image using the left image by applying the appropriate disparity values. Similarly, the right-to-left disparity map allows us to reconstruct the left image using the right image. These "reconstructed images" can be compared with the original images to evaluate the accuracy of the disparity map. This comparison is the main component of the loss function. Using both the reconstructed left and right images allows the network to enforce consistency between it's predictions. This reduces artifacts in the disparity maps that would occur if we only used one reconstructed image (instead of two). 

![Model Architecture](https://raw.githubusercontent.com/charlesyz/MonocularDepthEstimation/master/images/model.png)

The model itself is actually pretty straight forwards, a basic convolutional encoder-decoder architecture using the ResNet encoder and a lot of skip connections for resolution. 

For brevity, I've left out a bunch of the other details and improvements that Godard et al. made. See [my project on github a more complete demo and explanation](https://github.com/charlesyz/MonocularDepthEstimation) or [read the paper here](https://arxiv.org/pdf/1609.03677v3.pdf).

## Results

After training for 10 epochs on a small subset of the KITTI dataset (5268 left/right image pairs), I achieved the following results. Note that yellow = high disparity, blue = low disparity. These are some pretty good results for such little training! You can clearly see that the disparities are roughly correct and the vanishing point of the street is accurately represented. However, the output disparity map is pretty fuzzy, and the left disparity map has sharp lines on edge boundaries while the right disparity map does not. Some of the disparities are also wrong or inverted (for example, model predicts that the white banners on the right are inset into the wall instead of protruding).

![final demo](/assets/img/monocular-depth-estimation/final_demo.png)

These issues would likely be solved by more training time on a larger dataset. Training for 50 epochs over the entire KITTI dataset (~29,000 left-right pairs), the authors were able to achieve amazing results.

![monodepth](https://camo.githubusercontent.com/347a28083896fc6b18f12e29933fb7adc3ebfa485ee383897c59fe4a0983f97e/687474703a2f2f76697375616c2e63732e75636c2e61632e756b2f707562732f6d6f6e6f44657074682f6d6f6e6f64657074685f7465617365722e676966)

CS484 and this project sparked my interest in the creative ways that computer vision can be applied for autonomous driving. There's a lot of mathematics and geometry behind the classic computer vision approaches, and I'm starting to believe that a combined approach using both classic geometric computer vision with new deep neural networks is the way forward. 

Readings:
1. *Unsupervised Monocular Depth Estimation with Left-Right Consistency*: [https://arxiv.org/pdf/1609.03677v3.pdf](https://arxiv.org/pdf/1609.03677v3.pdf)
2. Open3d RGBD Images: [http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html](http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html)
3. KITTI Dataset: [http://www.cvlibs.net/datasets/kitti/index.php](http://www.cvlibs.net/datasets/kitti/index.php)
4. MonoDepth repository: [https://github.com/mrharicot/monodepth](https://github.com/mrharicot/monodepth)
5. CS484 Course Notes for Dense Stereo Reconstruction: [https://cs.uwaterloo.ca/~yboykov/Courses/cs484_2018/Lectures/lec08_stereo_u.pdf](https://cs.uwaterloo.ca/~yboykov/Courses/cs484_2018/Lectures/lec08_stereo_u.pdf)