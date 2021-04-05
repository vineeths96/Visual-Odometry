 ![Language](https://img.shields.io/badge/language-python--3.8-blue) [![Contributors][contributors-shield]][contributors-url] [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Issues][issues-shield]][issues-url] [![MIT License][license-shield]][license-url] [![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <h3 align="center">Visual Odometry</h3>
  <p align="center">
    5-point algorithm and 8-point algorithm method
    <br />
    <a href=https://github.com/vineeths96/Visual-Odometry><strong>Explore the repository»</strong></a>
    <br />
    <br />
    <a href=https://github.com/vineeths96/Visual-Odometry/blob/master/Problem_Statement.pdf>View Problem Statement</a>
    <a href=https://github.com/vineeths96/Visual-Odometry/blob/master/results/report.pdf>View Report</a>
  </p>



</p>

> tags : visual odometry, 5-point algorithm, 8-point algorithm, fast, sift essential matrix, digital video



<!-- ABOUT THE PROJECT -->

## About The Project

This project deals with the task of Visual Odometry using Nister’s five point algorithm and eight point algorithm for essential matrix estimation. We develop our own implementations for these methods. We implement RANSAC along with these methods for outlier rejection. We test our implementations and OpenCV implementations on a couple of sequences from KITTI dataset. We specifically use Sequences 03 and 10, since they are of relatively smaller size. We use FAST feature detection algorithm to detect the keypoint locations and keep track of at least 2000 keypoints in every frame at any time. A detailed description and analysis of the results are available in the [Report](./results/report.pdf).

### Built With
This project was built with 

* python v3.8
* The environment used for developing this project is available at [environment.yml](environment.yml).



<!-- GETTING STARTED -->

## Getting Started

Clone the repository into a local machine using

```shell
git clone https://github.com/vineeths96/Visual-Odometry
```

### Prerequisites

Create a new conda environment and install all the libraries by running the following command

```shell
conda env create -f environment.yml
```

The KITTI dataset used in this project should to be downloaded to the `input/` folder. 

### Instructions to run

To plot the visual odometry of the video sequence, run the following command. Set the parameters for odometry estimation and camera parameters in the [parameters](./visual_odometry/parameters.py) file. This will estimate the trajectory, plots it along with true trajectory, and store the plots in [this](./results/trajectory) folder.

```shell
python visual_odometry.py
```



<!-- RESULTS -->

## Results

**Note that the GIFs below might not be in sync depending on the network quality. Clone the repository to your local machine and open them locally to see them in sync.**



A detailed description of methods and analysis of the results are available in the [Report](./results/report.pdf).

As one would expect, the OpenCV implementations are much more accurate and much faster than the implementations we develop. Especially, we find our implementations to be time consuming due to the naive sampling and implementation of RANSAC. We can trade off the accuracy and time by reducing the number of iterations for RANSAC. However, due to the cumulative nature of odometry, an error at one step adversely affects the estimate at all the successive steps.



|       OpenCV Nister's 5 Point Method        |       OpenCV Nister's 5 Point Method        |
| :-----------------------------------------: | :-----------------------------------------: |
|    ![1](./results/trajectory_1_5PT.gif)     |    ![2](./results/trajectory_2_5PT.gif)     |
|  **Own implementation of 8 Point Method**   |  **Own implementation of 8 Point Method**   |
|    !![1](./results/trajectory_1_8PT.gif)    |    ![1](./results/trajectory_2_8PT.gif)     |
| **OpenCV implementation of 8 Point Method** | **OpenCV implementation of 8 Point Method** |
|   ![1](./results/trajectory_1_8PT_IB.gif)   |   ![1](./results/trajectory_2_8PT_IB.gif)   |
|           **<u>Trajectory 1</u>**           |           <u>**Trajectory 2**</u>           |



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Vineeth S - vs96codes@gmail.com

Project Link: [https://github.com/vineeths96/Visual-Odometry](https://github.com/vineeths96/Visual-Odometry)



<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* [Monocular Video Odometry](https://github.com/alishobeiri/Monocular-Video-Odometery)

  > Ali Shobeiri. Monocular Video Odometry Using OpenCV. https://github.com/alishobeiri/Monocular-
  > Video-Odometery . 2019.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/vineeths96/Visual-Odometry.svg?style=flat-square
[contributors-url]: https://github.com/vineeths96/Visual-Odometry/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vineeths96/Visual-Odometry.svg?style=flat-square
[forks-url]: https://github.com/vineeths96/Visual-Odometry/network/members
[stars-shield]: https://img.shields.io/github/stars/vineeths96/Visual-Odometry.svg?style=flat-square
[stars-url]: https://github.com/vineeths96/Visual-Odometry/stargazers
[issues-shield]: https://img.shields.io/github/issues/vineeths96/Visual-Odometry.svg?style=flat-square
[issues-url]: https://github.com/vineeths96/Visual-Odometry/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/vineeths96/Visual-Odometry/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/vineeths

