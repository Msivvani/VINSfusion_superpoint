# Enhanced VINS-Fusion with Deep Feature Extraction

## Overview
This repository contains a modified implementation of **VINS-Fusion**, an optimization-based multi-sensor state estimator, to integrate **deep feature extraction and matching** using **SuperPoint** and **SuperGlue**. These modifications aim to enhance the feature tracking and matching pipeline, improving localization accuracy and robustness in challenging environments.

**Key Features:**
- **Deep feature tracking and matching**: Replaced traditional feature extraction methods with **SuperPoint** and **SuperGlue** for more robust keypoint detection and matching.  
- **Integration with VINS-Fusion pipeline**: Enhancements to the **featureTracker** module to support deep feature extraction seamlessly.  
- Improved localization performance, including a **2.94% reduction in Absolute Trajectory Error (ATE)** on the **EuRoC dataset**.

---

## Environment
- **Operating System**: Ubuntu 20.04  
- **ROS**: ROS Noetic (ROS1)  
- **Ceres Solver**: v2.1  

---

## Repository Structure
### Key Modified Components:
- **`VINS_Fusion/src/feature_tracker/featureTrackerDL.cpp`**: Implements deep feature tracking using **SuperPoint** and **SuperGlue**.  
- Additional dependencies have been integrated into the build system for compatibility with deep learning libraries.  

---

## Installation
### Prerequisites:
1. Install **ROS Noetic** ([Installation Guide](http://wiki.ros.org/noetic/Installation/Ubuntu)).
2. Install **Ceres Solver 2.1** ([Installation Guide](http://ceres-solver.org/installation.html)).
3. Clone this repository:
   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/your_username/Enhanced-VINS-Fusion.git
   cd ../
   catkin_make
   source ~/catkin_ws/devel/setup.bash

## Usage
### Running the EuRoC Dataset
Download the **EuRoC MAV Dataset** ([Download Link](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)) and place it in your workspace.

1. Open three terminals:
   - **Terminal 1**: Run VINS Estimator:
     ```bash
     roslaunch vins vins_rviz.launch
     rosrun vins vins_node ~/catkin_ws/src/Enhanced-VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml
     ```
   - **Terminal 2**: Play the dataset:
     ```bash
     rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
     ```
   - **Terminal 3 (Optional)**: Enable loop closure:
     ```bash
     rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/Enhanced-VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml
     ```

---

## Results
The modified pipeline improves localization robustness and accuracy in challenging conditions:
- Achieved an **average reduction of 2.94% in Absolute Trajectory Error (ATE)** on the **EuRoC MAV Dataset**.  
- Enhanced feature matching accuracy in low-texture and dynamic lighting conditions using **SuperPoint** and **SuperGlue**.

---

## Acknowledgements
This project builds upon the original **VINS-Fusion** by [HKUST-Aerial-Robotics](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). Additional deep feature extraction components are inspired by **SuperPoint** ([GitHub](https://github.com/magicleap/SuperPoint)) and **SuperGlue** ([GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork)).


