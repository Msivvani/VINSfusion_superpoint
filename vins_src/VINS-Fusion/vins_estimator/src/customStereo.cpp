/*******************************************************
 * Custom Dataset Loader for VINS-Fusion
 * 
 * Reads left and right images from a specified dataset directory and publishes
 * them on ROS topics for use with the VINS-Fusion estimator. Logs the estimated
 * poses to a file named "vio.txt".
 *******************************************************/

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp> // Boost for filesystem operations
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "estimator/estimator.h"
#include "utility/visualization.h"

using namespace std;
using namespace Eigen;
namespace fs = boost::filesystem;

Estimator estimator;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vins_custom_dataset");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    ros::Publisher pubLeftImage = n.advertise<sensor_msgs::Image>("/leftImage", 1000);
    ros::Publisher pubRightImage = n.advertise<sensor_msgs::Image>("/rightImage", 1000);

    if (argc != 3)
    {
        printf("Usage: rosrun vins custom_stereo [config file] [dataset folder]\n");
        return 1;
    }

    string config_file = argv[1];
    string dataset_path = argv[2];
    string left_folder = dataset_path + "/image_0/data/";
    string right_folder = dataset_path + "/image_1/data/";

    // Ensure directories exist
    if (!fs::exists(left_folder) || !fs::exists(right_folder))
    {
        ROS_ERROR("Left or Right folder does not exist: %s, %s", left_folder.c_str(), right_folder.c_str());
        return 1;
    }

    string output_file = dataset_path + "/vio.txt";
    FILE* outFile = fopen(output_file.c_str(), "w");
    if (outFile == NULL)
    {
        ROS_ERROR("Cannot open output file for logging: %s", output_file.c_str());
        return 1;
    }

    readParameters(config_file);
    estimator.setParameter();
    registerPub(n);

    // Read image filenames
    vector<string> left_images, right_images;
    for (const auto& entry : fs::directory_iterator(left_folder))
    {
        if (fs::is_regular_file(entry.path()))
        {
            left_images.push_back(entry.path().filename().string());
        }
    }

    for (const auto& entry : fs::directory_iterator(right_folder))
    {
        if (fs::is_regular_file(entry.path()))
        {
            right_images.push_back(entry.path().filename().string());
        }
    }

    // Sort filenames
    sort(left_images.begin(), left_images.end());
    sort(right_images.begin(), right_images.end());

    // Check if the number of images match
    if (left_images.size() != right_images.size())
    {
        ROS_ERROR("Number of images in left and right folders do not match!");
        fclose(outFile);
        return 1;
    }

    // Process each pair of images
    for (size_t i = 0; i < left_images.size(); i++)
    {
        string left_image_path = left_folder + "/" + left_images[i];
        string right_image_path = right_folder + "/" + right_images[i];

        if (left_images[i] != right_images[i])
        {
            ROS_ERROR("Mismatched filenames: %s and %s", left_images[i].c_str(), right_images[i].c_str());
            fclose(outFile);
            return 1;
        }

        cv::Mat imLeft = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat imRight = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

        if (imLeft.empty() || imRight.empty())
        {
            ROS_ERROR("Failed to load images: %s or %s", left_image_path.c_str(), right_image_path.c_str());
            continue;
        }

        ros::Time timestamp = ros::Time::now();

        sensor_msgs::ImagePtr imLeftMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imLeft).toImageMsg();
        imLeftMsg->header.stamp = timestamp;
        pubLeftImage.publish(imLeftMsg);

        sensor_msgs::ImagePtr imRightMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imRight).toImageMsg();
        imRightMsg->header.stamp = timestamp;
        pubRightImage.publish(imRightMsg);

        estimator.inputImage(timestamp.toSec(), imLeft, imRight);

        Eigen::Matrix<double, 4, 4> pose;
        estimator.getPoseInWorldFrame(pose);

        if (outFile != NULL)
        {
            fprintf(outFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
                    pose(0, 0), pose(0, 1), pose(0, 2), pose(0, 3),
                    pose(1, 0), pose(1, 1), pose(1, 2), pose(1, 3),
                    pose(2, 0), pose(2, 1), pose(2, 2), pose(2, 3));
        }

        ROS_INFO("Processed and logged pose for image pair: %s", left_images[i].c_str());
        ros::Duration(0.1).sleep(); // Simulate real-time behavior
    }

    fclose(outFile);
    ROS_INFO("Finished processing all image pairs.");
    return 0;
}