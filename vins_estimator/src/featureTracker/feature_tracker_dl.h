#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.h> // ONNX Runtime for SuperPoint and SuperGlue
#include "/home/gangadhar-nageswar/siv/onnx/onnxruntime-linux-x64-gpu-1.16.3/include/onnxruntime_cxx_api.h"

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    // void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                   vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts, 
                   vector<cv::Point2f> &curRightPts, map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    // double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    // bool inBorder(const cv::Point2f &pt);

private:
    void runSuperPoint(const cv::Mat &image, vector<cv::Point2f> &keypoints, cv::Mat &descriptors);
    void runSuperGlue(const cv::Mat &desc1, const cv::Mat &desc2, vector<cv::DMatch> &matches);

    Ort::Env env;
    Ort::Session superpoint_session;
    Ort::Session superglue_session;
    Ort::SessionOptions session_options;

    int row, col;
    cv::Mat imTrack;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> prev_pts, cur_pts;
    vector<int> ids, track_cnt;
    double cur_time, prev_time;
    int n_id;
};
