#include "feature_tracker_dl.h"


FeatureTracker::FeatureTracker()
    : env(ORT_LOGGING_LEVEL_WARNING, "FeatureTracker"),
      session_options(),
      superpoint_session(env, "weights_dpl/superpoint_v1_sim_int32.onnx", session_options),
      superglue_session(env, "weights_dpl/superglue_indoor_sim_int32.onnx", session_options),
      n_id(0)
{
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void FeatureTracker::runSuperPoint(const cv::Mat &image, vector<cv::Point2f> &keypoints, cv::Mat &descriptors)
{
    // Convert image to grayscale and normalize
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255.0);

    std::vector<int64_t> input_dims = {1, 1, img_float.rows, img_float.cols};
    std::vector<float> input_tensor_values(img_float.begin<float>(), img_float.end<float>());

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                              input_tensor_values.size(), input_dims.data(), input_dims.size());

    auto output_tensors = superpoint_session.Run(Ort::RunOptions{nullptr},
                                                 &"input", &input_tensor, 1,
                                                 &"output", 1);

    const float *keypoints_data = output_tensors[0].GetTensorData<float>();
    const float *descriptors_data = output_tensors[1].GetTensorData<float>();

    int num_keypoints = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    int descriptor_dim = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[2];

    keypoints.clear();
    descriptors = cv::Mat(num_keypoints, descriptor_dim, CV_32F, (void *)descriptors_data).clone();

    for (int i = 0; i < num_keypoints; ++i)
    {
        keypoints.emplace_back(keypoints_data[i * 2], keypoints_data[i * 2 + 1]);
    }
}

void FeatureTracker::runSuperGlue(const cv::Mat &desc1, const cv::Mat &desc2, vector<cv::DMatch> &matches)
{
    std::vector<float> desc1_data(desc1.begin<float>(), desc1.end<float>());
    std::vector<float> desc2_data(desc2.begin<float>(), desc2.end<float>());

    std::vector<int64_t> input_shape1 = {1, desc1.rows, desc1.cols};
    std::vector<int64_t> input_shape2 = {1, desc2.rows, desc2.cols};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor1 = Ort::Value::CreateTensor<float>(memory_info, desc1_data.data(), desc1_data.size(), input_shape1.data(), input_shape1.size());
    Ort::Value input_tensor2 = Ort::Value::CreateTensor<float>(memory_info, desc2_data.data(), desc2_data.size(), input_shape2.data(), input_shape2.size());

    std::array<Ort::Value, 2> input_tensors = {input_tensor1, input_tensor2};
    auto output_tensors = superglue_session.Run(Ort::RunOptions{nullptr}, &"input1", &input_tensor1, 1, &"input2", &input_tensor2, 1);

    const float *match_data = output_tensors[0].GetTensorData<float>();
    int num_matches = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];

    matches.clear();
    for (int i = 0; i < num_matches; ++i)
    {
        matches.emplace_back(match_data[i * 2], match_data[i * 2 + 1], 0);
    }
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    cur_time = _cur_time;
    cur_img = _img.clone();
    row = cur_img.rows;
    col = cur_img.cols;

    vector<cv::Point2f> keypoints;
    cv::Mat descriptors;

    // Step 1: Detect features using SuperPoint
    runSuperPoint(cur_img, keypoints, descriptors);

    if (!prev_pts.empty())
    {
        // Step 2: Match features using SuperGlue
        vector<cv::DMatch> matches;
        runSuperGlue(prev_descriptors, descriptors, matches);

        // Step 3: Use matches to update tracking state
        cur_pts.clear();
        ids.clear();
        track_cnt.clear();

        for (const auto &match : matches)
        {
            cur_pts.push_back(keypoints[match.trainIdx].pt);
            ids.push_back(prev_ids[match.queryIdx]);
            track_cnt.push_back(prev_track_cnt[match.queryIdx] + 1);
        }
    }

    // Step 4: Package feature data
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << cur_pts[i].x, cur_pts[i].y, 1, cur_pts[i].x, cur_pts[i].y, 0, 0;
        featureFrame[ids[i]].emplace_back(0, xyz_uv_velocity);
    }

    // Visualization (if SHOW_TRACK is enabled)
    if (SHOW_TRACK)
        drawTrack(cur_img, cv::Mat(), ids, cur_pts, vector<cv::Point2f>(), map<int, cv::Point2f>());

    // Update state
    prev_pts = cur_pts;
    prev_img = cur_img.clone();
    prev_descriptors = descriptors.clone();
    prev_ids = ids;
    prev_track_cnt = track_cnt;

    return featureFrame;
}


void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
