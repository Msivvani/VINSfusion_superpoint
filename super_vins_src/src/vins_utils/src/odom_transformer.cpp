#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h> 
#include <ros/console.h>

// if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT))

class OdomTransformer
{
public:
    OdomTransformer(ros::NodeHandle& nh)
    {
        // Load the static transform values from parameters
        nh.param<std::string>("parent_frame", parent_frame_, "map");
        nh.param<std::string>("child_frame", child_frame_, "base_frame");

        nh.param<double>("x", x_, 0.0);
        nh.param<double>("y", y_, 0.0);
        nh.param<double>("z", z_, 0.0);
        nh.param<double>("roll", roll_, 0.0);
        nh.param<double>("pitch", pitch_, 0.0);
        nh.param<double>("yaw", yaw_, 0.0);

        // Subscribe to the odometry topic
        odom_sub_ = nh.subscribe("/vins_estimator/odometry", 1, &OdomTransformer::odomCallback, this);

        // Publisher for transformed odometry data
        odom_pub_ = nh.advertise<nav_msgs::Odometry>("/rsun/odometry", 1);
        // Publisher for pose-transformed
        pose_pub_ =  nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 1);
        
        // Broadcast a static transform
        broadcastStaticTransform();

        // Get Rotation Quaternion
        getRotationQuaternion();

        // Get Rotation Matrix
        getRotationMatix();

        // Broadcast static tf (base_link->camera_base_link)
        // broadcastStaticTransform_baseTocamera();

        // Broadcast static tf (camera_base_link->left_infra1_frame)
        
        // broadcastStaticTransform_cameraToinfra1();

        // Broadcast static tf (cam_imu_optical frame->imu_flu)
        // broadcastStaticTransform_imufluToimuoptical();

        broadcastStaticTransform_imufluTobaselink();

        broadcastStaticTransform_imufluTocamera();

        initialisePoseStamped();

        nh.param<double>("mavros_rate", m_dTimerR_, 30.0);
        ROS_INFO("Mavros Pose-Stamped Rate :%f", m_dTimerR_);
        
        // Timer for publishing at 30 Hz
        timer_ = nh.createTimer(ros::Duration(1.0 / m_dTimerR_), &OdomTransformer::publishTransformedData, this);
        
    }
    
private:
    ros::Subscriber odom_sub_;
    ros::Publisher odom_pub_, pose_pub_;
    
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;
    tf2_ros::TransformBroadcaster dynamic_broadcaster_;

    std::string parent_frame_, child_frame_;
    double x_, y_, z_, roll_, pitch_, yaw_;

    ros::Timer timer_;
    double m_dTimerR_;

    // To get rotation quaternion
    Eigen::Quaterniond rot_quaternion;

    // Rotation matrix for transforming into base-frame
    Eigen::Matrix3d rot_matix;


    // Pose-Stamped ros-msg
    geometry_msgs::PoseStamped m_sPStamp;

    void initialisePoseStamped(){
        // Initializing w/t zeros
        m_sPStamp.pose.position.x = 0;
        m_sPStamp.pose.position.y = 0;
        m_sPStamp.pose.position.z = 0;
        m_sPStamp.pose.orientation.x = 0;
        m_sPStamp.pose.orientation.y = 0;
        m_sPStamp.pose.orientation.z = 0;
        m_sPStamp.pose.orientation.w = 1;
        m_sPStamp.header.frame_id = parent_frame_;  // Set the new frame ID
        m_sPStamp.header.stamp = ros::Time::now();
    }

    


    void getRotationMatix()
    {
        // Rotate into base-frame coordinate system 
        rot_matix = Eigen::AngleAxisd(yaw_  , Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(roll_ , Eigen::Vector3d::UnitX());
    }


    void getRotationQuaternion()
    {
        // Get rotation quaternions
        Eigen::Quaterniond yaw_quat(Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond pitch_quat(Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY()));
        Eigen::Quaterniond roll_quat(Eigen::AngleAxisd(roll_, Eigen::Vector3d::UnitX()));

        rot_quaternion = yaw_quat * pitch_quat * roll_quat;
    }


    void broadcastStaticTransform()
    {
        geometry_msgs::TransformStamped static_transform;

        static_transform.header.stamp = ros::Time::now();
        static_transform.header.frame_id = parent_frame_;
        static_transform.child_frame_id = child_frame_;
        

        // Set the static transform (translation)
        static_transform.transform.translation.x = x_;
        static_transform.transform.translation.y = y_;
        static_transform.transform.translation.z = z_;

        // Set the static transform (rotation)
        tf2::Quaternion quat;
        quat.setRPY(roll_, pitch_, yaw_);
        static_transform.transform.rotation.x = quat.x();
        static_transform.transform.rotation.y = quat.y();
        static_transform.transform.rotation.z = quat.z();
        static_transform.transform.rotation.w = quat.w();

        // Broadcast the static transform
        static_broadcaster_.sendTransform(static_transform);
    }

    void broadcastStaticTransform_baseTocamera()
    {
        geometry_msgs::TransformStamped static_transform;

        static_transform.header.stamp = ros::Time::now();
        static_transform.header.frame_id = "base_link";
        static_transform.child_frame_id = "camera_base_link";
        

        // Set the static transform (translation)
        static_transform.transform.translation.x = 0.14;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = -0.065;

        // Set the static transform (rotation)
        tf2::Quaternion quat;
        quat.setRPY(roll_, pitch_, yaw_);
        static_transform.transform.rotation.x = 0;
        static_transform.transform.rotation.y = 0;
        static_transform.transform.rotation.z = 0;
        static_transform.transform.rotation.w = 1.0;

        // Broadcast the static transform
        static_broadcaster_.sendTransform(static_transform);
    }


    void broadcastStaticTransform_imufluTobaselink()
    {
        geometry_msgs::TransformStamped static_transform;

        static_transform.header.stamp = ros::Time::now();
        static_transform.header.frame_id = "imu_flu";
        static_transform.child_frame_id = "base_link";
        

        // Set the static transform (translation)
        static_transform.transform.translation.x = -0.135;
        static_transform.transform.translation.y = -0.018;
        static_transform.transform.translation.z = 0.056;


        static_transform.transform.rotation.x = 0.0;
        static_transform.transform.rotation.y = 0.0;
        static_transform.transform.rotation.z = 0.0;
        static_transform.transform.rotation.w = 1.0;


        // Broadcast the static transform
        static_broadcaster_.sendTransform(static_transform);

    }

    void broadcastStaticTransform_imufluTocamera()
    {
        geometry_msgs::TransformStamped static_transform;

        static_transform.header.stamp = ros::Time::now();
        static_transform.header.frame_id = "imu_flu";
        static_transform.child_frame_id = "camera_link";
        

        // Set the static transform (translation)
        static_transform.transform.translation.x = 0.016;
        static_transform.transform.translation.y = 0.03;
        static_transform.transform.translation.z = -0.007;


        static_transform.transform.rotation.x = 0.0;
        static_transform.transform.rotation.y = 0.0;
        static_transform.transform.rotation.z = 0.0;
        static_transform.transform.rotation.w = 1.0;


        // Broadcast the static transform
        static_broadcaster_.sendTransform(static_transform);

    }


    void broadcastStaticTransform_cameraToinfra1()
    {
        geometry_msgs::TransformStamped static_transform;

        static_transform.header.stamp = ros::Time::now();
        static_transform.header.frame_id = "camera_base_link";
        static_transform.child_frame_id = "camera_link";
        

        // Set the static transform (translation)
        static_transform.transform.translation.x = 0.011;
        static_transform.transform.translation.y = 0.048;
        static_transform.transform.translation.z = 0.0015;


        static_transform.transform.rotation.x = 0.0;
        static_transform.transform.rotation.y = 0.0;
        static_transform.transform.rotation.z = 0.0;
        static_transform.transform.rotation.w = 1.0;


        // Broadcast the static transform
        static_broadcaster_.sendTransform(static_transform);

    }

    void broadcastStaticTransform_imufluToimuoptical()
    {
        geometry_msgs::TransformStamped static_transform;

        static_transform.header.stamp = ros::Time::now();
        static_transform.header.frame_id = "camera_imu_optical_frame";
        static_transform.child_frame_id = "imu_flu";
        

        // Set the static transform (translation)
        static_transform.transform.translation.x = 0.0;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0;

        // Relative rotations (roll->pitch->yaw)
        tf2::Quaternion quat;
        quat.setRPY(1.57, -1.57, 0.0);
        static_transform.transform.rotation.x = quat.x();
        static_transform.transform.rotation.y = quat.y();
        static_transform.transform.rotation.z = quat.z();
        static_transform.transform.rotation.w = quat.w();


        // Broadcast the static transform
        static_broadcaster_.sendTransform(static_transform);

    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        try
        {
            // Create a transformed odometry message based on the input
            nav_msgs::Odometry transformed_odom = *msg;
            transformed_odom.header.frame_id = parent_frame_;  // Set the new frame ID

            // Rotate
            Eigen::Vector3d pose_eig_(transformed_odom.pose.pose.position.x, 
                                      transformed_odom.pose.pose.position.y, 
                                      transformed_odom.pose.pose.position.z);
            Eigen::Vector3d pose_transformed_ = rot_matix * pose_eig_;

            // Translate 
            transformed_odom.pose.pose.position.x = pose_transformed_.x() + x_;
            transformed_odom.pose.pose.position.y = pose_transformed_.y() + y_;
            transformed_odom.pose.pose.position.z = pose_transformed_.z() + z_;


            // Current rotation 
            double qx = transformed_odom.pose.pose.orientation.x;
            double qy = transformed_odom.pose.pose.orientation.y;
            double qz = transformed_odom.pose.pose.orientation.z;
            double qw = transformed_odom.pose.pose.orientation.w;

            Eigen::Quaterniond current_quat(qw, qx, qy, qz);
            
            auto final_rot_eul = current_quat.toRotationMatrix();
            double roll = atan2(final_rot_eul(2, 1), final_rot_eul(2, 2));  // rotation around x-axis
            double pitch = -asin(final_rot_eul(2, 0));                          // rotation around y-axis
            double yaw = atan2(final_rot_eul(1, 0), final_rot_eul(0, 0));

            double roll_new = pitch;
            double pitch_new = -(roll + M_PI/2);
            double yaw_new = yaw;

            auto final_rot = Eigen::AngleAxisd(yaw_new  , Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(pitch_new, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(roll_new , Eigen::Vector3d::UnitX());

            Eigen::Quaterniond rot_transformed_(final_rot);

            
            transformed_odom.pose.pose.orientation.x = rot_transformed_.x();
            transformed_odom.pose.pose.orientation.y = rot_transformed_.y();
            transformed_odom.pose.pose.orientation.z = rot_transformed_.z();
            transformed_odom.pose.pose.orientation.w = rot_transformed_.w();

            
            m_sPStamp.pose = transformed_odom.pose.pose;
            
            // Publish the transformed odometry data
            odom_pub_.publish(transformed_odom);


            // Publish transform
            geometry_msgs::TransformStamped dynamic_transform;
            dynamic_transform.header.stamp = ros::Time::now();
            dynamic_transform.header.frame_id = "map";
            dynamic_transform.child_frame_id = "imu_flu";

            // Set translation from transformed odometry
            dynamic_transform.transform.translation.x = transformed_odom.pose.pose.position.x;
            dynamic_transform.transform.translation.y = transformed_odom.pose.pose.position.y;
            dynamic_transform.transform.translation.z = transformed_odom.pose.pose.position.z;

            // Set rotation from transformed odometry
            dynamic_transform.transform.rotation.x = transformed_odom.pose.pose.orientation.x;
            dynamic_transform.transform.rotation.y = transformed_odom.pose.pose.orientation.y;
            dynamic_transform.transform.rotation.z = transformed_odom.pose.pose.orientation.z;
            dynamic_transform.transform.rotation.w = transformed_odom.pose.pose.orientation.w;

            // Broadcast the dynamic transform
            dynamic_broadcaster_.sendTransform(dynamic_transform);
        }
        catch (const tf2::TransformException& ex)
        {
            ROS_WARN("%s", ex.what());
        }
    }

    void publishTransformedData(const ros::TimerEvent&)
    {
        // Publish pose-stamped    
        m_sPStamp.header.stamp = ros::Time::now();
        pose_pub_.publish(m_sPStamp);

        // Logging 
        ROS_INFO("Pose-Stamped Position->x: %f, y: %f, z: %f | Orientation->x: %f, y: %f, z: %f, w: %f",
        m_sPStamp.pose.position.x, 
        m_sPStamp.pose.position.y, 
        m_sPStamp.pose.position.z,
        m_sPStamp.pose.orientation.x, 
        m_sPStamp.pose.orientation.y, 
        m_sPStamp.pose.orientation.z, 
        m_sPStamp.pose.orientation.w
        );
        
    }
    
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "odom_transformer_node");
    ros::NodeHandle nh;

    OdomTransformer odomTransformer(nh);

    ros::spin();
    return 0;
}
