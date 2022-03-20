/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

/* A ROS implementation of the Pure pursuit path tracking algorithm (Coulter 1992).
   Terminology (mostly :) follows:
   Coulter, Implementation of the pure pursuit algoritm, 1992 and
   Sorniotti et al. Path tracking for Automated Driving, 2017.
 */

#include <string>
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/transform_storage.h>
#include <tf2/buffer_core.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <kdl/frames.hpp>
#include <raiscar_msgs/ControlReport.h>
#include "raiscar_msgs/ControlCommand.h"
#include "std_msgs/Bool.h"
#include "controller.h"
#include "std_msgs/String.h"

using std::string;

class PurePursuit : public controller {
public:

    //! Constructor
    PurePursuit();

    //! Run the controller.
    void run();

    void open_csv(string cte_file, string head_file);

    void close_csv();

private:
    void controller_step(nav_msgs::Odometry odom);

    void ComputeCTE(tf2::Vector3 car_front, tf2::Vector3 car_back, tf2::Vector3 car_direction_norm);

    void ComputeHeadingError(tf2::Vector3 car_direction_norm);

    double ld_dist_;
    std::vector<tf2::Vector3> points_taken;
    int current_path_index = 0;
    double start_time_;
    void getError(tf2::Vector3 car_center, tf2::Vector3 car_dir);
};


PurePursuit::PurePursuit() {
    // Get parameters from the parameter server
    nh_private_.param<double>("lookahead_dist", ld_dist_, 0.4);
    std::cout << "Pure Pursuit controller started..." << std::endl;
}

void PurePursuit::open_csv(string cte_file, string head_file) {
    cte_csv_file.open(cte_file, std::ofstream::out);
    cte_csv_file << "Cross track error,Time\n";
    heading_csv_file.open(head_file, std::ofstream::out);
    heading_csv_file << "Heading error,Time\n";
}

void PurePursuit::close_csv() {
    cte_csv_file.close();
    heading_csv_file.close();
}

void PurePursuit::ComputeCTE(tf2::Vector3 car_front, tf2::Vector3 car_back, tf2::Vector3 car_direction_norm) {
    tf2::Vector3 car_center;
    tf2::Vector3 car_length;
    car_back.setZ(0.0);
    car_front.setZ(0.0);
    car_length = car_front - car_back;
    car_center = car_back + (car_length.length() / 2) * car_direction_norm;
    geometry_msgs::Vector3 car_center_msg;
    tf2::convert(car_center, car_center_msg);

    double cross_track_error = std::numeric_limits<int>::max();
    for (int i = 0; i < path_.size(); i++) {
        //get path position
        tf2::Vector3 path_pos = path_[i].getOrigin();
        //calculate distance btw path position and point in front of car
        double distance = path_pos.distance(car_center);
        cross_track_error = std::min(distance, cross_track_error);
    }

    ROS_INFO_STREAM("CTE:" << cross_track_error);
    ROS_INFO_STREAM("TIME STEPS:" << ++time_steps);
    cte_csv_file << cross_track_error << "," << time_steps << "\n";
}

void PurePursuit::ComputeHeadingError(tf2::Vector3 car_direction_norm) {
    double head_error = std::numeric_limits<int>::max();

    for (int i = 0; i < path_.size(); i++) {
        //get path position
        double car_angle = car_direction_norm.angle(path_[i].getOrigin());
        //calculate distance btw path position and point in front of car
        head_error = std::min(head_error, car_angle);
    }

    ROS_INFO_STREAM("HEADING ERROR:" << head_error);
    ROS_INFO_STREAM("TIME STEPS:" << time_steps);
    heading_csv_file << head_error << "," << time_steps << "\n";
}


void PurePursuit::getError(tf2::Vector3 car_center, tf2::Vector3 car_dir) {

    double cross_track_error = std::numeric_limits<int>::max();
    double heading_error;

    for (int i = 0; i < path_.size() - 1; i++) {

        //get path position
        tf2::Vector3 pathpos = path_[i].getOrigin();
        tf2::Vector3 pathdir = path_[i + 1].getOrigin() - pathpos;

        double t0 = pathdir.dot(car_center- path_[i].getOrigin()) / pathdir.dot(pathdir);

        tf2::Vector3 intersectPnt = pathpos + t0 * pathdir;

        //calculate distance btw path position and point in front of car
        double distance = intersectPnt.distance(car_center);

        if (distance < cross_track_error) {
            cross_track_error = distance;
            heading_error = pathdir.angle(car_dir);
        }
    }
    cte_csv_file << cross_track_error << "," << ++time_steps << "\n";
    heading_csv_file << heading_error << "," << time_steps << "\n";
}


/*
 * Implement your controller here! The function gets called each time a new odometry is incoming.
 * The path to follow is saved in the variable "path_". Once you calculated the new control outputs you can send it with
 * the pub_acker_ publisher.
 */
void PurePursuit::controller_step(nav_msgs::Odometry odom) {
    // Code blocks that could be useful:
    //ld_dist_ = 1.5;
    // The following code block could receive the current pose (saved in map_t_fa)

    if (!goal_reached_) {

        geometry_msgs::TransformStamped tf_msg;
        geometry_msgs::TransformStamped front_axis_tf_msg;
        geometry_msgs::TransformStamped rear_axis_tf_msg;
        tf2::Stamped<tf2::Transform> map_t_fa;

        try {
            tf_msg = tf_buffer_.lookupTransform(map_frame_id_, rear_axis_frame_id_, ros::Time(0));
            front_axis_tf_msg = tf_buffer_.lookupTransform(map_frame_id_, front_axis_frame_id_, ros::Time(0));
            rear_axis_tf_msg = tf_buffer_.lookupTransform(map_frame_id_, rear_axis_frame_id_, ros::Time(0));
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN_STREAM(ex.what());
        }
        tf2::convert(tf_msg, map_t_fa);

        tf2::Vector3 current_path_point;
        double current_path_distance = 0;

        // Determine car position
        tf2::Vector3 car_pos;
        car_pos[0] = tf_msg.transform.translation.x;
        car_pos[1] = tf_msg.transform.translation.y;
        car_pos[2] = tf_msg.transform.translation.z;

        tf2::Vector3 car_front;
        car_front[0] = front_axis_tf_msg.transform.translation.x;
        car_front[1] = front_axis_tf_msg.transform.translation.y;
        car_front[2] = front_axis_tf_msg.transform.translation.z;

        tf2::Vector3 car_back;
        car_back[0] = rear_axis_tf_msg.transform.translation.x;
        car_back[1] = rear_axis_tf_msg.transform.translation.y;
        car_back[2] = rear_axis_tf_msg.transform.translation.z;

        double alpha = 0;

        tf2::Quaternion q(
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // car direction vector
        tf2::Vector3 car_direction;
        car_direction[0] = cos(yaw) * cos(pitch);
        car_direction[1] = sin(yaw) * cos(pitch);
        car_direction[2] = sin(pitch);

        // choose point in front of the car in range of lookahead distance
        tf2::Vector3 car_direction_norm = car_direction.normalize();
        tf2::Vector3 point_on_circle = car_pos + ld_dist_ * car_direction_norm;

//        ComputeCTE(car_front, car_back, car_direction_norm);
//        ComputeHeadingError(car_direction_norm);
        tf2::Vector3 car_center;
        tf2::Vector3 car_length;
        car_back.setZ(0.0);
        car_front.setZ(0.0);
        car_length = car_front - car_back;
        car_center = car_back + (car_length.length() / 2) * car_direction_norm;

        getError(car_center, car_direction);

        if (path_.size() > 0) {
            //loop through points
            for (int i = 0; i < path_.size(); i++) {

                //get path position
                tf2::Vector3 path_pos = path_[i].getOrigin();
                geometry_msgs::Vector3 tf_translation = tf_msg.transform.translation;

                //calculate distance btw path position and point in front of car
                double distance = path_pos.distance(point_on_circle);

                // look only in range from l_distance and points which were not taken
                //

                if (distance < ld_dist_ &&
                    not(std::find(points_taken.begin(), points_taken.end(), path_pos) != points_taken.end())) {
                    // get the maximal distance point
                    if (current_path_distance < distance) {
                        // add to points taken
                        points_taken.push_back(current_path_point);
                        //set new path point
                        current_path_index = i;
                        current_path_distance = distance;
                        current_path_point = path_pos;
                    }
                }
            }

            // remove all points which appear before the chosen path point
            int limit = std::min(float(current_path_index), float(0));
            for (int j = 0; j < limit; j++) {
                points_taken.push_back(path_[j].getOrigin());
            }
            //points_taken.push_back(current_path_point);
            tf2::Vector3 goal_pos = path_[path_.size() - 1].getOrigin();
            ROS_INFO_STREAM("GOAL DISTANCE: " << goal_pos.distance(car_pos));
            if (goal_pos.distance(car_pos) < pos_tol_) {
                cmd_control_.brake = 5.0;
                cmd_control_.steering = 1.;
                cmd_control_.throttle = 0.;
                cmd_control_.throttle_mode = 0;
                pub_acker_.publish(cmd_control_);
                goal_reached_ = true;
                sendGoalMsg(true);
                ROS_INFO_STREAM("GOAL REACHED");
                close_csv();
            }


            tf2::Vector3 next_point = path_[current_path_index].getOrigin();

            // Calculate the direction vector from the car to the chosen path point
            tf2::Vector3 car_to_point;
            car_to_point = current_path_point - car_pos;


            // Calculate the angle alpha
            alpha = car_to_point.angle(car_direction);

            // Determine if right or left steering angle
            tf2::Vector3 car_point2 = car_pos + car_direction;
            float value = (current_path_point[0] - car_pos[0]) * (car_point2[1] - car_pos[1]) -
                          (current_path_point[1] - car_pos[1]) * (car_point2[0] - car_pos[0]);

            //current_path_point.cross(car_pos);

            if (value > 0) {
                alpha = -alpha;
            }
            alpha = atan(((2 * L_) / ld_dist_) * sin(alpha));

            //std::cout << "closest distance: " << current_path_distance << std::endl;
            //std::cout << "angle: " << alpha << std::endl;


            // The following code block sends out the boolean true to signal that the last waypoint is reached:

            //sendGoalMsg(true);

            // The following code block could be used for sending out a tf-pose for debugging

            target_p_.transform.translation.x = car_direction[0];// dummy value
            target_p_.transform.translation.y = car_direction[1]; // dummy value
            target_p_.transform.translation.z = car_direction[2]; // dummy value
            target_p_.header.frame_id = rear_axis_frame_id_;
            target_p_.header.stamp = ros::Time::now();
            tf_broadcaster_.sendTransform(target_p_);

            // The following code block can be used to control a certain velocity using PID control
            float pid_vel_out = 0.0;
            //des_vel = 0.05;
            if (des_v_ >= 0) {
                pid_vel_out = vel_pid.step((des_v_ - odom.twist.twist.linear.x), ros::Time::now());
            } else {
                pid_vel_out = des_v_;
                vel_pid.resetIntegral();
            }

            // The following code block can be used to send control commands to the car
            cmd_control_.steering = alpha / (70.0 * M_PI / 180.0); //  DUMMY_STEERING_ANGLE should be a value in radiant
            cmd_control_.throttle = pid_vel_out;
            cmd_control_.throttle_mode = 0;

            cmd_control_.throttle = std::min(cmd_control_.throttle, throttle_limit_);
            cmd_control_.throttle = std::max(std::min((double) cmd_control_.throttle, 1.0), 0.0);
            cmd_control_.throttle = 0.1;
            pub_acker_.publish(cmd_control_);
        }
    } else {
        // goal reached
        cmd_control_.steering = 0.; //  DUMMY_STEERING_ANGLE should be a value in radiant
        cmd_control_.throttle = 0.;
        cmd_control_.throttle_mode = 0;
        pub_acker_.publish(cmd_control_);
    }
}

void PurePursuit::run() {
    ros::spin();
}

int main(int argc, char **argv) {
    std::cout << "init..." << std::endl;
    ros::init(argc, argv, "pure_pursuit_controller");
    std::cout << "start controller..." << std::endl;

    PurePursuit controller;
    controller.open_csv(
            "/home/freicar/freicar_ws/src/freicar_ss21_exercises/01-01-control-exercise/freicar_control/cte_errors.csv",
            "/home/freicar/freicar_ws/src/freicar_ss21_exercises/01-01-control-exercise/freicar_control/heading_errors.csv");
    controller.run();
    return 0;
}
