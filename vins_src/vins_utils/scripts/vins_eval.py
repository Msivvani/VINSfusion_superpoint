import numpy as np
import argparse
import rosbag
from tqdm import tqdm
from nav_msgs.msg import Path

def main(args):
    bag = rosbag.Bag(args.bag, "r")
    
    # Raw Odometry
    path_length = 0.0
    start_pose = prev_pose = np.array([0.0, 0.0, 0.0])
    for topic, msg, t in tqdm(bag.read_messages(topics=[args.topics[0]])):
        curr_pose = np.array([msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y, msg.poses[-1].pose.position.z])
        path_length += np.linalg.norm(curr_pose - prev_pose)
        prev_pose = curr_pose

    epe = np.linalg.norm(curr_pose - start_pose)
    print(f"--- {args.topics[0]} Metrics ---")
    print("Path Length: ", path_length)
    print("EPE: ", epe)
    print(f"Relative Drift: {epe * 100.0 / path_length}%")

    # IMU Propagated
    path_length = 0.0
    start_pose = prev_pose = np.array([0.0, 0.0, 0.0])
    for topic, msg, t in tqdm(bag.read_messages(topics=[args.topics[1]])):
        curr_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        path_length += np.linalg.norm(curr_pose - prev_pose)
        prev_pose = curr_pose

    bag.close()

    epe = np.linalg.norm(curr_pose - start_pose)
    print(f"--- {args.topics[1]} Metrics ---")
    print("Path Length: ", path_length)
    print("EPE: ", epe)
    print(f"Relative Drift: {epe * 100.0 / path_length}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VINS data.")
    parser.add_argument("-b", "--bag", help="path to bag containing trajectory poses")
    parser.add_argument("-t", "--topics", nargs='+', help="topic in bag containing trajectory poses")
    args = parser.parse_args()

    main(args)