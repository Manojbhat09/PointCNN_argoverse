#!/bin/bash 
# usage: . ./ros_process.sh 2019-09-15-15-40-36.bag

ROS_MAKE=melodic
ROS_POINTCLOUD_TOPIC=pandar_points2

if [ "$1" != "" ]; then
    BAG_NAME=$1
    echo "##############################################################################"
    echo "Bag file is "$BAG_NAME
    echo "##############################################################################"
else
    echo "No arguements found"
fi

sleep 3

echo $ROS_MAKE
# Source ros env
source /opt/ros/$ROS_MAKE/setup.bash  # Replace with appropiate path

# run roscore
roscore &

# Start conversion 
rosrun pcl_ros pointcloud_to_pcd input:=/$ROS_POINTCLOUD_TOPIC &

# play rosbag
rosbag play $BAG_NAME; rosnode kill -a; killall -9 rosmaster; kill -9 $(pgrep amarok)

