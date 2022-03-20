#!/usr/bin/env python
import rospy
import csv
import numpy as np
import csv
from std_msgs.msg import String

position_error_array = []

def callback(data):

    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    position_error = data.data.split('_')[0]
    time = data.data.split('_')[1]
    position_error_array.append({'position_error': float(position_error), 'time': float(time)})


def position_error_listener():
    rospy.init_node('position_error_listener')

    rospy.Subscriber("position_error", String, callback)

    rospy.spin()

if __name__ == '__main__':
    while not rospy.is_shutdown():
        position_error_listener()
    with open('position_error.csv', 'w', newline='') as csvfile:
        fieldnames = ['position_error', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in position_error_array:
            writer.writerow(row)
