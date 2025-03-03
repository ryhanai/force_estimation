#!/usr/bin/env python3

import rospy

from dynamic_reconfigure.server import Server
from force_estimation.cfg import force_estimationConfig

def callback(config, level):
    rospy.loginfo("""Reconfigure Request: {calc_lifting_direction}, {force_vis_threshold},\ 
          {sensor_type}""".format(**config))
    return config

if __name__ == "__main__":
    rospy.init_node("test_dynamic_reconfigure", anonymous = False)

    srv = Server(force_estimationConfig, callback)
    rospy.spin()