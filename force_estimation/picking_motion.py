# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy
from jog_msgs.msg import JogFrame
import tf2_ros
import numpy as np
from tf.transformations import *


jello_v_omega = np.array([ 0.04550926,
                           0.44922131,
                           0.89226073,
                           0.86268325,
                           -0.19106699,
                           0.46826575])


phase = 0


rospy.init_node('picking_motion')
rospy.loginfo('picking_motion node started')

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)
pub = rospy.Publisher('/jog_frame', JogFrame, queue_size=1)
rate = rospy.Rate(15.0)


while not rospy.is_shutdown():
    # trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time.now(), rospy.Duration(0.2))
    # p = trans.transform.translation
    # print('translation = ', np.array([p.x, p.y, p.z]))
    # p = trans.transform.rotation
    # print('rotation = ', np.array([p.x, p.y, p.z, p.w]))

    if phase == 0:
        try:
            trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time.now(), rospy.Duration(0.2))
            p = trans.transform.translation
            initial_position = np.array([p.x, p.y, p.z])
            print(initial_position)
            next_goal = initial_position + np.array([0, 0, 0.10])
            phase = 1
        except:
            rate.sleep()
        continue

    elif phase == 1:
        try:
            trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time.now(), rospy.Duration(0.2))
            p = trans.transform.translation
            p = np.array([p.x, p.y, p.z])
            dp = initial_position - p
            if np.linalg.norm(dp) > 0.02 : 
                print('phase1 finished')
                phase = 2
                continue

            # v = np.array([0, 0, 0.003])
            v = np.array([0.04550926, 0.44922131, 0.89226073]) * 0.002
            omega = np.array([0, 0.86268325, 0]) * 0.005
            msg = JogFrame()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_link'
            msg.group_name = 'manipulator'
            msg.link_name = 'tool0'
            msg.angular_delta.x = omega[0]
            msg.angular_delta.y = omega[1]
            msg.angular_delta.z = omega[2]
            msg.linear_delta.x = v[0]
            msg.linear_delta.y = v[1]
            msg.linear_delta.z = v[2]
            pub.publish(msg)
        except:
            rate.sleep()
        continue

    elif phase == 2:
        try:
            trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time.now(), rospy.Duration(0.2))
            p = trans.transform.translation
            p = np.array([p.x, p.y, p.z])
            dp = initial_position - p
            if np.linalg.norm(dp) > 0.07 : 
                print('phase2 finished')
                phase = 3
                continue

            v = np.array([0, 0, 0.003])
            omega = np.array([0, 0, 0])
            msg = JogFrame()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_link'
            msg.group_name = 'manipulator'
            msg.link_name = 'tool0'
            msg.angular_delta.x = omega[0]
            msg.angular_delta.y = omega[1]
            msg.angular_delta.z = omega[2]
            msg.linear_delta.x = v[0]
            msg.linear_delta.y = v[1]
            msg.linear_delta.z = v[2]
            pub.publish(msg)
        except:
            rate.sleep()
        continue

    else:
        break
