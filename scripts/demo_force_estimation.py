#!/usr/bin/env python3

import numpy as np
import scipy
from scipy.optimize import minimize
import forcemap
import force_estimation_v2 as fe

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import force_distribution_viewer

from dynamic_reconfigure.server import Server
from force_estimation.cfg import force_estimationConfig
import copy


image_topic = '/camera/color/image_raw'


model = fe.model_rgb_to_fmap_res50()
model.load_weights('../runs/ae_cp.basket-filling2.model_resnet.20221202165608/cp.ckpt')
fmap = forcemap.GridForceMap('seria_basket')
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
params = copy.copy(force_estimationConfig.defaults)


bridge = CvBridge()


def crop_center_d415(img):
    c = (-40, 75)
    crop = 64
    return img[180+c[0]:540+c[0], 320+c[1]+crop:960+c[1]-crop]


def crop_center_sr305(img):
    c = (-40, 75)
    crop = 64
    img = cv2.resize(img, (1280, 720))
    return img[180+c[0]:540+c[0], 320+c[1]+crop:960+c[1]-crop]


def f(y_pred, object_center=[0.02, -0.04, 0.79], object_radius=0.05):
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = y_pred
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fmap.get_positions()
    idx = np.where(scipy.linalg.norm(ps - object_center, axis=1) < object_radius)[0]
    ps = ps[idx]
    g_vecs = g_vecs[idx]
    idx = np.where(np.sum((ps - object_center) * g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]
    return fps, fg_vecs

cons = (
    {'type': 'eq', 'fun': lambda x: scipy.linalg.norm(x[:3]) - 1},
    {'type': 'eq', 'fun': lambda x: scipy.linalg.norm(x[3:]) - 1}
)


def f_target(fps,
             fg_vecs,
             c=np.array([0.03, -0.02, 0.78]),
             x=np.array([0, 0, 1, 0, 0, 1]),
             omega=np.array([0, 0, 1]),
             delta=0.05,
             alpha=1.0):
    v = x[:3]
    omega = x[3:]
    dp = v + alpha * np.cross(omega, fps - c)
    return np.sum(fg_vecs * (delta * dp))


def pick_direction_plan(y_pred, object_center, object_radius, scale=[0.005, 0.01, 0.004]):
    fps, fg_vecs = f(y_pred, object_center, object_radius)

    def f_objective(x):
        return -f_target(fps, fg_vecs, object_center, x=x)

    result = minimize(f_objective, x0=np.array([0, 0, 1, 0, 0, 1]), constraints=cons)
    print(result)
    pick_direction = result.x[:3]
    pick_omega = result.x[3:]

    viewer.rviz_client.draw_sphere(object_center, [1, 0, 0, 1], [0.01, 0.01, 0.01])
    viewer.rviz_client.draw_arrow(object_center, object_center + pick_direction * 0.1, [1, 0, 1, 1], scale)
    # viewer.rviz_client.draw_arrow(object_center, object_center + pick_omega * 0.1, [1, 1, 0, 1], scale)
    return pick_direction, pick_omega


def process_image(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    except CvBridgeError as e:
        rospy.logerr('CvBridge Error: {0}'.format(e))

    sensor = params['sensor_type']
    if sensor == 0:
        img = crop_center_d415(cv_image)
    elif sensor == 1:
        img = crop_center_sr305(cv_image)
    else:
        print('unknown sensor: ', sensor)
        cv2.waitKey(1)
        return

    fimg = img.astype(np.float64) / 255.

    xs = np.expand_dims(fimg, 0)
    predicted_force_map = model.predict(xs)

    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = predicted_force_map[0]
    fmap.set_values(fv)
    viewer.publish_bin_state(None, 
                             fmap, 
                             draw_fmap=True, 
                             draw_force_gradient=False, 
                             force_threshold=params['force_vis_threshold'])

    if params['calc_lifting_direction'] == True:
        object_center = viewer.rviz_client.getObjectPosition()
        v, omega = pick_direction_plan(predicted_force_map, object_center, object_radius=0.05)
        print(v, omega)
        viewer.rviz_client.show()

    bgr_center_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Input Image', bgr_center_image)
    cv2.waitKey(1)


def parameter_callback(config, level):
    global params
    rospy.loginfo("""Reconfigure Request: {calc_lifting_direction}, {force_vis_threshold}, {sensor_type}""".format(**config))
    params = config
    return config


def start_node():
    rospy.Subscriber(image_topic, Image, process_image)
    param_srv = Server(force_estimationConfig, parameter_callback)
    rospy.spin()    


if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
