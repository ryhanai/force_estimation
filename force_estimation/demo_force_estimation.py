#!/usr/bin/env python3

import numpy as np
import copy
import json
from pathlib import Path
from collections import deque
from sklearn.cluster import KMeans

# PyTorch
import torch
# import torch._dynamo
from torchinfo import summary

# Parameter management
import hydra
from omegaconf import DictConfig

# Forcemap
from force_estimation import forcemap, force_distribution_viewer
# import force_estimation.force_estimation_v4
# import force_estimation.force_estimation_v5
from force_estimation.pick_planning import LiftingDirectionPlanner
from force_estimation.fm_utils import *

## OpenCV
import cv2
import ros2_numpy  # alternative for some functions in cv_bridge (doesn't work with numpy 2)

## ROS
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3


# torch._dynamo.config.verbose = False
# torch._dynamo.config.suppress_errors = True

import os
import re
import importlib


class Tester:
    """
    Class for force prediction using a trained model
    """

    def __init__(self, cfg: DictConfig):
        self._device = "cuda"
        self.setup_model(cfg)

    def setup_model(self, cfg):
        chkpt_dir = Path(__file__).absolute().parent.parent / 'runs' / cfg.model.checkpoint_directory
        with open(chkpt_dir / "args.json", "r") as f:
            model_params = json.load(f)

        model_class_path = model_params["model"]
        print_info(f"building model [{model_class_path}]")
        mod_name = re.sub('\.[^\.]+$', '', model_class_path)
        model_module = importlib.import_module(mod_name)
        model_class_name = re.sub('^.*\.', '', model_class_path)
        model = getattr(model_module, model_class_name)(initialize_encoder_with_pretrained_weight=False)

        weight_file = f"{chkpt_dir}/{cfg.model.weight_file}"
        print_info(f"loading pretrained weight [{weight_file}]")
        ckpt = torch.load(f"{weight_file}")

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        # model = torch.compile(model)
        summary(model, input_size=(1, 3, 360, 512))
        self._model = model

    def predict(self, image, log_scale=True):
        img = image.transpose(2, 0, 1)
        img = normalization(img.astype(np.float32), (0.0, 255.0), [0.1, 0.9])
        img = torch.from_numpy(img).float()
        x_batch = torch.unsqueeze(img, 0)
        x_batch = x_batch.to(self._device)

        y = self._model(x_batch)[0]

        y = tensor2numpy(y)
        y = y.transpose(1, 2, 0)
        if not log_scale:
            y = np.exp((y - 0.1) / 0.8) - 1.0

        return y


class Demonstration():
    """
        main class for the demonstration
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._image_topic = cfg.node.image_topic
        # self._params = copy.copy(force_estimationConfig.defaults)
        self._tester = Tester(cfg=cfg)
        self._fmap = forcemap.GridForceMap(cfg.forcemap.name)
        self._viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
        self._node = self._viewer.rviz_client._node
        self._planner = LiftingDirectionPlanner(self._fmap)
        self._object_center = None
        self._time_window = 8
        self._n_clusters = 4
        self._lifting_direction_queue = deque(maxlen=self._time_window)
        self._lifting_direction_smoother = KMeans(self._n_clusters)

        ir = IntegerRange()
        ir.from_value = 0
        ir.to_value = 1
        ir.step = 1
        pd = ParameterDescriptor(description='Set a sensor type (D415:0, SR305:1)', integer_range=[ir])
        self._node.declare_parameter('sensor_type', 0)

        fr = FloatingPointRange()
        fr.from_value = 0.0
        fr.to_value = 0.9
        # fr.step = 0.01
        pd = ParameterDescriptor(description='Lower limit of force value to draw', floating_point_range=[fr])
        self._node.declare_parameter('force_vis_threshold', 0.45, descriptor=pd)

        pd = ParameterDescriptor(description='Set true if the lifting direction is needed')
        self._node.declare_parameter('calc_lifting_direction', True, descriptor=pd)

        ir = IntegerRange()
        ir.from_value = 0
        ir.to_value = 1
        ir.step = 1
        pd = ParameterDescriptor(description='Select how to specify object position (Interactive Marker:0, Object Recognition:1)', integer_range=[ir])
        self._node.declare_parameter('object_position', 0)  # 'Interactive_marker' or 'Object_recognition'

        fr = FloatingPointRange()
        fr.from_value = 0.01
        fr.to_value = 0.20
        pd = ParameterDescriptor(description='Object radius used for lifting planning', floating_point_range=[fr])
        self._node.declare_parameter('object_radius', 0.12, descriptor=pd)

        pd = ParameterDescriptor(description='Set true to draw calibration objects')
        self._node.declare_parameter('draw_calibration_objects', False, descriptor=pd)

        self._node.add_on_set_parameters_callback(self.parameters_callback)
        self._lifting_direction_pub = self._node.create_publisher(Vector3, cfg.node.lifting_direction_topic, 1)  # 1: queue_size
        self._node.create_subscription(Image, cfg.node.image_topic, self.process_image, 1)
        self._node.create_subscription(Vector3, cfg.node.object_position_topic, self.object_position_callback, 1)

    def preprocess_HDTV(self, img):
        c = self._cfg.preprocess.roi_center
        crop = 64
        roi_sz = (1280, 720)
        img = cv2.resize(img, roi_sz)
        roi = img[180+c[0]:540+c[0], 320+c[1]+crop:960+c[1]-crop]
        return roi

    def preprocess_VGA(self, img):
        c = self._cfg.preprocess.roi_center
        img = cv2.resize(img, (960, 720))
        roi = img[180+c[0]:540+c[0], 224+c[1]:736+c[1]]
        return roi

    def do_plan(self, y, object_center):
        print_info(f"object center: {object_center}")

        # unnormalize the predicted force
        # bounds = np.log([1e-8, 1e-3])  # use fixed bounds
        # predicted_force_map = 1e6 * np.exp((y - 0.1) / 0.8 * (bounds[1] - bounds[0]) + bounds[0])
        bounds = np.log([1e-5, 1e-0])  # use fixed bounds
        predicted_force_map = np.exp(normalization(y, [0.1, 0.9], bounds))
        print_info(f"AVERAGE predicted force: {np.average(predicted_force_map)}")

        v_omega = self._planner.pick_direction_plan(
            predicted_force_map,
            object_center,
            object_radius=self._node.get_parameter('object_radius').value,
        )
        print_info(f"planning result [V, omega]: {v_omega[0]}, {v_omega[1]}")

        direction = v_omega[0]

        smooth_direction = True
        if smooth_direction:
            self._lifting_direction_queue.append(direction)
            if len(self._lifting_direction_queue) > self._n_clusters:
                self._lifting_direction_smoother.fit(self._lifting_direction_queue)
                labels = self._lifting_direction_smoother.labels_
                direction = self._lifting_direction_smoother.cluster_centers_[np.argmax(np.unique(labels, return_counts=True)[1])]            

        if direction[2] < 0.0:
            direction[2] = 0.0
        direction /= np.linalg.norm(direction)

        # draw the planned lifting direction
        self._planner.draw_result(self._viewer, 
                                    object_center,
                                    direction,
                                    rgba=[1., 0., 1., 1.],
                                    arrow_scale=[0.01, 0.02, 0.008])

        msg = Vector3()
        msg.x = direction[0]
        msg.y = direction[1]
        msg.z = direction[2]
        self._lifting_direction_pub.publish(msg)

        return direction

    def process_image(self, msg, save_result=False):
        try:
            cv_image = ros2_numpy.numpify(msg) 
        except TypeError as e:
            self.get_logger().error(f'Type Error: {e}')

        if cv_image.shape == (480, 640, 3):
            img = self.preprocess_VGA(cv_image)
        elif cv_image.shape == (720, 1280, 3):
            img = self.preprocess_HDTV(cv_image)
        elif cv_image.shape == (360, 512, 3):
            img = cv_image
        else:
            print_warn(f'INPUT IMAGE SIZE={cv_image.shape}. Only VGA and HDTV are supported')
            return

        print_info(f'INPUT IMAGE SIZE={cv_image.shape}')

        y = self._tester.predict(img)

        self._fmap.set_values(y)
        bin_state = None

        self._viewer.publish_bin_state(bin_state,
                                       self._fmap, 
                                       draw_range=[self._node.get_parameter('force_vis_threshold').value, 0.9])


        if self._node.get_parameter('calc_lifting_direction').value == True:
            if self._node.get_parameter('object_position').value == 0:
                self._object_center = self._viewer.rviz_client.getInteractiveMarkerPose()[0]
                self.do_plan(y, self._object_center)
            else:
                if isinstance(self._object_center, np.ndarray):
                    print_error(f"{self._object_center}")
                    self.do_plan(y, self._object_center)

        if self._node.get_parameter('draw_calibration_objects').value == True:
            self._viewer.draw_calibration_objects()

        # refresh the viewer
        self._viewer.rviz_client.show()

        # Show the input RGB image
        bgr_center_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Input Image', bgr_center_image)
        cv2.waitKey(1)

    def object_position_callback(self, msg: Vector3):
        if self._node.get_parameter('object_position').value != 1:
            print_warn("object recognition is accepted only if 'object_position' is set to '1 (Object_Recognition)'")
        else:
            self._object_center = np.array([msg.x, msg.y, msg.z])

    def parameters_callback(self, params):
        self._node.get_logger().info(f'Reconfigure Request: {params}')
        return SetParametersResult(successful=True)


@hydra.main(config_name='hydra_config.yaml', version_base=None, config_path='../config')
def main(cfg: DictConfig) -> None:
    demo = Demonstration(cfg)
    # rclpy.spin(demo._node)
    # demo_node.desctoy_node()
    # rclpy.shutdown()


# import rosgraph

if __name__ == '__main__':
    main()
    # if not rosgraph.is_master_online():
    #     print_error("Please run roscore before executing this script")
    #     raise Exception('roscore is not ready')
    # else:
    #     main()

# Test (publish a static image)
# $ rosrun image_publisher image_publisher /home/artuser/Dataset/forcemap/tabletop_airec241008/rgb00000_00000.jpg 
# There is no way to remap the image_publisher topic to a fixed topic. So, just relay it.
# $ rosrun topic_tools relay /image_publisher_1729594651257683803/image_raw /camera/color/image_raw
# Launch the viewer
# $ roslaunch force_estimation viewer.launch
# Launch rqt to change the demo settings online.
# $ rqt

# Send the object position using topic
# rostopic pub -1 /force_estimation/object_position geometry_msgs/Vector3 "{x: 0, y: 0.0, z: 0.75}"
