#!/usr/bin/env python3

import numpy as np
import copy
import json
from pathlib import Path

# PyTorch
import torch
# import torch._dynamo
from torchinfo import summary

# Parameter management
import hydra
from omegaconf import DictConfig

# Forcemap
import forcemap
import force_distribution_viewer
# from force_estimation_v4 import *
from force_estimation_v5 import *
from pick_planning import LiftingDirectionPlanner
from fm_utils import *

## OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

## ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from dynamic_reconfigure.server import Server
from force_estimation.cfg import force_estimationConfig


# torch._dynamo.config.verbose = False
# torch._dynamo.config.suppress_errors = True


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

        weight_file = f"{chkpt_dir}/{cfg.model.weight_file}"
        print_info(f"loading pretrained weight [{weight_file}]")
        ckpt = torch.load(f"{weight_file}")

        model_class = model_params["model"]
        print_info(f"building model [{model_class}]")
        model = globals()[model_class](initialize_encoder_with_pretrained_weight=False)
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


class Demonstration:
    """
        main class for the demonstration
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._image_topic = cfg.node.image_topic
        self._params = copy.copy(force_estimationConfig.defaults)
        self._bridge = CvBridge()
        self._tester = Tester(cfg=cfg)
        self._fmap = forcemap.GridForceMap(cfg.forcemap.name)
        self._viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
        self._planner = LiftingDirectionPlanner(self._fmap)
        self._object_center = None
        self._lifting_direction_pub = rospy.Publisher(cfg.node.lifting_direction_topic, Vector3, queue_size=1)

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
            object_radius=self._cfg.lifting_planning.object_radius,
        )
        print_info(f"planning result [V, omega]: {v_omega[0]}, {v_omega[1]}")

        # draw the planned lifting direction
        direction = v_omega[0]
        self._planner.draw_result(self._viewer, 
                                    object_center,
                                    direction,
                                    rgba=[1, 0, 1, 1],
                                    arrow_scale=[0.005, 0.01, 0.004])

        msg = Vector3()
        msg.x = direction[0]
        msg.y = direction[1]
        msg.z = direction[2]
        self._lifting_direction_pub.publish(msg)

        return direction

    def process_image(self, msg, save_result=False):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError as e:
            rospy.logerr('CvBridge Error: {0}'.format(e))

        if cv_image.shape == (480, 640, 3):
            img = self.preprocess_VGA(cv_image)
        elif cv_image.shape == (720, 1280, 3):
            img = self.preprocess_HDTV(cv_image)
        else:
            print_warn(f'INPUT IMAGE SIZE={cv_image.shape}. Only VGA and HDTV are supported')
            return

        print_info(f'INPUT IMAGE SIZE={cv_image.shape}')

        y = self._tester.predict(img)

        self._fmap.set_values(y)
        bin_state = None
        self._viewer.publish_bin_state(bin_state,
                                       self._fmap, 
                                       draw_range=[self._params['force_vis_threshold'], 0.9])

        if self._params['calc_lifting_direction'] == True:
            if self._params['object_position'] == force_estimationConfig.force_estimation_Interactive_marker:
                self._object_center = self._viewer.rviz_client.getObjectPosition()
                self.do_plan(y, self._object_center)
            else:
                if isinstance(self._object_center, np.ndarray):
                    print_error(f"{self._object_center}")
                    self.do_plan(y, self._object_center)

        # refresh the viewer
        self._viewer.rviz_client.show()

        # Show the input RGB image
        bgr_center_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Input Image', bgr_center_image)
        cv2.waitKey(1)

    def object_position_callback(self, msg: Vector3):
        if self._params['object_position'] != force_estimationConfig.force_estimation_Object_recognition:
            print_warn("object recognition is accepted only if 'object_position' is set to 'Object_Recognition'")
        else:
            self._object_center = np.array([msg.x, msg.y, msg.z])

    def parameter_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {calc_lifting_direction}, {force_vis_threshold}, {sensor_type}""".format(**config))
        self._params = config
        return config


@hydra.main(config_name='hydra_config.yaml', version_base=None, config_path='../config')
def main(cfg: DictConfig) -> None:
    demo = Demonstration(cfg)
    rospy.Subscriber(cfg.node.image_topic, Image, demo.process_image)
    rospy.Subscriber(cfg.node.object_position_topic, Vector3, demo.object_position_callback)
    param_srv = Server(force_estimationConfig, demo.parameter_callback)
    rospy.spin()


import rosgraph

if __name__ == '__main__':
    if not rosgraph.is_master_online():
        print_error("Please run roscore before executing this script")
        raise Exception('roscore is not ready')
    else:
        main()


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
