# -*- coding: utf-8 -*-

import rclpy
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import numpy as np
import threading
from rclpy.executors import ExternalShutdownException


def normalizeQuaternion(quaternion_msg):
    norm = quaternion_msg.x**2 + quaternion_msg.y**2 + quaternion_msg.z**2 + quaternion_msg.w**2
    s = norm**(-0.5)
    quaternion_msg.x *= s
    quaternion_msg.y *= s
    quaternion_msg.z *= s
    quaternion_msg.w *= s


def spin_in_background():
    executor = rclpy.get_global_executor()
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def to_Vector3(v):
    m = Vector3()
    m.x = v[0]
    m.y = v[1]
    m.z = v[2]
    return m


def to_Point(p):
    m = Point()
    m.x = p[0]
    m.y = p[1]
    m.z = p[2]
    return m


def to_ColorRGBA(c):
    m = ColorRGBA()
    m.r = c[0]
    m.g = c[1]
    m.b = c[2]
    m.a = c[3]
    return m


def add_control(interaction_marker, name, orientation, interaction_mode):
    control = InteractiveMarkerControl()
    control.name = name
    control.orientation.w = float(orientation[0])
    control.orientation.x = float(orientation[1])
    control.orientation.y = float(orientation[2])
    control.orientation.z = float(orientation[3])
    normalizeQuaternion(control.orientation)
    control.interaction_mode = interaction_mode
    control.orientation_mode = InteractiveMarkerControl.FIXED
    interaction_marker.controls.append(control)


class RVizClient:
    def __init__(self, 
                 node_name='forcemap_client', 
                 base_frame='fmap_frame', 
                 marker_scale=0.1,
                 use_6dof_marker=False):
        self._message_id = 0
        self._node_name = node_name
        self._base_frame_id = base_frame
        self._6dof = use_6dof_marker
        self._interactive_marker_scale = marker_scale
        self._markerArray = MarkerArray()
        self.start_ros_node()

    def __del__(self):
        self._pub.unregister()
        self._executor_thread.join()

    def _now(self):
        return rclpy.clock.Clock().now().to_msg()

    def start_ros_node(self):
        rclpy.init()
        self._executor_thread = threading.Thread(target=spin_in_background)
        self._executor_thread.start()
        self._node = rclpy.create_node(self._node_name)
        rclpy.get_global_executor().add_node(self._node)
        self._pub = self._node.create_publisher(MarkerArray, "scene_objects", 1)
        self._rate = self._node.create_rate(30)  # 30Hz

        self._interactive_marker_pose = (np.array([0., 0., 0.78]), np.array([0., 0., 0., 1.]))

        self._interactive_marker_server = InteractiveMarkerServer(self._node, namespace='simple_marker')
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._base_frame_id
        int_marker.name = "object_center"
        int_marker.description = "Object Center"
        int_marker.pose.position = to_Point(self._interactive_marker_pose[0])
        int_marker.scale = self._interactive_marker_scale

        box_marker = self._make_marker(Marker.SPHERE)
        box_marker.pose = int_marker.pose
        box_marker.scale = to_Vector3([0.005, 0.005, 0.005])
        box_marker.color = to_ColorRGBA([1.0, 0.0, 0.0, 0.5])

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(box_marker)
        int_marker.controls.append(control)

        add_control(int_marker, name='move_x', orientation=[1,1,0,0], interaction_mode=InteractiveMarkerControl.MOVE_AXIS)
        add_control(int_marker, name='move_y', orientation=[1,0,1,0], interaction_mode=InteractiveMarkerControl.MOVE_AXIS)
        add_control(int_marker, name='move_z', orientation=[1,0,0,1], interaction_mode=InteractiveMarkerControl.MOVE_AXIS)
        if self._6dof:
            add_control(int_marker, name='rotate_x', orientation=[1,1,0,0], interaction_mode=InteractiveMarkerControl.ROTATE_AXIS)
            add_control(int_marker, name='rotate_y', orientation=[1,0,1,0], interaction_mode=InteractiveMarkerControl.ROTATE_AXIS)
            add_control(int_marker, name='rotate_z', orientation=[1,0,0,1], interaction_mode=InteractiveMarkerControl.ROTATE_AXIS)

        self._interactive_marker_server.insert(int_marker, feedback_callback=self.processInteractiveMarkerFeedback)
        self._interactive_marker_server.applyChanges()
        self._interactiveMarkerCallback = None

        self._interactive_marker_pose = (np.array([0, 0, 0.78]), np.array([0, 0, 0, 1]))

    def processInteractiveMarkerFeedback(self, feedback):
        p = feedback.pose.position
        q = feedback.pose.orientation
        self._interactive_marker_pose = (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))
        if self._interactiveMarkerCallback != None:
            self._interactiveMarkerCallback()

    # def getObjectPosition(self):
        return self._interactive_marker_pose[0]

    def getInteractiveMarkerPose(self):
        return self._interactive_marker_pose

    def addInteractiveMarkerCallback(self, func):
        self._interactiveMarkerCallback = func

    def show(self):
        self._pub.publish(self._markerArray)

    def delete_all(self):
        self._message_id = 0
        markerD = Marker()
        markerD.header.frame_id = self._base_frame_id
        markerD.action = markerD.DELETEALL
        self._markerArray = MarkerArray()
        self._markerArray.markers.append(markerD)

    def draw_mesh(self, mesh_file, pose, rgba, scale=[1., 1., 1.]):
        marker = self._make_marker(Marker.MESH_RESOURCE)
        marker.mesh_resource = mesh_file
        marker.mesh_use_embedded_materials = True

        xyz, quat = pose
        marker.pose.position.x = xyz[0]
        marker.pose.position.y = xyz[1]
        marker.pose.position.z = xyz[2]
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        self._markerArray.markers.append(marker)

    def draw_points(self, points, rgbas, point_size=0.0015):
        marker = self._make_marker(Marker.POINTS)

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = point_size
        marker.scale.y = point_size
        marker.scale.z = point_size

        for position, rgba in zip(points, rgbas):
            marker.points.append(to_Point(position))
            marker.colors.append(to_ColorRGBA(rgba))

        self._markerArray.markers.append(marker)

    def _make_marker(self, marker_type):
        marker = Marker()
        marker.type = marker_type
        marker.header.frame_id = self._base_frame_id
        marker.header.stamp = self._now()
        marker.lifetime = rclpy.duration.Duration().to_msg()
        marker.id = self._message_id
        marker.action = marker.ADD
        self._message_id += 1
        return marker

    def draw_arrow(self, tail, tip, rgba, scale):
        marker = self._make_marker(Marker.ARROW)
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = to_Vector3(scale)
        marker.color = to_ColorRGBA(rgba)
        marker.points = [to_Point(tail), to_Point(tip)]
        self._markerArray.markers.append(marker)

    def draw_arrows(self, tails, tips, rgba=[0.2, 0.5, 1.0, 0.3], scale=[0.001, 0.001, 0.000]):
        """
        This methods is very slow because it sends a topic for each arrow.
        """
        for tail, tip in zip(tails, tips):
            self.draw_arrow(tail, tip, rgba, scale)

    def draw_sphere(self, center, rgba, scale):
        marker = self._make_marker(Marker.SPHERE)
        marker.pose.position = to_Point(center)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = to_Vector3(scale)
        marker.color = to_ColorRGBA(rgba)
        self._markerArray.markers.append(marker)

    def draw_cube(self, center, rgba, scale):
        marker = self._make_marker(Marker.CUBE)
        marker.pose.position = to_Point(center)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = to_Vector3(scale)
        marker.color = to_ColorRGBA(rgba)
        self._markerArray.markers.append(marker)