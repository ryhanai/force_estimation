# -*- coding: utf-8 -*-

import rclpy
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import numpy as np
import threading
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time


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


class RVizClient:
    def __init__(self, node_name='forcemap_client', base_frame='fmap_frame'):
        self._message_id = 0
        self._node_name = node_name
        self._base_frame_id = base_frame
        self._markerArray = MarkerArray()
        self.start_ros_node()

    def __del__(self):
        self._pub.unregister()
        self._executor_thread.join()

    def start_ros_node(self):
        rclpy.init()
        self._executor_thread = threading.Thread(target=spin_in_background)
        self._executor_thread.start()
        self._node = rclpy.create_node(self._node_name)
        rclpy.get_global_executor().add_node(self._node)
        self._pub = self._node.create_publisher(MarkerArray, "scene_objects", 1)

        self._object_position = np.array([0, 0, 0.78])

        self._interactive_marker_server = InteractiveMarkerServer(self._node, namespace='simple_marker')
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._base_frame_id
        int_marker.name = 'object_center'
        int_marker.description = 'Object Center'
        int_marker.pose.position = self.to_Point(self._object_position)
        int_marker.scale = 0.1

        box_marker = self._make_marker(Marker.SPHERE)
        box_marker.pose = int_marker.pose
        box_marker.scale = self.to_Vector3([0.005, 0.005, 0.005])
        box_marker.color = self.to_ColorRGBA([1.0, 0.0, 0.0, 0.0, 0.5])

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(box_marker)
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = 'move_x'
        control.orientation.w = 1.0
        control.orientation.x = 1.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = 'move_y'
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = 'move_z'
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 1.0
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)
        
        self._interactive_marker_server.insert(int_marker, feedback_callback=self.processInteractiveMarkerFeedback)
        self._interactive_marker_server.applyChanges()

        self._rate = self._node.create_rate(30)  # 30Hz

    def processInteractiveMarkerFeedback(self, feedback):
        p = feedback.pose.position
        self._object_position = np.array([p.x, p.y, p.z])

    def getObjectPosition(self):
        return self._object_position

    def show(self):
        self._pub.publish(self._markerArray)

    def delete_all(self):
        self._message_id = 0
        markerD = Marker()
        markerD.header.frame_id = self._base_frame_id
        markerD.action = markerD.DELETEALL
        self._markerArray.markers.clear()
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

        for position, rgba, in zip(points, rgbas):
            marker.points.append(self.to_Point(position))
            marker.colors.append(self.to_ColorRGBA(rgba))

        self._markerArray.markers.append(marker)

    def now(self):
        return rclpy.clock.Clock().now().to_msg()

    def to_Vector3(self, v):
        m = Vector3()
        m.x = v[0]
        m.y = v[1]
        m.z = v[2]
        return m

    def to_Point(self, p):
        m = Point()
        m.x = p[0]
        m.y = p[1]
        m.z = p[2]
        return m

    def to_ColorRGBA(self, c):
        m = ColorRGBA()
        m.r = c[0]
        m.g = c[1]
        m.b = c[2]
        m.a = c[3]
        return m

    def _make_marker(self, marker_type):
        marker = Marker()
        marker.type = marker_type
        marker.header.frame_id = self._base_frame_id
        marker.header.stamp = self.now()
        marker.lifetime = rclpy.duration.Duration().to_msg()
        marker.id = self._message_id
        marker.action = marker.ADD
        self._message_id += 1
        return marker

    def draw_arrow(self, tail, tip, rgba, scale):
        marker = self._make_marker(Marker.ARROW)
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = self.to_Vector3(scale)
        marker.color = self.to_ColorRGBA(rgba)
        marker.points = [self.to_Point(tail), self.to_Point(tip)]

        self._markerArray.markers.append(marker)

    def draw_arrows(self, tails, tips, rgba=[0.2, 0.5, 1.0, 0.3], scale=[0.001, 0.001, 0.000]):
        """
            This methods is very slow because it sends a topic for each arrow.
        """
        for tail, tip in zip(tails, tips):
            self.draw_arrow(tail, tip, rgba, scale)

    def draw_sphere(self, center, rgba, scale):
        marker = self._make_marker(Marker.SPHERE)
        marker.pose.position = self.to_Point(center)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = self.to_Vector3(scale)
        marker.color = self.to_ColorRGBA(rgba)
        self._markerArray.markers.append(marker)

    def draw_cube(self, center, rgba, scale):
        marker = self._make_marker(Marker.CUBE)
        marker.pose.position = self.to_Point(center)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = self.to_Vector3(scale)
        marker.color = self.to_ColorRGBA(rgba)
        self._markerArray.markers.append(marker)
