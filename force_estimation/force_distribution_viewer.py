# -*- coding: utf-8 -*-

import colorsys
import numpy as np
from force_estimation import rviz_client
from typing import List


class ForceDistributionViewer:
    """
        Singleton pattern
        Duplicated instantiation causes error in ROS node intialization
    """

    _unique_instance = None

    def __new__(cls):
        raise NotImplementationError('Cannot generate instance by constructor')

    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    @classmethod
    def get_instance(cls):
        if not cls._unique_instance:
            cls._unique_instance = cls.__internal_new__()
            cls.rviz_client = rviz_client.RVizClient()
            cls._unique_instance.set_bin_RGBA([0.5, 0.5, 0.5, 0.4])
        return cls._unique_instance

    def set_object_info(self, object_info):
        self._object_info = object_info

    def set_bin_RGBA(self, rgba):
        self._rgba = rgba

    def publish_bin_state(self,
                          bin_state,
                          fmap,
                          draw_fmap=True,
                          draw_force_gradient=False,
                          draw_range=[0.5, 0.9],
                          refresh=True,
                          draw_bin_objects=True):
        """
            deprecated, use draw_bin_state()
        """

        if refresh:
            self.rviz_client.delete_all()
        if draw_bin_objects:
            self.draw_bin_objects(fmap)

        positions = fmap.get_positions()
        fvals = fmap.get_values()

        if bin_state is not None:
            self.draw_objects(bin_state)
        if draw_fmap:
            self.draw_force_distribution(positions, fvals, draw_range=draw_range)
        if draw_force_gradient:
            self.draw_force_gradient(positions, fvals)
        self.rviz_client.show()

    def draw_bin_state(self,
                       bin_state,
                       fmap,
                       draw_fmap=True,
                       draw_force_gradient=False,
                       draw_range=[0.5, 0.9],
                       refresh=True,
                       draw_bin_objects=True):

        if refresh:
            self.rviz_client.delete_all()
        if draw_bin_objects:
            self.draw_bin_objects(fmap)

        positions = fmap.get_positions()
        fvals = fmap.get_values()

        if bin_state is not None:
            self.draw_objects(bin_state)
        if draw_fmap:
            self.draw_force_distribution(positions, fvals, draw_range=draw_range)
        if draw_force_gradient:
            self.draw_force_gradient(positions, fvals)
        self.rviz_client.show()

    def draw_bin_objects(self, fmap):
        scene = fmap.get_scene()
        if scene == 'seria_basket':
            mesh_file = 'env/seria_basket.dae'
            mesh_pose = ([0., 0., 0.73], [0., 0., 0.70711, 0.70711])
            scale = [1., 1., 1.]
        elif scene == 'konbini_shelf':
            mesh_file = 'env/simple_shelf.obj'
            mesh_pose = ([0., 0., 0.], [0., 0., 0., 1.])
            scale = [0.01, 0.01, 0.01]
        elif scene == 'small_table':
            mesh_file = 'env/table_surface.obj'
            mesh_pose = ([0., 0., 0.68], [0., 0., 0., 1.])
            scale = [1., 1., 1.]
        else:
            print(f'[VIEWER] unknown scene: {scene}')
            return

        self.rviz_client.draw_mesh_file(f"package://force_estimation/meshes/{mesh_file}",
                                        mesh_pose,
                                        rgba=self._rgba,
                                        scale=scale)

    def draw_objects(self, bin_state):
        for object_state in bin_state:
            name, pose = object_state
            mesh_file, scale = self._object_info.rviz_mesh_file(name)
            assert mesh_file, f"mesh file for {name} not found"
            self.rviz_client.draw_mesh_file(mesh_file, pose, self._rgba)

    def draw_force_distribution(self, positions, fvals, draw_range=[0.5, 0.9]):
        fvals = fvals.flatten()
        fmax = np.max(fvals)
        fmin = np.min(fvals)
        points = []
        rgbas = []
        if fmax - fmin < 1e-3:
            print('the range of force values too small')
            return

        for (x, y, z), f in zip(positions, fvals):
            if draw_range[0] <= f and f <= draw_range[1]:
                points.append([x, y, z])
                hue = max(0, (0.7 - f) / 0.7)
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                rgbas.append([r, g, b, 1.0])

        self.rviz_client.draw_points(points, rgbas)

    def draw_mesh(self,
                  vertices: List[List[float]],
                  faces: List[List[float]],
                  colors: List[List[float]],
                  frame_id="map"):
        self.rviz_client.draw_mesh(vertices, faces, colors)

    def draw_vector_field(self, positions, values, scale=0.5, frame_id="map"):
        self.rviz_client.draw_arrows(positions,
                                     positions + values * scale,
                                     rgba=[1.0, 0.0, 0.0, 1.0],
                                     scale=[0.01, 0.01, 0.0],
                                     frame_id=frame_id)

    def draw_force_gradient(self, positions, fvals, scale=0.3, threshold=0.008):
        gxyz = np.gradient(- fvals)
        g_vecs = np.column_stack([g.flatten() for g in gxyz])
        pos_val_pairs = [(p, g) for (p, g) in zip(positions, g_vecs) if np.linalg.norm(g) > threshold]
        positions, values = zip(*pos_val_pairs)
        self.draw_vector_field(np.array(positions), np.array(values), scale=scale)

    def draw_calibration_objects(self):
        rgba = [1.,1.,1.,0.6]
        height = 0.74
        self.rviz_client.draw_cube([0.045,0.075,height], rgba, [0.09,0.15,0.03])
        self.rviz_client.draw_cube([0.045,-0.075,height], rgba, [0.09,0.15,0.03])
        self.rviz_client.draw_cube([-0.045,0.075,height], rgba, [0.09,0.15,0.03])
        self.rviz_client.draw_cube([-0.045,-0.075,height], rgba, [0.09,0.15,0.03])        

    def set_static_transform(self, translation: list, rotation: list = [0., 0., 0., 1.], parent_frame='map', child_frame='base'):
        self.rviz_client.set_static_transform(translation, rotation, parent_frame, child_frame)

    def set_joint_positions(self, joint_names, positions):
        self.rviz_client.set_joint_positions(joint_names, positions)

    def load_urdf(self, urdf_path: str):
        self.rviz_client.publish_robot_description(urdf_path)

    def clear(self):
        self.rviz_client.delete_all()

    def show(self):
        self.rviz_client.show()