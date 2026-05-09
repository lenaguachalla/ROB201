""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        score = 0

        # erase points after maximum distance of laser
        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        max_range = lidar.max_range

        mask = distances < max_range
        distances = distances[mask]
        angles = angles[mask]

        # estimate positions of detection in world coordinates
        x = np.cos(angles + pose[2]) * distances + pose[0]
        y = np.sin(angles + pose[2]) * distances + pose[1]

        # convert positions into index of cells in the grid  
        x_map, y_map = self.grid.conv_world_to_map(x, y)

        #erase points outise of grid limits
        mask = (x_map >= 0) & (x_map < self.grid.x_max_map) & (y_map >= 0) & (y_map < self.grid.y_max_map)
        x_map = x_map[mask]
        y_map = y_map[mask]

        # read and add values of the cells in the grid to calculate score
        score = np.sum(self.grid.occupancy_map[x_map, y_map])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        d = np.sqrt(odom_pose[0]**2 + odom_pose[1]**2)
        alpha = np.arctan2(odom_pose[1], odom_pose[0])

        x = odom_pose_ref[0] + d * np.cos(alpha + odom_pose_ref[2])
        y = odom_pose_ref[1] + d * np.sin(alpha + odom_pose_ref[2])
        theta = odom_pose_ref[2] + odom_pose[2]        

        corrected_pose = np.array([x, y, theta])

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = 0
        i = 0
        n = 100
        sigma = [2, 2, 0.1]

        pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar,pose)
        best_pose_ref = self.odom_pose_ref.copy()

        while i < n:
            # random offset with gaussian distribution
            offset = np.random.normal(0, sigma)
            pose_ref = best_pose_ref + offset

            # compute score of this pose
            pose = self.get_corrected_pose(raw_odom_pose, pose_ref)
            score = self._score(lidar, pose)

            # update best pose if better score
            if score > best_score:
                best_score = score
                best_pose_ref = pose_ref
                i = 0
            else:
                i += 1

        self.odom_pose_ref = best_pose_ref            

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3

        # convert lidar data from polar coordinates in robot frame to global cartesian coordinates 
        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        
        x_list = np.cos(angles + pose[2]) * distances + pose[0]
        y_list = np.sin(angles + pose[2]) * distances + pose[1]

        # update points on the line between robot and point with weak probability
        for x, y in zip(x_list, y_list):
            self.grid.add_value_along_line(pose[0], pose[1], x, y, val=-0.95)

        # update points with strong probability
        self.grid.add_map_points(x_list, y_list, val=2)

        # threshold to avoid divergences
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -20, 20)