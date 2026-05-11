""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    # get lidar data
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # adjusting field of view to front of the robot 
    vision_angle = 20 * np.pi / 180   
    front_distances = distances[(angles > -vision_angle) & (angles < vision_angle)]

    # threshold for obstacle detection
    if np.min(front_distances) < 20:
        command = {
            "forward": 0.0,
            "rotation": 0.5
        }
    else:
        command = {
            "forward": 0.5, 
            "rotation": 0.0
        }

    return command

def wall_follow(lidar):
    """
    Simple wall following behavior
    lidar : placebot object with lidar data
    """
    # gets lidar data
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # gets vision from front and right sides of the robot
    vision_angle = 20 * np.pi / 180 
    front = np.mean(distances[np.abs(angles) < vision_angle])
    right = np.mean(distances[np.abs(angles + np.pi/2) < vision_angle])

    wall_dist       = 30   # desired distance from wall
    front_threshold = 60  # obstacle ahead threshold
    no_wall_dist    = 200  # distance to no wall detected

    # turn left for obstacle ahead
    if front < front_threshold:
        forward  = 0.3
        rotation = 1.0

    # go straight if no wall on the right
    elif right > no_wall_dist:
        forward  = 0.5
        rotation = 0.0

    # turn left if close to right wall
    elif right < wall_dist:
        forward  = 0.5
        rotation = 0.4

    # turn right if far from right wall
    elif right > wall_dist:
        forward  = 0.5
        rotation = -0.4

    # go straight if good distance
    else:
        forward  = 0.5
        rotation = 0.0

    return {"forward": forward, "rotation": rotation}

def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    # parameters
    K_goal = 1
    K_obs = 8500
    safe_dist = 20.0

    # robot position
    curr_pos = current_pose[:2]
    goal_pos = goal_pose[:2]
    theta = current_pose[2]

    # distance between robot and goal
    goal_dist = np.linalg.norm(goal_pos - curr_pos)
    
    # stop condition when close to goal
    if goal_dist < 30.0:
        return {"forward": 0.0, "rotation": 0.0}
    else:
        # calculates attractive gradient
        grad_attr = K_goal * (goal_pos - curr_pos) / goal_dist
    
    # reference shift to robot frame 
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, s], [-s, c]])
    grad_attr = rot_matrix @ grad_attr
    
    # repulsive gradient
    grad_rep = np.zeros(2)
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    # sum repulsive gradient of close objects
    for d, angle in zip(distances, angles):
        if (d < safe_dist) and (d > 0.1):
            mag = K_obs * (1.0/d - 1.0/safe_dist) * (1.0/d**2)
            grad_rep -= mag * np.array([np.cos(angle), np.sin(angle)]) 

    # final potential field gradient
    grad_final = grad_attr + grad_rep
    
    # forward and rotation speed according to gradient
    speed = np.linalg.norm(grad_final)
    # rotation speed is very sensible, set small angle gain
    rotation = np.arctan2(grad_final[1], grad_final[0]) * 0.5

    # clips speeds to avoid bad cartography
    speed = np.clip(speed, -0.4, 0.4)
    rotation = np.clip(rotation, -0.2, 0.2)
    
    command = {"forward": speed,
               "rotation": rotation}
    
    return command