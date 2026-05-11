"""
Planner class
Implementation of A*
"""

import copy
import heapq
import math
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None

    def get_neighbors(self, current_cell):
        """ Return list of free (i.e. not obstacle) neighbour cells 
            with the format of current_cell: (i, j) in the map frame
        """
        # TODO for TP5: iterate through neighbors and add free ones to neighbor_list

        # list to add neighbors        
        neighbor_list = []
        x, y = current_cell

        # adjacent cells to current cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:

                # if current cell
                if dx == 0 and dy == 0:
                    continue
                
                neighbor = (x + dx, y + dy)
                
                # check grid limits to add neighbor into list
                if (0 <= neighbor[0] < self.grid.x_max_map) and (0 <= neighbor[1] < self.grid.y_max_map):
                    neighbor_list.append(neighbor)

        return neighbor_list

    def heuristic(self, cell_1: Tuple[int, int], cell_2: Tuple[int, int]):
        """ Return heuristic goal distance """
        # TODO for TP5: compute heuristic distance between cell_1 and cell_2
        
        # simple euclidean distance
        h = np.sqrt((cell_1[0] - cell_2[0])**2 + (cell_1[1] - cell_2[1])**2)

        return h

    def reconstruct_path(self, came_from, goal):
        """ Extract path after cost computation """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)

        total_path = np.array(total_path)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(total_path[:, 0], total_path[:, 1])
        return np.vstack((traj_world_x, traj_world_y))

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        start: Tuple[int, int] = self.grid.conv_world_to_map(start[0], start[1])
        goal: Tuple[int, int] = self.grid.conv_world_to_map(goal[0], goal[1])
    
        # creates a copy of occupancy map to modify it and take into account
        # a margin in the walls
        self.map_walls = copy.deepcopy(self.grid.occupancy_map)
        
        # TODO for TP5: dilate walls in self.map_walls to take into account a margin around obstacles
        
        # initialize kernel to dilate walls
        kernel = np.ones((15, 15), np.uint8)
        wall_mask = (self.map_walls > 0).astype(np.uint8)
        self.map_walls = cv2.dilate(wall_mask, kernel, iterations=1)    

        #cv2.imshow("map_walls", self.map_walls)

        # min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)

        # dictionary to trace back route
        came_from = {}

        # cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0

        # best guess of cost for each cell (cost + heuristic)
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = 0.0 + self.heuristic(start, goal)

        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            current_f, current_cell = current
            # lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)

            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                # skip cell if it is a wall
                if self.map_walls[cell[0], cell[1]] > 0:
                    continue
                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, cell)
                if tentative_g_score < g_score[cell]:
                    # better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + self.heuristic(cell, goal)
                    heapq.heappush(open_set, (f_score[cell], cell))

        # goal was never reached
        print('failed getting to objective')
        return None
    
    def explore_frontiers(self, robot_pose):
        """
        Frontier based exploration
        """

        occupancy = self.grid.occupancy_map
        
        # masks to different cell occupation levels
        free_mask    = occupancy < -1.0
        unknown_mask = np.abs(occupancy) < 0.2
        wall_mask    = occupancy > 1.0

        # dilate walls to avoid them
        walls_dilated = cv2.dilate(wall_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        # dilate unknown cells to choose them
        unknown_dilated = cv2.dilate(unknown_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        frontier_mask = free_mask & (unknown_dilated > 0) & (walls_dilated == 0)
        
        frontier_cells = np.argwhere(frontier_mask)  

        # robot position in world coordinates
        robot_cell = self.grid.conv_world_to_map(float(robot_pose[0]), float(robot_pose[1]))
        robot_cell_rc = np.array([
            int(np.clip(robot_cell[1], 0, self.grid.y_max_map - 1)),
            int(np.clip(robot_cell[0], 0, self.grid.x_max_map - 1))])

        # sort frontier cells by distance 
        distances = np.linalg.norm(frontier_cells - robot_cell_rc, axis=1)
        sorted_indices = np.argsort(distances)

        # test nearest candidates
        for idx in sorted_indices[:20]:

            # get coordinates of candidate cell
            cell = frontier_cells[idx]  
            fx, fy = self.grid.conv_map_to_world(int(cell[1]), int(cell[0]))
            fx = float(np.clip(fx, self.grid.x_min_world + 10, self.grid.x_max_world - 10))
            fy = float(np.clip(fy, self.grid.y_min_world + 10, self.grid.y_max_world - 10))
            goal_world = np.array([fx, fy, 0.0])

            # try to find path
            path = self.plan(robot_pose, goal_world)
            if path is not None:
                print(f"Frontier at ({fx:.1f}, {fy:.1f}), dist={distances[idx]:.1f} cells")
                return goal_world, path

        print("No reachable frontier in top 20 candidates")
        return None, None