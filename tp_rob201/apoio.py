''' 

Functions to test in my_robot_slam.py for frontier based exploration

    def control_frontiers(self):
        """
        Control function for frontier based exploration
        """
        pose = self.odometer_values()

        if self.counter < 10:
            self.tiny_slam.update_map(self.lidar(), pose)

        if self.counter > 10:
            score = self.tiny_slam.localise(self.lidar(), pose)
            corrected_pose = self.tiny_slam.get_corrected_pose(pose)

            if score > 50:
                self.tiny_slam.update_map(self.lidar(), corrected_pose)
                
            else:
                self.tiny_slam.update_map(self.lidar(), pose)

        if self.counter < 2500:

            command = potential_field_control(self.lidar(), pose, self.goal)

            goal_dist = np.linalg.norm(pose[:2] - np.array(self.goal[:2]))

            reached = goal_dist < 30
            stuck   = self.frontier_stuck_counter > 300

            if self.counter > 100:

                if reached or stuck or self.frontier_goal_world is None:

                    if reached:
                        print(f"Frontier reached after {self.frontier_stuck_counter} steps")
                        self.visited_frontiers.append(self.goal[:2].copy())
                    if stuck:
                        print(f"Stuck at frontier, marking inaccessible")
                        self.visited_frontiers.append(self.goal[:2].copy())

                    self.frontier_stuck_counter = 0

                    # Try frontiers until we find a reachable one
                    new_goal = None
                    for _ in range(10):  # max 10 attempts
                        candidate = self.planner.explore_frontiers(pose, self.visited_frontiers)
                        if candidate is None:
                            print("No more frontiers, switching to return phase")
                            self.counter = 2500
                            break

                        # Validate reachability with A*
                        test_path = self.planner.plan(pose, candidate)
                        if test_path is not None:
                            new_goal = candidate
                            break
                        else:
                            # Not reachable — mark as visited so we skip it next time
                            print(f"Frontier at {candidate[:2]} unreachable, skipping")
                            self.visited_frontiers.append(candidate[:2])

                    if new_goal is not None:
                        self.goal = new_goal
                        self.frontier_goal_world = new_goal
                        print(f"New frontier goal: {self.goal[:2]}")

                else:
                    self.frontier_stuck_counter += 1

            if self.counter % 10 == 0:
                self.occupancy_grid.display_cv(pose, self.goal)

        else:

            if self.path is None:
                start = pose  # use corrected pose, not raw odometry
                self.path = self.planner.plan(start, np.array([0, 0, 0]))

                if self.path is not None:
                    self.path = self.path.T  # now shape (N, 2)
                    self.path_index = 0
                    print(f"Path found: {len(self.path)} waypoints")
                else:
                    print("Planning failed")

            if self.path is not None and self.path_index < len(self.path):
                target = self.path[self.path_index]

                # Distance in corrected frame
                dist = np.linalg.norm(target - pose[:2])
                print(f"waypoint {self.path_index}/{len(self.path)}, dist={dist:.1f}")

                if dist < 30.0:  
                    self.path_index += 1

                if self.path_index < len(self.path):
                    target = self.path[self.path_index]

                target_pose = np.array([target[0], target[1], 0.0])
                command = potential_field_control(self.lidar(), pose, target_pose)

            else:
                # Reached end of path or no path — go directly to origin
                target_pose = np.array([0.0, 0.0, 0.0])
                command = potential_field_control(self.lidar(), pose, target_pose)

                if np.linalg.norm(pose[:2]) < 20:
                    print("Reached origin!")
                    command = {'forward': 0, 'rotation': 0}

            if self.counter % 10 == 0:
                traj = self.path.T if self.path is not None else None
                self.occupancy_grid.display_cv(pose, np.array([0, 0, 0]), traj)

        return command
    
    def control_fr(self):
        """
        Control function for TP5
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()

        # update for initialization
        if self.counter < 10:
            self.tiny_slam.update_map(self.lidar(), pose)

        # lozalization and map update
        if self.counter > 10:
            score = self.tiny_slam.localise(self.lidar(), pose)
            corrected_pose = self.tiny_slam.get_corrected_pose(pose)

            # update map with corrected pose if it's good
            if score > 50:
                self.tiny_slam.update_map(self.lidar(), corrected_pose)
            else:
                self.tiny_slam.update_map(self.lidar(), pose)

        # time chosen to explore map before returning to origin
        if self.counter < 3500:

            # command for first goal set in initialization, updated with frontier exploration
            command = potential_field_control(self.lidar(), pose, self.goal)

            # checks distance to goal
            goal_dist = np.linalg.norm(pose[:2] - np.array(self.goal[:2]))

            # threshold to consider reaching goal
            reached = goal_dist < 30

            # time threshold to consider being stuck
            stuck = self.frontier_stuck_counter > 400

            # threshold to start checking frontiers, lets the map initialize
            if self.counter > 500:

                if reached or stuck or self.frontier_goal_world is None:

                    # frontier reached
                    if reached:
                        print(f"Frontier reached after {self.frontier_stuck_counter} steps")
                        self.visited_frontiers.append(self.goal[:2].copy())
                    
                    # robot stuck
                    if stuck:
                        print(f"Stuck at frontier, marking inaccessible")
                        self.visited_frontiers.append(self.goal[:2].copy())

                    self.frontier_stuck_counter = 0

                    # try frontiers until a reachable one is
                    new_goal = None
                    for _ in range(10):  # max 10 attempts
                        candidate = self.planner.explore_frontiers(pose, self.visited_frontiers)
                        if candidate is None:
                            print("No more frontiers, switching to return phase")
                            self.counter = 2500
                            break

                        # Validate reachability with A*
                        test_path = self.planner.plan(pose, candidate)
                        if test_path is not None:
                            new_goal = candidate
                            break
                        else:
                            # Not reachable — mark as visited so we skip it next time
                            print(f"Frontier at {candidate[:2]} unreachable, skipping")
                            self.visited_frontiers.append(candidate[:2])

                    if new_goal is not None:
                        self.goal = new_goal
                        self.frontier_goal_world = new_goal
                        print(f"New frontier goal: {self.goal[:2]}")

                else:
                    self.frontier_stuck_counter += 1

            if self.counter % 10 == 0:
                self.occupancy_grid.display_cv(pose, self.goal)

        else:

            if self.path is None:
                start = pose[:2]  # use corrected pose, not raw odometry
                self.path = self.planner.plan(start, np.array([0, 0]))

                if self.path is not None:
                    self.path = self.path.T  # now shape (N, 2)
                    self.path_index = 0
                    print(f"Path found: {len(self.path)} waypoints")
                else:
                    print("Planning failed")

            if self.path is not None and self.path_index < len(self.path):
                target = self.path[self.path_index]

                # Distance in corrected frame
                dist = np.linalg.norm(target - pose[:2])
                print(f"waypoint {self.path_index}/{len(self.path)}, dist={dist:.1f}")

                if dist < 30.0:  
                    self.path_index += 1

                if self.path_index < len(self.path):
                    target = self.path[self.path_index]

                target_pose = np.array([target[0], target[1], 0.0])
                command = potential_field_control(self.lidar(), pose, target_pose)

            else:
                # Reached end of path or no path — go directly to origin
                target_pose = np.array([0.0, 0.0, 0.0])
                command = potential_field_control(self.lidar(), pose, target_pose)

                if np.linalg.norm(pose[:2]) < 20:
                    print("Reached origin!")
                    command = {'forward': 0, 'rotation': 0}

            if self.counter % 10 == 0:
                traj = self.path.T if self.path is not None else None
                self.occupancy_grid.display_cv(pose, np.array([0, 0, 0]), traj)

        return command
    
    def control_test(self):
        """
        Control function for TP5
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        command = {'forward': 0, 'rotation': 0}

        # --- SLAM update ---
        if self.counter < 10:
            self.tiny_slam.update_map(self.lidar(), pose)
            corrected_pose = pose
        else:
            score = self.tiny_slam.localise(self.lidar(), pose)
            corrected_pose = self.tiny_slam.get_corrected_pose(pose)
            if score > 50:
                self.tiny_slam.update_map(self.lidar(), corrected_pose)
            else:
                self.tiny_slam.update_map(self.lidar(), pose)

        # --- Phase 1: frontier exploration ---
        if self.counter < 2500:

            goal_dist = np.linalg.norm(corrected_pose[:2] - np.array(self.goal[:2]))
            reached = goal_dist < 30
            stuck   = self.frontier_stuck_counter > 300

            if reached or stuck or self.path is None:
                if reached:
                    print(f"Frontier reached, {len(self.visited_frontiers)} visited so far")
                    self.visited_frontiers.append(self.goal[:2].copy())
                if stuck:
                    print("Stuck, marking frontier as inaccessible")
                    self.visited_frontiers.append(self.goal[:2].copy())
                    self.frontier_stuck_counter = 0

                new_goal, new_path = self.planner.explore_frontiers(corrected_pose, self.visited_frontiers)

                if new_goal is None:
                    print("No more frontiers, switching to return phase")
                    self.counter = 2500
                else:
                    self.goal = new_goal
                    self.path = new_path.T  # shape (N, 2)
                    self.path_index = 0
                    self.frontier_stuck_counter = 0
                    print(f"New frontier goal: {self.goal[:2]}, path: {len(self.path)} waypoints")

            # follow path waypoints
            if self.path is not None and self.path_index < len(self.path):
                target = self.path[self.path_index]
                if np.linalg.norm(target - corrected_pose[:2]) < 15:
                    self.path_index += 1
                if self.path_index < len(self.path):
                    target = self.path[self.path_index]
                command = potential_field_control(self.lidar(), corrected_pose,
                                                np.array([target[0], target[1], 0.0]))
            else:
                command = potential_field_control(self.lidar(), corrected_pose, self.goal)

            self.frontier_stuck_counter += 1

            if self.counter % 10 == 0:
                traj = self.path.T if self.path is not None else None
                self.occupancy_grid.display_cv(corrected_pose, self.goal, traj)

        # --- Phase 2: return to origin ---
        else:
            if self.path is None:
                print(f"Planning return path from {corrected_pose[:2]} to origin")
                self.path = self.planner.plan(corrected_pose, np.array([0, 0, 0]))
                if self.path is not None:
                    self.path = self.path.T  # shape (N, 2)
                    self.path_index = 0
                    print(f"Return path found: {len(self.path)} waypoints")
                else:
                    print("Return planning failed, going direct")

            if self.path is not None and self.path_index < len(self.path):
                target = self.path[self.path_index]
                dist = np.linalg.norm(target - corrected_pose[:2])
                if dist < 15:
                    self.path_index += 1
                if self.path_index < len(self.path):
                    target = self.path[self.path_index]
                command = potential_field_control(self.lidar(), corrected_pose,
                                                np.array([target[0], target[1], 0.0]))
            else:
                # end of path or no path — go directly to origin
                command = potential_field_control(self.lidar(), corrected_pose,
                                                np.array([0.0, 0.0, 0.0]))

            if np.linalg.norm(corrected_pose[:2]) < 20:
                print("Reached origin!")
                command = {'forward': 0, 'rotation': 0}

            if self.counter % 10 == 0:
                traj = self.path.T if self.path is not None else None
                self.occupancy_grid.display_cv(corrected_pose, np.array([0, 0, 0]), traj)

        return command
    
    def control_aaa(self):
        """
        Control function for TP5
        Main control function with full SLAM, frontier exploration and path planning
        """
        pose = self.odometer_values()

        # --- SLAM update ---
        if self.counter < 10:
            self.tiny_slam.update_map(self.lidar(), pose)

        if self.counter > 10:
            score = self.tiny_slam.localise(self.lidar(), pose)
            if score > 50:
                self.tiny_slam.update_map(self.lidar(), pose)

        # --- Plan path to next frontier every 1500 steps ---
        if self.counter % 1500 == 0:
            frontier_goal, frontier_path = self.planner.explore_frontiers(pose, self.visited_frontiers)

            if frontier_goal is not None and frontier_path is not None:
                self.path = frontier_path
                self.goal = frontier_goal
            else:
                # No more frontiers — plan return to origin
                print("No frontiers left, planning return to origin")
                self.path = self.planner.plan(pose, np.array([0, 0, 0]))
                if self.path is not None and self.path.shape[1] > 0:
                    self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])

        # --- Check if reached origin ---
        if self.path is not None and np.linalg.norm(pose[:2]) < 20:
            print("Origin reached")
            command = {'forward': 0, 'rotation': 0}

        else:
            command = potential_field_control(self.lidar(), pose, self.goal)

            if command == {'forward': 0, 'rotation': 0}:
                if self.path is not None and self.path.shape[1] > 0:
                    # Advance to next waypoint
                    self.goal = np.array([self.path[0, 0], self.path[1, 0], 0])
                    self.path = self.path[:, 1:]

                    # Mark frontier as visited when path is exhausted
                    if self.path.shape[1] == 0:
                        print(f"Frontier reached, {len(self.visited_frontiers) + 1} visited")
                        self.visited_frontiers.append(self.goal[:2].copy())
                else:
                    # No path — pick random goal as fallback
                    self.goal = np.random.uniform(low=[-600, -600, 0], high=[100, 100, 0])

            if self.counter % 10 == 0:
                self.occupancy_grid.display_cv(pose, self.goal, self.path)

        return command
'''

'''

Functions to test in planner.py (is line free and frontier exploration)

    def is_line_free(self, cell1, cell2):
        """
        Check if straight line between two map cells is free.
        Blocks on: walls (positive) OR unknown (near zero).
        Only passes through confirmed free (clearly negative) cells.
        """
        x0, y0 = int(cell1[0]), int(cell1[1])
        x1, y1 = int(cell2[0]), int(cell2[1])

        dist = np.linalg.norm([x1 - x0, y1 - y0])
        n_steps = max(int(dist * 3), 2)  # 3 samples per cell for better coverage

        xs = np.round(np.linspace(x0, x1, n_steps)).astype(int)
        ys = np.round(np.linspace(y0, y1, n_steps)).astype(int)

        # Skip first and last points (robot cell and frontier cell themselves)
        xs = xs[1:-1]
        ys = ys[1:-1]

        if len(xs) == 0:
            return True

        xs = np.clip(xs, 0, self.grid.x_max_map - 1)
        ys = np.clip(ys, 0, self.grid.y_max_map - 1)

        values = self.grid.occupancy_map[xs, ys]

        # Block if ANY cell along the line is a wall OR unknown
        has_wall    = np.any(values > 1.0)    # confirmed wall
        has_unknown = np.any(np.abs(values) < 1.0)  # never properly mapped

        if has_wall:
            return False
        if has_unknown:
            return False
        return True

    def explore_frontiers(self, robot_pose, visited_frontiers=None):
        """
        Frontier based exploration
        """
        if visited_frontiers is None:
            visited_frontiers = []

        occupancy = self.grid.occupancy_map
        free_mask    = occupancy < -0.5
        unknown_mask = np.abs(occupancy) < 2.0  # tuned: was 0.5, now catches more unknown cells
        wall_mask    = occupancy > 2.0          # tuned: was 0.5, avoids weak wall detections

        # Dilate walls for exclusion zone around obstacles
        wall_kernel = np.ones((9, 9), np.uint8)
        walls_dilated = cv2.dilate(wall_mask.astype(np.uint8), wall_kernel, iterations=1)

        # Frontier = free cell adjacent to unknown, not too close to wall
        unknown_kernel = np.ones((3, 3), np.uint8)
        unknown_dilated = cv2.dilate(unknown_mask.astype(np.uint8), unknown_kernel, iterations=1)

        frontier_mask = free_mask & (unknown_dilated > 0) & (walls_dilated == 0)
        
        if not np.any(frontier_mask):
            print("No frontiers found - exploration complete!")
            return None
        
        frontier_cells = np.argwhere(frontier_mask)  # [row, col]
       
        if len(frontier_cells) == 0:
            print("No valid frontiers after bounds filtering!")
            return None

        # Robot position in map frame
        robot_cell = self.grid.conv_world_to_map(robot_pose[0], robot_pose[1])
        robot_cell = (int(np.clip(robot_cell[0], 0, self.grid.x_max_map - 1)),
                    int(np.clip(robot_cell[1], 0, self.grid.y_max_map - 1)))
        robot_cell_arr = np.array([robot_cell[0], robot_cell[1]])  # [row, col]

        # Filter visited frontiers
        visit_radius = 30
        def is_visited(cell):
            for v in visited_frontiers:
                v_cell = self.grid.conv_world_to_map(v[0], v[1])
                if np.linalg.norm(np.array([v_cell[0], v_cell[1]]) - cell) < visit_radius:
                    return True
            return False

        frontier_cells = np.array([c for c in frontier_cells if not is_visited(c)])

        if len(frontier_cells) == 0:
            print("All frontiers visited!")
            return None

        # Sort by distance, check line of sight, pick nearest visible one
        distances = np.linalg.norm(frontier_cells - robot_cell_arr, axis=1)
        sorted_indices = np.argsort(distances)

        for idx in sorted_indices:
            cell = frontier_cells[idx]  # [row, col]
            # is_line_free takes [col, row] = [x, y] in map frame
            if self.is_line_free(robot_cell, (cell[1], cell[0])):
                fx, fy = self.grid.conv_map_to_world(cell[1], cell[0])
                fx = np.clip(fx, self.grid.x_min_world + 10, self.grid.x_max_world - 10)
                fy = np.clip(fy, self.grid.y_min_world + 10, self.grid.y_max_world - 10)
                print(f"Frontier at ({fx:.1f}, {fy:.1f}), dist={distances[idx]:.1f} cells")
                return np.array([fx, fy, 0.0])

        # No line-of-sight frontier found — return nearest regardless
        print("No LOS frontier found, returning nearest")
        nearest = frontier_cells[sorted_indices[0]]
        fx, fy = self.grid.conv_map_to_world(nearest[1], nearest[0])
        fx = np.clip(fx, self.grid.x_min_world + 10, self.grid.x_max_world - 10)
        fy = np.clip(fy, self.grid.y_min_world + 10, self.grid.y_max_world - 10)
        return np.array([fx, fy, 0.0])
    
    def explore_frontiers(self, robot_pose, visited_frontiers=None):
        """
        Frontier based exploration.
        Finds the nearest reachable frontier (free cell adjacent to unknown space).
        Returns (goal_world, path) or (None, None) if no frontier found.
        """
        if visited_frontiers is None:
            visited_frontiers = []

        occupancy = self.grid.occupancy_map
        free_mask    = occupancy < -1.0
        unknown_mask = (occupancy > -1.0) & (occupancy < 1.0)
        wall_mask    = occupancy > 1.0

        # Dilate walls to exclude frontiers too close to obstacles
        walls_dilated = cv2.dilate(wall_mask.astype(np.uint8),
                                np.ones((9, 9), np.uint8), iterations=1)

        # Frontier = free cell adjacent to unknown, not near a wall
        unknown_dilated = cv2.dilate(unknown_mask.astype(np.uint8),
                                    np.ones((3, 3), np.uint8), iterations=1)
        frontier_mask = free_mask & (unknown_dilated > 0) & (walls_dilated == 0)

        if not np.any(frontier_mask):
            print("No frontiers found - exploration complete!")
            return None, None

        # Remove frontier cells too close to map border
        margin = 5
        frontier_cells = np.argwhere(frontier_mask)  # shape (N, 2): [row, col]
        frontier_cells = frontier_cells[
            (frontier_cells[:, 0] >= margin) &
            (frontier_cells[:, 0] < self.grid.y_max_map - margin) &
            (frontier_cells[:, 1] >= margin) &
            (frontier_cells[:, 1] < self.grid.x_max_map - margin)
        ]

        if len(frontier_cells) == 0:
            print("No valid frontiers after border filtering!")
            return None, None

        # Robot position in map frame [row, col]
        robot_cell = self.grid.conv_world_to_map(float(robot_pose[0]), float(robot_pose[1]))
        robot_cell_rc = np.array([
            int(np.clip(robot_cell[1], 0, self.grid.y_max_map - 1)),
            int(np.clip(robot_cell[0], 0, self.grid.x_max_map - 1))
        ])

        # Filter out already visited frontiers
        visit_radius = 20  # cells
        def is_visited(cell):
            for v in visited_frontiers:
                v_cell = self.grid.conv_world_to_map(float(v[0]), float(v[1]))
                v_rc = np.array([v_cell[1], v_cell[0]])
                if np.linalg.norm(v_rc - cell) < visit_radius:
                    return True
            return False

        frontier_cells = np.array([c for c in frontier_cells if not is_visited(c)])
        if len(frontier_cells) == 0:
            print("All frontiers visited!")
            return None, None

        # Sort by distance to robot, test A* on the 20 nearest
        distances = np.linalg.norm(frontier_cells - robot_cell_rc, axis=1)
        sorted_indices = np.argsort(distances)

        for idx in sorted_indices[:20]:
            cell = frontier_cells[idx]  # [row, col]
            fx, fy = self.grid.conv_map_to_world(int(cell[1]), int(cell[0]))
            fx = float(np.clip(fx, self.grid.x_min_world + 10, self.grid.x_max_world - 10))
            fy = float(np.clip(fy, self.grid.y_min_world + 10, self.grid.y_max_world - 10))
            goal_world = np.array([fx, fy, 0.0])

            path = self.plan(robot_pose, goal_world)
            if path is not None:
                print(f"Frontier at ({fx:.1f}, {fy:.1f}), "
                    f"dist={distances[idx]:.1f} cells, reachable via A*")
                return goal_world, path

        print("No reachable frontier found in top 20 candidates")
        return None, None

'''