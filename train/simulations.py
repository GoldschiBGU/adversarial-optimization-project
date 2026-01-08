import simulations_helpfunc as simu
import matplotlib.pyplot as plt
import math
import os
import random

# Set seeds for stability
SEED = 42
random.seed(SEED)


class DroneSimulation:
    """Simulation class for different levels"""

    def __init__(self, time_step, interceptors, detectors, detector_range, target_coords, interceptor_delay, border_y):
        # Constant Definitions
        self.N = 5.0
        self.time_step = time_step
        self.interceptors = interceptors
        self.detectors = detectors
        self.detector_range = detector_range
        self.offense_drone_speed = 0.05  # km/sec
        self.interceptor_delay = interceptor_delay
        self.interception_predict_steps = interceptor_delay * time_step
        self.interceptor_speed = 0.03  # km/sec
        self.border_y = border_y
        self.y_map_lim = (0, 115)
        self.x_map_lim = (0, 80)
        self.all_targets = target_coords

        # Dynamic Definitions to change in different levels
        self.target_coords = target_coords
        self.num_drones = 20
        self.level = 1
        self.start_pos = [(40, self.border_y + 45)]
        self.intelligence = "no"

    def levelup(self):
        self.level += 1

    def leveldown(self):
        self.level -= 1

    def update_interceptors(self, interceptors):
        self.interceptors = interceptors

    def update_detectors(self, detectors):
        self.detectors = detectors

    def init_level2(self):
        match self.level:
            case 0:
                print("Learning Failed")
                return 0
            case 1:
                self.num_drones = 30
                self.interceptor_speed = 0.03  # km/sec
                self.intelligence = "no"
                self.start_pos = [(40, self.border_y + 45)]
                self.target_coords = self.all_targets[:4]
                print(
                    f"Initialized Level 1: Basic drone wave with {self.num_drones} Drones and {len(self.target_coords)} targets.")
                return 1
            case 2:
                self.num_drones = 30
                self.interceptor_speed = 0.03  # km/sec
                self.intelligence = "no"
                self.start_pos = [(0, self.border_y + 45), (40, self.border_y + 45), (80, self.border_y + 45)]
                self.target_coords = self.all_targets[:4]
                print(
                    f"Initialized Level 2: Basic drone wave with {self.num_drones} Drones and {len(self.target_coords)} targets.")
                return 1
            case 3:
                self.num_drones = 30
                self.interceptor_speed = 0.03  # km/sec
                self.intelligence = "no"
                self.start_pos = [(0, self.border_y + 45), (40, self.border_y + 45), (80, self.border_y + 45)]
                self.target_coords = self.all_targets
                print(f"Initialized Level 3: Basic drone wave with {self.num_drones} Drones and {len(self.target_coords)} targets.")
                return 1
            case 4:
                self.num_drones = 30
                self.interceptor_speed = 0.03  # km/sec
                self.intelligence = "partial"
                self.start_pos = [(0, self.border_y + 45), (40, self.border_y + 45), (80, self.border_y + 45)]
                self.target_coords = self.all_targets
                print(
                    f"Initialized Level 4: Intelligent drone wave with {self.num_drones} Drones and {len(self.target_coords)} targets.")
                return 1
            case 5:
                self.num_drones = 30
                self.interceptor_speed = 0.03  # km/sec
                self.intelligence = "yes"
                self.start_pos = [(0, self.border_y + 45), (40, self.border_y + 45), (80, self.border_y + 45)]
                self.target_coords = self.all_targets
                print(f"Initialized Level 5: Intelligent drone wave with {self.num_drones} Drones and {len(self.target_coords)} targets.")
                return 1
            case 6:
                self.num_drones = 30
                self.interceptor_speed = 0.03  # km/sec
                self.intelligence = "yes"
                self.start_pos = [(40, self.border_y + 45)]
                self.target_coords = self.all_targets
                print(
                    f"Initialized Level 6: Intelligent drone wave with {self.num_drones} Drones and {len(self.target_coords)} targets.")
                return 1
            case 7:
                print("Learning succeed !!")
                return 0

    def simulate(self):
        # The level of the simulation is set based on the following:
        #       Number of drones = [20, 100]
        #       Number of targets for the offense drone
        #       Initial positions of the attackers [(0 -> 80, 115)]
        #       A* with or without intelligence
        intercepted_points = []
        target_hits = 0
        intercept_hits = 0
        inter_reward = 0
        detect_reward = 0
        # Initialize score list for the interceptors with default values (e.g., 0 for each coordinate)
        intercept_list = []
        detect_list = []
        for coord in self.interceptors:
            intercept_list.append(coord)
            intercept_list.append(0)  # Default value
        for coord in self.detectors:
            detect_list.append(coord)
            detect_list.append(0)

        for drone_id in range(self.num_drones):
            start = (random.choice(self.start_pos))  # Start above the defense border
            if "no" in self.intelligence:
                intercepted_points = []
            if "partial" in self.intelligence:  # Partial intelligence to guide the learning
                if drone_id % 10 == 0:
                    intercepted_points = []

            # Initialize the offensive drone
            off_drone = simu.Drone(position=start, velocity=self.offense_drone_speed, angle=math.radians(0))
            # Create a copy of the offensive drone to monitor last position in case of no visibility
            last_seen_drone = simu.Drone(position=off_drone.position, velocity=off_drone.velocity, angle=off_drone.angle)
            target = (random.choice(self.target_coords))
            if intercepted_points:
                drop_zones = simu.generate_drop_zones(intercepted_points, radius=3, weight=5, stretch_factor=4,
                                                      step_size=0.5)
                path = simu.a_star_drone(start, target, self.x_map_lim, self.y_map_lim, 0.5, drop_zones)
            else:
                path = simu.a_star_drone(start, target, self.x_map_lim, self.y_map_lim, 0.5, intercepted_points)

            if not path:
                print("Path Was not found")
                continue  # Skip if no valid path found

            # Reset Variables
            vision = False
            interceptor_assign = False
            last_index = 0  # Start tracking from first path point
            delay = 0

            # Start the flight
            for step in range(1000000):  # Allow enough iterations for the drone to reach the target
                # start follow the path
                last_index = simu.pure_pursuit_target(off_drone, path, lookahead_dist=2.0, time_step=self.time_step,
                                                 last_index=last_index)
                # detection scan
                for detection in self.detectors:
                    if simu.is_in_range(off_drone.position, detection, self.detector_range, shape=2, detector_angle=detection[2]):
                        vision = True
                        simu.set_score(detect_list, detection)
                        # save "last seen" data for when the offense drone left the detection area
                        last_seen_drone.position = off_drone.position
                        last_seen_drone.angle = off_drone.angle
                        detect_reward += 1
                        break
                    else:
                        vision = False
                        if off_drone.position[1] < self.border_y and self.detectors[-1] == detection:
                            detect_reward -= 1

                if vision and not interceptor_assign:  # If the offensive drone is in sight of a detector and
                    # an intercepted has not assigned yet
                    inter_reward += 10
                    interceptor_assign, interceptor, assigned_inter = simu.assign_interceptor(self.interceptors, off_drone,
                                                                                              self.interceptor_speed,
                                                                                              self.interception_predict_steps)

                if vision and interceptor_assign:
                    # if the offensive drone is in sight and an interceptor is assigned
                    # Apply launch delay
                    delay += 1
                    if delay > self.interceptor_delay:
                        # After delay start moving towards the offensive drone
                        simu.proportional_navigation(interceptor, off_drone, self.time_step, self.N)

                if interceptor_assign and not vision:  # If an interceptor is assigned and there is no vision
                    delay += 1
                    if delay > self.interceptor_delay:
                        # Onboard vision
                        # If the range between the Interceptor and offense is less than 40m -> apply vision
                        if simu.fast_euclid_distance(off_drone.position, interceptor.position) <= (0.04 ** 2):
                            simu.proportional_navigation(interceptor, off_drone, self.time_step, self.N)
                        else:
                            simu.proportional_navigation(interceptor, last_seen_drone, self.time_step, self.N)

                # Target Hit!
                if simu.fast_euclid_distance(off_drone.position, target) <= (0.01 ** 2): # 10m from the target
                    target_hits += 1
                    inter_reward -= 40
                    del off_drone
                    del last_seen_drone
                    if interceptor_assign:
                        del interceptor
                    else:
                        inter_reward -= 100
                    break
                # Interception!
                if interceptor_assign:
                    if simu.fast_euclid_distance(off_drone.position, interceptor.position) <= (0.01 ** 2): # 10m from the target
                        intercept_hits += 1
                        inter_reward += 50
                        simu.set_score(intercept_list, assigned_inter)
                        intercepted_points.append(off_drone.position)
                        del off_drone
                        del last_seen_drone
                        del interceptor
                        break

        interception_rate = intercept_hits / self.num_drones
        interception_rate = interception_rate * 100
        inter_reward = inter_reward + interception_rate
        return round(inter_reward, 1), round(detect_reward, 1), intercept_list, detect_list, round(interception_rate, 1)

    def test_simulate(self):
        # The level of the simulation is set based on the following:
        #       Number of drones = [20, 100]
        #       Number of targets for the offense drone
        #       Initial positions of the attackers [(0 -> 80, 115)]
        #       A* with or without intelligence
        intercepted_points = []
        target_hits = 0
        intercept_hits = 0
        # Initialize score list for the interceptors with default values (e.g., 0 for each coordinate)
        intercept_list = []
        detect_list = []
        for coord in self.interceptors:
            intercept_list.append(coord)
            intercept_list.append(0)  # Default value
        for coord in self.detectors:
            detect_list.append(coord)
            detect_list.append(0)

        for drone_id in range(self.num_drones):
            start = (random.choice(self.start_pos))  # Start above the defense border
            if "no" in self.intelligence:
                intercepted_points = []

            # Initialize the offensive drone
            off_drone = simu.Drone(position=start, velocity=self.offense_drone_speed, angle=math.radians(0))  # Moving at 270°
            # Create a copy of the offensive drone to monitor last position in case of no visibility
            last_seen_drone = simu.Drone(position=off_drone.position, velocity=off_drone.velocity, angle=off_drone.angle)
            target = (random.choice(self.target_coords))
            if intercepted_points:
                drop_zones = simu.generate_drop_zones(intercepted_points, radius=3, weight=5, stretch_factor=4,
                                                      step_size=0.5)
                path = simu.a_star_drone(start, target, self.x_map_lim, self.y_map_lim, 0.5, drop_zones)
            else:
                path = simu.a_star_drone(start, target, self.x_map_lim, self.y_map_lim, 0.5, intercepted_points)

            if not path:
                print("Path Was not found")
                continue  # Skip if no valid path found

            # Reset Variables
            vision = False
            interceptor_assign = False
            last_index = 0  # Start tracking from first path point
            delay = 0

            # Start the attack
            for step in range(1000000):  # Allow enough iterations for the drone to reach the end
                # start follow the path of attack
                last_index = simu.pure_pursuit_target(off_drone, path, lookahead_dist=2.0, time_step=self.time_step,
                                                 last_index=last_index)

                for detection in self.detectors:
                    if simu.is_in_range(off_drone.position, detection, self.detector_range, shape=2, detector_angle=detection[2]):
                        vision = True
                        simu.set_score(detect_list, detection)
                        last_seen_drone.position = off_drone.position
                        last_seen_drone.angle = off_drone.angle
                        break
                    else:
                        vision = False

                if vision and not interceptor_assign:  # If the offensive drone is in sight of a detector and
                    # an intercepted has not assigned yet
                    interceptor_assign, interceptor, assigned_inter = simu.assign_interceptor(self.interceptors, off_drone,
                                                                                              self.interceptor_speed,
                                                                                              self.interception_predict_steps)

                if vision and interceptor_assign:
                    # if the offensive drone is in sight and an interceptor is assigned
                    # Apply launch delay
                    delay += 1
                    if delay > self.interceptor_delay:
                        # After delay start moving towards the offensive drone
                        simu.proportional_navigation(interceptor, off_drone, self.time_step, self.N)

                if interceptor_assign and not vision:  # If an interceptor is assigned and there is no vision
                    delay += 1
                    if delay > self.interceptor_delay:

                        if simu.fast_euclid_distance(off_drone.position, interceptor.position) <= (0.04 ** 2):
                            simu.proportional_navigation(interceptor, off_drone, self.time_step, self.N)
                        else:
                            simu.proportional_navigation(interceptor, last_seen_drone, self.time_step, self.N)

                # Target Hit!
                if simu.fast_euclid_distance(off_drone.position, target) <= (0.01 ** 2):
                    target_hits += 1
                    del off_drone
                    del last_seen_drone
                    if interceptor_assign:
                        del interceptor
                    break
                # Interception!
                if interceptor_assign:
                    if simu.fast_euclid_distance(off_drone.position, interceptor.position) <= (0.01 ** 2):
                        intercept_hits += 1
                        simu.set_score(intercept_list, assigned_inter)
                        intercepted_points.append(off_drone.position)
                        del off_drone
                        del last_seen_drone
                        del interceptor
                        break

        interception_rate = intercept_hits / self.num_drones
        return interception_rate * 100


    def animate_simulate(self, episode):
        from matplotlib.patches import Wedge
        # The level of the simulation is set based on the following:
        #       Number of drones = [20, 100]
        #       Number of targets for the offense drone
        #       Initial positions of the attackers [(0 -> 80, 115)]
        #       A* with or without intelligence
        intercepted_points = []
        drop_zones = []
        paths = []
        target_hits = 0
        intercept_hits = 0
        # Initialize score list for the interceptors with default values (e.g., 0 for each coordinate)
        intercept_list = []
        detect_list = []
        total_steps = 0
        frame_counter = 0
        os.makedirs(f'animation/episode{episode}', exist_ok=True)  # Create the folder if it doesn't exist

        for coord in self.interceptors:
            intercept_list.append(coord)
            intercept_list.append(0)  # Default value
        for coord in self.detectors:
            detect_list.append(coord)
            detect_list.append(0)

        for drone_id in range(self.num_drones):
            start = (random.choice(self.start_pos))  # Start above the defense border
            if "no" in self.intelligence:
                intercepted_points = []
            # Initialize the offensive drone (Evader)
            off_drone = simu.Drone(position=start, velocity=self.offense_drone_speed,
                                   angle=math.radians(0))  # Moving at 270°
            # Create a copy of the offensive drone to monitor last position in case of no visibility
            last_seen_drone = simu.Drone(position=off_drone.position, velocity=off_drone.velocity, angle=off_drone.angle)
            target = (random.choice(self.target_coords))
            if intercepted_points:
                drop_zones = simu.generate_drop_zones(intercepted_points, radius=3, weight=5, stretch_factor=4,
                                                      step_size=0.5)
                path = simu.a_star_drone(start, target, self.x_map_lim, self.y_map_lim, 0.5, drop_zones)
                paths.append(path)
                path_x, path_y = zip(*path)
            else:
                path = simu.a_star_drone(start, target, self.x_map_lim, self.y_map_lim, 0.5, intercepted_points)
                paths.append(path)
                path_x, path_y = zip(*path)

            if not path:
                print("Path Was not found")
                continue  # Skip if no valid path found

            # Reset Variables
            vision = False
            interceptor_assign = False
            last_index = 0  # Start tracking from first path point
            delay = 0

            # Start the attack
            for step in range(1000000):  # Allow enough iterations for the drone to reach the end
                # start follow the path of attack
                last_index = simu.pure_pursuit_target(off_drone, path, lookahead_dist=2.0, time_step=self.time_step,
                                                      last_index=last_index)
                # plot every 200 steps
                if step % 200 == 0:
                    frame_counter += 1
                    plt.figure(figsize=(8, 6))
                    if drop_zones:
                        for point in drop_zones:
                            plt.plot(point[0], point[1], 'r.') # plot intercepted points
                    plt.plot(target[0], target[1], 'rX') # plot the attacker target
                    plt.plot(start[0], start[1], 'k1') # plot start point of the attacker
                    plt.plot(off_drone.position[0], off_drone.position[1], 'rx', label='Attacker', markersize=10) # plot the attacker

                    if drone_id == self.num_drones - 1:
                        for path in paths:
                            path_x, path_y = zip(*path)
                            plt.plot(path_x, path_y, '--k')
                    else:
                        plt.plot(path_x, path_y, '--k')
                    plt.plot([0, 80], [70, 70], '-r')  # plot the border
                    for interceptor_id in self.interceptors:
                        plt.plot(interceptor_id[0], interceptor_id[1], 'ko')
                    if interceptor_assign:
                        plt.plot(interceptor.position[0], interceptor.position[1], 'bp', label='Interceptor', markersize=10) # plot the interceptor
                        plt.plot(assigned_inter[0], assigned_inter[1], 'bo') # plot the interceptor start point

                    FOV = 60
                    for det in self.detectors:
                        # Checks if this specific detector is the one seeing the drone to color it red
                        is_active = vision and simu.is_in_range(off_drone.position, det, self.detector_range,
                                                                shape=2, detector_angle=det[2])
                        c = 'red' if is_active else 'blue'

                        # Create Wedge: (center_x, center_y), radius, start_angle, end_angle
                        w = Wedge((det[0], det[1]), self.detector_range, det[2] - FOV / 2, det[2] + FOV / 2,
                                  color=c, alpha=0.2)
                        plt.gca().add_patch(w)

                    plt.xlabel('x-axis')
                    plt.ylabel('y-axis')
                    plt.xlim([-5, 85])
                    plt.ylim([-5, 120])
                    plt.title(f'Drone Number {drone_id}         Intercepted = {intercept_hits}   Missed = {target_hits}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'animation/episode{episode}/frame{frame_counter}.png', dpi=300)
                    plt.close()

                for detection in self.detectors:
                    if simu.is_in_range(off_drone.position, detection, self.detector_range, shape=2,
                                        detector_angle=detection[2]):
                        vision = True
                        simu.set_score(detect_list, detection)
                        last_seen_drone.position = off_drone.position
                        last_seen_drone.angle = off_drone.angle
                        break
                    else:
                        vision = False

                if vision and not interceptor_assign:  # If the offensive drone is in sight of a detector and
                    # an intercepted has not assigned yet
                    interceptor_assign, interceptor, assigned_inter = simu.assign_interceptor(self.interceptors, off_drone,
                                                                                              self.interceptor_speed,
                                                                                              self.interception_predict_steps)

                if vision and interceptor_assign:
                    # if the offensive drone is in sight and an interceptor is assigned
                    # Apply launch delay
                    delay += 1
                    if delay > self.interceptor_delay:
                        # After delay start moving towards the offensive drone
                        simu.proportional_navigation(interceptor, off_drone, self.time_step, self.N)

                if interceptor_assign and not vision:  # If an interceptor is assigned and there is no vision
                    delay += 1
                    if delay > self.interceptor_delay:

                        if simu.fast_euclid_distance(off_drone.position, interceptor.position) <= (0.04 ** 2):
                            simu.proportional_navigation(interceptor, off_drone, self.time_step, self.N)
                        else:
                            simu.proportional_navigation(interceptor, last_seen_drone, self.time_step, self.N)

                # Target Hit!
                if simu.fast_euclid_distance(off_drone.position, target) <= (0.01 ** 2):
                    target_hits += 1
                    print(f'drone num {drone_id}, IR = {intercept_hits * 100 / (drone_id + 1)} steps = {step: ,}')
                    total_steps += step

                    # plot frame if target hit ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    frame_counter += 1
                    plt.figure(figsize=(8, 6))
                    if drop_zones:
                        for point in drop_zones:
                            plt.plot(point[0], point[1], 'r.')  # plot intercepted points
                    plt.plot(target[0], target[1], 'rX')  # plot the attacker target
                    plt.plot(start[0], start[1], 'k1')  # plot start point of the attacker
                    plt.plot([0, 80], [70, 70], '-r')  # plot the border
                    plt.plot(off_drone.position[0], off_drone.position[1], 'rx', label='Attacker',
                             markersize=10)  # plot the attacker
                    for interceptor_id in self.interceptors:
                        plt.plot(interceptor_id[0], interceptor_id[1], 'ko')
                    if interceptor_assign:
                        plt.plot(interceptor.position[0], interceptor.position[1], 'bp', label='Interceptor',
                                 markersize=10)  # plot the interceptor
                        plt.plot(assigned_inter[0], assigned_inter[1], 'bo')  # plot the interceptor start point
                    plt.xlabel('x-axis')
                    plt.ylabel('y-axis')
                    plt.xlim([-5, 85])
                    plt.ylim([-5, 120])
                    plt.title(
                        f'Drone Number {drone_id}         Intercepted = {intercept_hits}   Missed = {target_hits}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'animation/episode{episode}/frame{frame_counter}.png', dpi=300)
                    plt.close()

                    del off_drone
                    del last_seen_drone
                    if interceptor_assign:
                        del interceptor
                    break
                # Interception!
                if interceptor_assign:
                    if simu.fast_euclid_distance(off_drone.position, interceptor.position) <= (0.01 ** 2):
                        intercept_hits += 1
                        # plot frame if intercepted ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        frame_counter += 1
                        plt.figure(figsize=(8, 6))
                        if drop_zones:
                            for point in drop_zones:
                                plt.plot(point[0], point[1], 'r.')  # plot intercepted points
                        plt.plot(target[0], target[1], 'rX')  # plot the attacker target
                        plt.plot(start[0], start[1], 'k1')  # plot start point of the attacker
                        plt.plot([0, 80], [70, 70], '-r')  # plot the border
                        plt.plot(off_drone.position[0], off_drone.position[1], 'rx', label='Attacker',
                                 markersize=10)  # plot the attacker
                        for interceptor_id in self.interceptors:
                            plt.plot(interceptor_id[0], interceptor_id[1], 'ko')
                        if interceptor_assign:
                            plt.plot(interceptor.position[0], interceptor.position[1], 'bp', label='Interceptor',
                                     markersize=10)  # plot the interceptor
                            plt.plot(assigned_inter[0], assigned_inter[1], 'bo')  # plot the interceptor start point
                        plt.xlabel('x-axis')
                        plt.ylabel('y-axis')
                        plt.xlim([-5, 85])
                        plt.ylim([-5, 120])
                        plt.title(
                            f'Drone Number {drone_id}         Intercepted = {intercept_hits}   Missed = {target_hits}')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f'animation/episode{episode}/frame{frame_counter}.png', dpi=300)
                        plt.close()

                        simu.set_score(intercept_list, assigned_inter)
                        intercepted_points.append(off_drone.position)
                        print(f'drone num {drone_id}, IR = {intercept_hits * 100 / (drone_id + 1)} steps = {step: ,}')
                        total_steps += step
                        del off_drone
                        del last_seen_drone
                        del interceptor
                        break

        interception_rate = intercept_hits / self.num_drones
        return interception_rate * 100, total_steps