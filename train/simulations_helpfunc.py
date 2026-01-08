import math
from queue import PriorityQueue
import numpy as np


class Drone:
    """Represents a drone with a position, velocity, and heading angle."""

    def __init__(self, position, velocity, angle):
        self.position = position  # (x, y) coordinates
        self.velocity = velocity  # Speed (constant)
        self.angle = angle  # Heading angle in radians
        self.prev_theta_LOS = None

    def move(self, time_step=0.5):
        """Moves the drone in the direction of its heading."""
        x, y = self.position
        x += self.velocity * math.cos(self.angle) * time_step # Update x-coordinate
        y += self.velocity * math.sin(self.angle) * time_step # Update y-coordinate
        self.position = (x, y)  # Update position

# Function to update a coordinate's value
def set_score(coordinate_list, coord):
    try:
        index = coordinate_list.index(coord) + 1  # Find coordinate and get value index
        coordinate_list[index] += 1
    except ValueError:
        print(f"Coordinate {coord} not found!")

def find_lookahead_point(position, path, lookahead_dist, last_index):
    """
    Finds a valid lookahead point ahead of the drone.
    The point is the first one that is at least `lookahead_dist` away
    and is ahead of the current closest index.
    """
    for i in range(last_index, len(path)):  # Start from last_index to avoid backtracking
        if euclid_distance(position, path[i]) >= lookahead_dist:
            return path[i], i  # Return new target point and update last index
    return path[-1], len(path) - 1  # If no point found, return last waypoint

def pure_pursuit_target(drone, path, lookahead_dist=2.0, time_step=0.5, last_index=0):
    """
    Pure Pursuit Controller for the target drone.
    Adjusts the drone's heading to move toward a lookahead point.
    Keeps track of the closest visited index to ensure forward progress.
    """
    lookahead_point, new_index = find_lookahead_point(drone.position, path, lookahead_dist, last_index)

    # Compute new heading angle toward lookahead point
    new_angle = math.atan2(lookahead_point[1] - drone.position[1], lookahead_point[0] - drone.position[0])

    # Update drone's heading and move forward
    drone.angle = new_angle
    drone.move(time_step)

    return new_index  # Return updated index to track progress

def euclid_distance(point1, point2):
    distance = math.sqrt(pow((point2[1] - point1[1]), 2) + pow((point2[0] - point1[0]), 2))
    return distance

def fast_euclid_distance(point1, point2):
    # Return the distance without the sqrt() !!
    distance = pow((point2[1] - point1[1]), 2) + pow((point2[0] - point1[0]), 2)
    return distance


def proportional_navigation(interceptor, target, time_step, N=2.5):
    """
    Updates interceptor's heading angle using Proportional Navigation.

    Parameters:
    - interceptor: The drone trying to intercept the target.
    - target: The moving target drone.
    - N: The navigation constant (typically between 2-5, higher values = more aggressive turns).
    """
    # Extract positions of the interceptor and target
    x_I, y_I = interceptor.position
    x_T, y_T = target.position

    # Compute the current Line-of-Sight (LOS) angle between the interceptor and target
    theta_LOS = math.atan2(y_T - y_I, x_T - x_I)  # Angle from interceptor to target

    # Compute the rate of change of the LOS angle (approximate derivative)
    if interceptor.prev_theta_LOS is None:
        interceptor.prev_theta_LOS = theta_LOS  # Initialize the first time

    theta_LOS_rate = theta_LOS - interceptor.prev_theta_LOS  # Change in LOS angle
    interceptor.prev_theta_LOS = theta_LOS  # Update for next step

    # Adjust the interceptor’s heading angle based on Proportional Navigation formula
    interceptor.angle += N * theta_LOS_rate / time_step  # Proportional correction

    # Move the interceptor in its new direction
    interceptor.move(time_step)

def assign_interceptor(interceptors, offensive_drone, interceptor_speed, prediction_steps=5):
    """
    Assign interceptor based on coordinate prediction in a few steps ahead.
    Args:
        interceptors: A list of interceptors coordinates.
        offensive_drone: The moving target drone.
        interceptor_speed: The speed of the interceptor
        prediction_steps: How many steps to predict to assign an interceptor

    Returns:

    """
    predicted_x = offensive_drone.position[0] + offensive_drone.velocity * math.cos(offensive_drone.angle) * prediction_steps
    predicted_y = offensive_drone.position[1] + offensive_drone.velocity * math.sin(offensive_drone.angle) * prediction_steps
    predicted_pos = (predicted_x, predicted_y)

    best_score = 0
    best_interceptor = None
    for inter in interceptors:  # Iterate through available interceptors
        dist = euclid_distance(inter, predicted_pos)
        delta_x = predicted_pos[0] - inter[0]
        delta_y = predicted_pos[1] - inter[1]
        angle_to_target = math.degrees(math.atan2(delta_y, delta_x))

        delta_angle = abs(angle_to_target - (math.degrees(offensive_drone.angle) + 180))

        score = 10 * (1 / (delta_angle + 1e-6))
        if score > best_score:
            best_score = score
            best_interceptor = inter

    if best_interceptor:
        delta_x = predicted_pos[0] - best_interceptor[0]
        delta_y = predicted_pos[1] - best_interceptor[1]
        angle_to_target = math.atan2(delta_y, delta_x)

        interceptor = Drone(position=best_interceptor, velocity=interceptor_speed, angle=angle_to_target)
        return True, interceptor, best_interceptor
    return False, None, None

# Heuristic function (Euclidean distance for 2D grid)
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def generate_drop_zones(centers, radius=3, weight=5, stretch_factor=2, step_size=0.5):
    drop_zone = {}

    for center in centers:
        cx, cy = center
        # Rounding the number to match the step size
        cx = round(cx / step_size) * step_size
        cy = round(cy / step_size) * step_size

        # Iterate using step_size instead of integers
        dx_range = np.arange(-radius, radius + step_size, step_size)
        dy_range = np.arange(0, radius * stretch_factor + step_size, step_size)

        for dx in dx_range:
            for dy in dy_range:
                distance = np.sqrt(dx ** 2 + (dy / stretch_factor) ** 2)  # Stretch vertically
                if distance <= radius:
                    penalty = weight * (1 - (distance / radius))  # Decay effect
                    key = (round(cx + dx, 2), round(cy + dy, 2))  # Ensure precision

                    # Merge penalties if multiple drop zones overlap
                    drop_zone[key] = max(drop_zone.get(key, 0), penalty)

    return drop_zone

# A* Pathfinding with adaptive weight modification for intercepted points
def a_star_drone(start, end, x_lim, y_lim, step_size, intercepted_points):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end:
            return reconstruct_path(came_from, end)

        # Possible moves: up, down, left, right
        directions = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                      (-step_size, -step_size), (-step_size, step_size), (step_size, -step_size), (step_size, step_size)]

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Ensure neighbor is within bounds (not outside the map)
            if x_lim[0] <= neighbor[0] <= x_lim[1] and y_lim[0] <= neighbor[1] <= y_lim[1]:
                temp_g_score = g_score[current] + 1

                # If the neighbor was an intercepted point or in an avoidance zone, increase its cost
                if neighbor in intercepted_points:
                    temp_g_score += intercepted_points[neighbor]  # Increase cost dynamically

                if temp_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, end)

                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

    return None  # No path found

# Function to reconstruct the path from the final node to the start
def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

def is_in_range(point1, point2, det_range, shape=1, detector_angle=0, sweep_angle=60):
    """
    Checks if point1 is within the detection range of point2, considering the shape of the detector.

    Parameters:
        point1 (tuple): Coordinates of the point to check (x, y).
        point2 (tuple): Coordinates and angle of the detector (x, y, angle).
        range (float): Detection range.
        shape (int): 1 for circular, 2 for sector-based.
        detector_angle (float): Orientation angle of the detector (in degrees).
        sweep_angle (float): Sweep angle of the sector (in degrees).

    Returns:
        int: 1 if within range, 0 otherwise.
    """
    if shape == 1:
        # Circular detection range
        distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        return 1 if distance < det_range ** 2 else 0
    elif shape == 2:
        # Sector-based detection range
        distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        if distance > det_range ** 2:
            return False  # Out of range

        # Compute angle between detector and point
        delta_x = point1[0] - point2[0]
        delta_y = point1[1] - point2[1]
        point_angle = math.degrees(math.atan2(delta_y, delta_x))  # Angle in degrees

        # Normalize angles to [0, 360)
        detector_angle = detector_angle % 360
        point_angle = point_angle % 360

        # Calculate angle difference
        angle_diff = (point_angle - detector_angle + 180) % 360 - 180

        # Check if within sweep angle
        if -sweep_angle / 2 <= angle_diff <= sweep_angle / 2:
            return True  # Within detection sector

        return False  # Outside detection sector
    else:
        return -1  # Indicate an error
