import math
import numpy as np
import matplotlib.pyplot as plt
import heapq
import json
import torch
import os


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

def plot_border(x0, x1, y0, y1, density, color_n_type, label='Border'):
    plt.plot([x0, x1], [y0, y1], color_n_type, label=label)
    X = np.linspace(x0, x1, density)
    Y = np.linspace(y0, y1, density)
    border_points = [(float(X[i]), float(Y[i])) for i in range(len(Y))]
    return border_points

def get_town_edges(towns):
    town_edges = []
    for town in towns:
        # Determine the number of edges by counting available edge_x columns
        num_edges = sum(1 for key in town if 'edge_' in key and '_x' in key)

        # Collect all edge points
        for i in range(num_edges):
            town_edges.append((town[f'edge_{i}_x'], town[f'edge_{i}_y']))

    # Clean NaN from town_edges
    town_edges = [(x, y) for x, y in town_edges if not (np.isnan(x) or np.isnan(y))]
    return town_edges

def save_best_config(best_configs, interceptors, detectors, inter_reward, avg_reward, level, episode):

    combined_score = inter_reward + avg_reward

    # Create a config dictionary
    config = {
        "episode": episode,
        "level": level,
        "interceptors": interceptors,
        "detectors": detectors,
        "inter_reward": inter_reward,
        "avg_reward": avg_reward,
        "combined_score": combined_score
    }

    # Use a min-heap to maintain top 10 configurations
    if len(best_configs) < 10:
        heapq.heappush(best_configs, (combined_score, episode, config))
    else:
        heapq.heappushpop(best_configs, (combined_score, episode, config))

    return best_configs

def save_to_file(best_configs, level):
    filename = f"best_configs{level}.json"
    with open(filename, "w") as f:
        json.dump([config for _, _, config in sorted(best_configs, reverse=True)], f, indent=4)


def compute_density_penalty(inter_state, min_dist, penalty_factor=10):
    """
    Computes a penalty based on the density of interceptors.

    Args:
        inter_state (list of tuples): List of (x, y) positions of interceptors.
        min_dist (float): Minimum allowed distance between interceptors.
        penalty_factor (float): Scaling factor for the penalty.
        distance_func (function): Custom function to compute Euclidean distance.

    Returns:
        float: Penalty value.
    """
    num_interceptors = len(inter_state)
    penalty = 0.0

    # Compute pairwise distances
    for i in range(num_interceptors):
        for j in range(i + 1, num_interceptors):
            dist = (inter_state[i][0] - inter_state[j][0]) ** 2 + (inter_state[i][1] - inter_state[j][1]) ** 2
            if dist < min_dist ** 2:  # instead of using sqrt()
                penalty += penalty_factor * (min_dist - math.sqrt(dist))

    return round(penalty, 1)

def get_arc_edge_point(detector, radius):
    """
    Returns the (x, y) coordinate at the center of the arc along the center direction.

    Args:
        detector: [(x, y, angle)]
        radius (float): Distance from the center to the edge of the arc

    Returns:
        (float, float): Coordinates of the center point of detection along the detector's center line
    """
    x, y, angle_deg = detector
    radius = radius / 2  # We want to use the center of the vector to avoid covering the same areas
    angle_rad = math.radians(angle_deg)
    edge_x = round(x + radius * math.cos(angle_rad), 2)
    edge_y = round(y + radius * math.sin(angle_rad), 2)
    return edge_x, edge_y

def get_detectors_pointers(detectors, radius):
    """

    Args:
        detectors: [[(x, y, angle)],...] list of detectors
        radius: The detectors radius

    Returns:
        pointers to the center of detection
    """
    pointers = []
    for detector in detectors:
        edge_point = get_arc_edge_point(detector, radius)
        pointers.append(edge_point)
    return pointers

def compute_detect_density(detectors, radius, density):
    pointers = get_detectors_pointers(detectors, radius)
    penalty = compute_density_penalty(pointers, density)
    return round(penalty, 1)


def save_models(inter_actor, inter_critic, detect_actor, detect_critic, level):
    folder = "saved_models"
    os.makedirs(folder, exist_ok=True)

    torch.save(inter_actor.state_dict(), f"{folder}/inter_actor_level_{level}.pth")
    torch.save(inter_critic.state_dict(), f"{folder}/inter_critic_level_{level}.pth")
    torch.save(detect_actor.state_dict(), f"{folder}/detect_actor_level_{level}.pth")
    torch.save(detect_critic.state_dict(), f"{folder}/detect_critic_level_{level}.pth")
    print(f"✅ Models saved for level {level}")
