import simulations
import numpy as np
import json
import os


# Load the saved best configurations
def load_best_configs(filename="best_configs1.json"):
    with open(filename, "r") as f:
        best_configs = json.load(f)
    return best_configs

def accumulate(main_list, list):
    for idx, cord in enumerate(list):
        if idx % 2 == 0:
            continue
        else:
            main_list[idx] += cord
    return main_list

def find_expandable(score_list):
    """
    This function takes lists of instances with scores.
    Args:
        score_list:
        The score list shape like this: [(x, y), score] or [(x, y, angle), score]
    Returns:
    The instances to delete and a new list without the deleted instances
    """
    instances_to_delete = []
    new_list = []
    for index, coord in enumerate(score_list):
        if index % 2 == 0:
            continue
        else:
            if coord == 0:
                instances_to_delete.append(score_list[index - 1])
            else:
                new_list.append(score_list[index - 1])
    return instances_to_delete, new_list

# [(10, 10), (30, 19), (55, 25), (60, 49), (70, 40), (10, 50), (25, 40), (40, 42)]
selected_targets = [(3, 30), (10, 10), (25, 25), (50, 22), (55, 40), (30, 40), (60, 20), (65, 42)]

# Initialize interceptors and detectors
intercept_radius = 20
detector_range = 40
border_y = 70
interceptor_delay = 250  # Time steps for interceptor to reach height (250 * 0.2 sec each step = 50 sec)
time_step = 0.2  # 0.05

model_num = 504
num_test = 30

# Load the trained configurations
best_configs = load_best_configs(f"best_configs{4}.json")

# for a specific configuration to evaluate
# num_config = 7



intercept_rec_log = []
detect_rec_log = []
runs_record = []
interception_rate_rec_log = []

for config in range(0, 10): # range(0, 10) # if specific configuration use -> config = num_config
    inter_state = best_configs[config]['interceptors']
    detectors_state = best_configs[config]['detectors']

    # Switch to manual deployment (optional)
    # inter_state = []
    # detectors_state = []

    # Initiate simulation
    simulation = simulations.DroneSimulation(time_step, inter_state, detectors_state, detector_range, selected_targets, interceptor_delay, border_y)
    for level in range(1, 7): # range(1, 7)
        simulation.level = level
        simulation.init_level2()
        simulation.interceptor_speed = 0.03

        for _ in range(num_test):
            inter_reward, detect_reward, intercept_list, detect_list, dump = simulation.simulate()
            if _ == 0:
                intercept_rec_log = intercept_list
                detect_rec_log = detect_list
            else:
                intercept_rec_log = accumulate(intercept_rec_log, intercept_list)
                detect_rec_log = accumulate(detect_rec_log, detect_list)
            runs_record.append(_)
            print(f"Level: {level} -- Test Number {_} -- Interceptors reward: {inter_reward} -- Configuration: {config}")
    deleted_interceptors, new_inter_state = find_expandable(intercept_rec_log)
    deleted_detectors, new_detect_state = find_expandable(detect_rec_log)
    print(f'Deleted = {deleted_interceptors} Interceptors ----- New state = {new_inter_state}')
    print(f'Deleted = {deleted_detectors} Detectors ----- New state = {new_detect_state}')

    simulation.interceptors = new_inter_state
    simulation.detectors = new_detect_state

    performance_data = {f"config{config}": {}}

    for level in range(1, 7): # range(1, 7)
        simulation.level = level
        simulation.init_level2()
        simulation.interceptor_speed = 0.03
        interception_rate_rec_log = []
        for _ in range(num_test):
            interception_rate = simulation.test_simulate()
            interception_rate_rec_log.append(interception_rate)
            runs_record.append(_)
            print(f"Level: {level} -- Test Number {_} -- Interceptors rate: {interception_rate} -- Configuration: {config}")

        mean = np.mean(interception_rate_rec_log)
        std = np.std(interception_rate_rec_log)
        print(f"Level: {level} -- Interception Rate Mean: {mean} -- Interceptors Rate Std: {std} -- Configuration: {config}")
        # Update performance_data
        performance_data[f"config{config}"][f"Level_{level}"] = {
            "mean": mean,
            "std": std,
            "num_inter": len(new_inter_state),
            "num_detect": len(new_detect_state),
            "inter_state": new_inter_state,
            "detect_state": new_detect_state
        }

    # Choose your save directory
    save_dir = f"model_{model_num}_results"
    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Create a filename
    filename = f"performance_config{config}.json"

    # Full path
    file_path = os.path.join(save_dir, filename)

    # Save performance_data
    with open(file_path, "w") as f:
        json.dump(performance_data, f, indent=4)

    print(f"Performance data saved to {file_path}")
