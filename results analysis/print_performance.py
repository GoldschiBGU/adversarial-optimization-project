import os
import json
import pandas as pd

# Model directory (e.g., "model_1_results")
model = 504
n_configs = 10
n_levels = 6

# Data holder
matrix_data = []

# Get the directory where this script is located
data_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one directory to reach Drone_x and join with the filename
data_file = os.path.join(data_dir, '..', f'train/model_{model}_results')
data_file = os.path.abspath(data_file)


# Iterate through configurations
for config_id in range(n_configs):
    # config_id = 0
    # Load the config file
    file_path = os.path.join(data_file, f"performance_config{config_id}.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    row = []
    for level in range(1, n_levels + 1):
        key = f'Level_{level}'
        result = data[f'config{config_id}'][key]
        mean = result['mean']
        std = result['std']

        # Format cell: if 100 with 0 std, just "100", else "mean ± std"
        if mean == 100.0 and std == 0.0:
            cell = "100"
        else:
            cell = f"{mean:.1f} ± {std:.1f}"
        row.append(cell)

    # Add config info to start of row
    num_inter = result['num_inter']
    num_detect = result['num_detect']
    config_label = f"{config_id} ({num_inter}, {num_detect})"
    matrix_data.append([config_label] + row)

# Create DataFrame
columns = ["Level Config"] + [str(lvl) for lvl in range(1, n_levels + 1)]
df = pd.DataFrame(matrix_data, columns=columns)

# Save to CSV
df.to_csv("summary_matrix.csv", index=False)

# Show the matrix
print(df.to_string(index=False))


# Function to extract mean and std from cell
def parse_mean_std(cell):
    if isinstance(cell, str):
        if '±' in cell:
            mean_str, std_str = cell.split('±')
            mean = float(mean_str.strip())
            std = float(std_str.strip())
        else:
            mean = float(cell.strip())
            std = 0.0
        return mean, std
    return None, None

# Initialize lists for adjusted scores
adjusted_scores = []

# Iterate through rows
for _, row in df.iterrows():
    adjusted_vals = []
    for col in df.columns[1:7]:  # columns "1" to "5"
        mean, std = parse_mean_std(row[col])
        adjusted_vals.append(mean - std)
    avg_adjusted = sum(adjusted_vals) / len(adjusted_vals)
    adjusted_scores.append(avg_adjusted)

# Add to DataFrame
df['Adjusted Mean - Std'] = adjusted_scores

# Sort and show best configurations
best_df = df.sort_values(by='Adjusted Mean - Std', ascending=False)

# Display top results
print("Top Configurations (mean - std):")
print(best_df[['Level Config', 'Adjusted Mean - Std']].head(10))
