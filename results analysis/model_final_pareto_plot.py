import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

# Define the specific (Model, Configuration) pairs you want to compare
# Format: (model_id, config_id)
SELECTED_CONFIGS = [
    (303, 1),
    (303, 7),
    (304, 0),
    (304, 9),
    (305, 6),
    (305, 7),
    (306, 1),
    (306, 9),
    (302, 8),
    (302, 9),
    (301, 0),
    (301, 9),
    (108, 0),
    (110, 0)
]

# Constants
N_LEVELS = 6

# Try importing adjustText for superior label placement
try:
    from adjustText import adjust_text

    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("Warning: 'adjustText' library not found. Label placement might overlap.")
    print("To fix, run: pip install adjustText")

# --- STYLE SETUP (MATCHING ORIGINAL) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
# Ensure high-res output settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.autolayout'] = True

# --- DATA LOADING ---
plot_data = []
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes the script is inside a folder at the same level as 'train'
base_data_dir = os.path.join(current_dir, '..', 'train')

print(f"Searching for data in: {os.path.abspath(base_data_dir)}")

for model_id, config_id in SELECTED_CONFIGS:
    # Construct path dynamically based on model_id
    model_folder = f'model_{model_id}_results'
    file_name = f"performance_config{config_id}.json"
    file_path = os.path.join(base_data_dir, model_folder, file_name)

    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"Warning: File not found for M{model_id} C{config_id}: {file_path}")
        continue

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        config_key = f'config{config_id}'

        if config_key not in data:
            print(f"Warning: Key {config_key} not found in {file_path}")
            continue

        result = data[config_key]

        # Calculate Score (Mean - Std per level, averaged)
        adjusted_vals = []
        final_resources = 0

        for level in range(1, N_LEVELS + 1):
            key = f'Level_{level}'
            if key not in result:
                continue
            level_res = result[key]

            num_inter = level_res.get('num_inter', 0)
            num_detect = level_res.get('num_detect', 0)
            final_resources = num_inter + num_detect

            adjusted_vals.append(level_res['mean'] - level_res['std'])

        if not adjusted_vals:
            continue

        y_score = sum(adjusted_vals) / len(adjusted_vals)
        label = f"Model {model_id}"

        plot_data.append({
            'Model': model_id,
            'Config': config_id,
            'Resources': final_resources,
            'Score': y_score,
            'Label': label
        })

    except Exception as e:
        print(f"Error processing M{model_id} C{config_id}: {e}")

if not plot_data:
    print("No data loaded. Check paths and configuration IDs.")
    exit()

df = pd.DataFrame(plot_data)

# --- PARETO FRONTIER LOGIC ---
df_sorted = df.sort_values(by=['Resources', 'Score'], ascending=[True, False])
pareto_points = []
current_max_score = -float('inf')

for _, row in df_sorted.iterrows():
    if row['Score'] > current_max_score:
        pareto_points.append(row)
        current_max_score = row['Score']
pareto_df = pd.DataFrame(pareto_points)

# --- ACADEMIC PLOTTING (STRICTLY ORIGINAL STYLE) ---
fig, ax = plt.subplots(figsize=(7, 5))

# Plot Sub-optimal points
ax.scatter(df['Resources'], df['Score'],
           color='red',  # Red color as requested
           s=80,
           alpha=0.8,
           edgecolor='none',
           label='Sub-optimal Configurations')

# Plot Pareto Frontier
if not pareto_df.empty:
    ax.plot(pareto_df['Resources'], pareto_df['Score'],
            color='black',
            linewidth=1.5,
            linestyle='-',
            marker='D',
            markersize=10,
            markerfacecolor='white',
            markeredgewidth=1.5,
            label='Pareto Frontier')

# Clean Axis Formatting
ax.set_xlabel('Resources (|D| + |I|)', fontweight='bold', labelpad=12)
ax.set_ylabel('Average Interception Rate (IR)', fontweight='bold', labelpad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(True, which='major', axis='both', linestyle=':', color='gray', linewidth=0.5, alpha=0.5)

# Ensure integer ticks on X-axis
ax.xaxis.get_major_locator().set_params(integer=True)

ax.legend(frameon=False, loc='lower right')

# Annotations with Overlap Management
texts = []
if not pareto_df.empty:
    for _, row in pareto_df.iterrows():
        t = ax.text(row['Resources'], row['Score'], row['Label'], fontsize=14)
        texts.append(t)

if ADJUST_TEXT_AVAILABLE and len(texts) > 0:
    # adjust_text repositions labels to avoid overlaps with points and lines
    adjust_text(texts,
                x=df['Resources'],
                y=df['Score'],
                ax=ax,
                force_points=0.05,
                force_text=0.1,
                expand_points=(1.05, 1.05),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
else:
    # Fallback simple offset if library is missing
    for t in texts:
        t.set_position((t.get_position()[0], t.get_position()[1] + (ax.get_ylim()[1] * 0.005)))
        t.set_ha('center')

# Save outputs
output_filename_png = "pareto_frontier_comparison_original_style.png"
output_filename_pdf = "pareto_frontier_comparison_original_style.pdf"
png_path = os.path.join(current_dir, output_filename_png)
pdf_path = os.path.join(current_dir, output_filename_pdf)

plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.savefig(pdf_path, bbox_inches='tight')

print(f"Plots saved to: {png_path} and {pdf_path}")
plt.show()
