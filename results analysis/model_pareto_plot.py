import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Try importing adjustText for superior label placement
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("Warning: 'adjustText' library not found. Label placement might overlap.")
    print("To fix, run: pip install adjustText")

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

# Ensure high-res output settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.autolayout'] = True

# --- Data Loading ---
model = 306
n_configs = 10
n_levels = 6
plot_data = []
data_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(data_dir, '..', f'train/model_{model}_results')
data_file = os.path.abspath(data_file)

print(f"Reading data from: {data_file}")

for config_id in range(n_configs):
    file_path = os.path.join(data_file, f"performance_config{config_id}.json")
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r") as f:
        data = json.load(f)

    config_key = f'config{config_id}'
    result = data[config_key]

    adjusted_vals = []
    for level in range(1, n_levels + 1):
        key = f'Level_{level}'
        level_res = result[key]
        num_inter = level_res['num_inter']
        num_detect = level_res['num_detect']
        resource_sum = num_inter + num_detect
        adjusted_vals.append(level_res['mean'] - level_res['std'])

    y_score = sum(adjusted_vals) / len(adjusted_vals)
    # Shorter label for cleaner plot
    label = f"Configuration {config_id}({num_inter},{num_detect})"
    plot_data.append([config_id, resource_sum, y_score, label])

df = pd.DataFrame(plot_data, columns=['Config ID', 'Resources', 'Score', 'Label'])

# --- Pareto Frontier Logic ---
df_sorted = df.sort_values(by=['Resources', 'Score'], ascending=[True, False])
pareto_points = []
current_max_score = -float('inf')

for _, row in df_sorted.iterrows():
    if row['Score'] > current_max_score:
        pareto_points.append(row)
        current_max_score = row['Score']
pareto_df = pd.DataFrame(pareto_points)


fig, ax = plt.subplots(figsize=(7, 5))

# Plot Sub-optimal points
ax.scatter(df['Resources'], df['Score'],
           color='red',  # Medium gray
           s=80,             # Smaller marker size
           alpha=0.8,
           edgecolor='none',
           label='Sub-optimal Configurations')

# Plot Pareto Frontier
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
ax.set_xlabel('Resources (|I| + |D|)', fontweight='bold', labelpad=12)
ax.set_ylabel('Average Interception Rate (IR)', fontweight='bold', labelpad=12)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(True, which='major', axis='both', linestyle=':', color='gray', linewidth=0.5, alpha=0.5)

# Ensure integer ticks on X-axis (resources number)
ax.xaxis.get_major_locator().set_params(integer=True)

ax.legend(frameon=False, loc='lower right')

# Annotations with Overlap Management
texts = []
for _, row in pareto_df.iterrows():
    # Create text object but don't place it finally yet
    t = ax.text(row['Resources'], row['Score'], row['Label'], fontsize=14)
    texts.append(t)

if ADJUST_TEXT_AVAILABLE and len(texts) > 0:
    # adjust_text repositions labels to avoid overlaps with points and lines
    adjust_text(texts,
                x=df['Resources'],
                y=df['Score'],
                ax=ax,
                force_points=0.2,
                force_text=0.5,
                expand_points=(1.2, 1.2),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
else:
    # Fallback simple offset if library is missing
    for t in texts:
        t.set_position((t.get_position()[0], t.get_position()[1] + (ax.get_ylim()[1]*0.02)))
        t.set_ha('center')

# Save outputs
# Save as PNG for quick viewing
png_path = os.path.join(data_dir, f"pareto_frontier_model_{model}.png")
# Save as PDF for high quality
pdf_path = os.path.join(data_dir, f"pareto_frontier_model_{model}.pdf")

plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.savefig(pdf_path, bbox_inches='tight')

print(f"Plots saved to: {png_path} and {pdf_path}")
plt.show()
