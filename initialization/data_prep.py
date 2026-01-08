import numpy as np
import pandas as pd


def plot_border(x0, x1, y0, y1, density):
    X = np.linspace(x0, x1, density)
    Y = np.linspace(y0, y1, density)
    border_points = [(float(X[i]), float(Y[i])) for i in range(len(Y))]
    return border_points

# Load town data from Excel file
excel_file = 'town_data.xlsx'
town_df = pd.read_excel(excel_file)
towns = town_df.to_dict(orient='records')


defense_border = plot_border(0, 80, 70, 70, 10)

# Plot Towns
town_centers = []
town_edges = []
for town in towns:
    # Determine the number of edges by counting available edge_x columns
    num_edges = sum(1 for key in town if 'edge_' in key and '_x' in key)
    town_centers.append((town['center_x'], town['center_y']))

    # Collect all edge points
    for i in range(num_edges):
        town_edges.append((town[f'edge_{i}_x'], town[f'edge_{i}_y']))

# Clean NaN from town_edges
town_edges = [(x, y) for x, y in town_edges if not (np.isnan(x) or np.isnan(y))]


# --- Output Data ---
defense_points = []  # Store defense points (town edges and centers)
print("\nTown Centers:", town_centers)
defense_points.extend(town_centers)  # Save town center as a defense points
print("\nTown Edges:", town_edges) # List of (x, y) tuples
defense_points.extend(town_edges)  # Save town edges as a defense points

defense_points.extend(defense_border) # Save border as defence points

# Sort all points by x-coordinate, then by y-coordinate
def_points_sorted = sorted(defense_points, key=lambda p: (p[0], p[1]))
