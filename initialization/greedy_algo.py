import numpy as np
import pandas as pd
import random
from data_prep import def_points_sorted
from train import help_func as hf


def assign_table(table, values, radius, shape=1, sweep_angle = 60):
    """
    Fills the table with 1s and 0s by checking whether each generated point is within range of each defense point.

    Parameters:
    - table (pd.DataFrame): DataFrame with defense points (interceptors or detectors) as index.
    - values (list of tuples): List of generated points.
    - radius (float): Detection radius.
    - shape (int): 1 for circular detection, 2 for sector-based detection.

    Returns:
    - score (np.array): Score for each generated point (how many defense points cover it).
    - table (pd.DataFrame): Updated table with detection values.
    """
    temp = np.zeros((len(table.index),), dtype=int)
    score = np.zeros(len(values), dtype=int)  # Array of scores

    for pos, gen_point in enumerate(values):
        if shape == 2:
            x, y, detector_angle = gen_point  # Unpack (x, y, angle)
        else:
            x, y = gen_point
            detector_angle = 0  # No angle needed for circular detection

        for index, def_point in enumerate(table.index):
            if shape == 2:
                temp[index] = hf.is_in_range(def_point, (x, y, detector_angle), radius, shape, detector_angle, sweep_angle)
            else:
                temp[index] = hf.is_in_range(def_point, (x, y), radius, shape, detector_angle, sweep_angle)

        if shape == 2:
            col_name = f"({int(gen_point[0])}, {int(gen_point[1])}, {int(detector_angle)})"
        else:
            col_name = f"({int(gen_point[0])}, {int(gen_point[1])})"
        table[col_name] = temp
        score[pos] = np.sum(temp)

    return score, table


def recursive_solver(
    table,
    shape=1,
    depth=0,
    max_lead_solutions=50,
    current_solution=None,
    min_score_threshold=5,
    max_low_score_solutions=2,
    random_seed=None
):
    """
    Solves the set cover problem recursively to find complete solutions, prioritizing higher scores and adding randomness.

    Parameters:
        table (pd.DataFrame): The DataFrame representing the problem.
        depth (int): Depth of recursion, for debugging purposes.
        max_lead_solutions (int): Maximum number of solutions to find.
        current_solution (list): Current path of selected columns (coordinates).
        min_score_threshold (int): Minimum score to prioritize without limiting solutions.
        max_low_score_solutions (int): Maximum number of low-score solutions to explore.
        random_seed (int): Random seed for reproducibility (optional).

    Returns:
        solutions (list): List of complete solutions (sequences of coordinates).
    """
    if current_solution is None:
        current_solution = []

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Base case: If the table is empty, return the current solution as a valid solution
    if table.empty:
        print(f"Solution found at depth {depth}: {current_solution}")
        return [current_solution]

    # Step 1: Calculate scores for each column
    column_scores = table.sum(axis=0)

    # Step 2: Find columns with the highest scores
    max_score = column_scores.max()
    if pd.isna(max_score):  # Handle NaN case if the table is invalid
        print(f"Stopping recursion at depth {depth}: No valid scores in table.")
        return []

    # Identify top columns and their coordinates
    top_columns = column_scores[column_scores == max_score].index.tolist()
    lead_coordinates = [tuple(map(int, col.strip("()").split(", "))) for col in top_columns]

    # Limit solutions for lower scores and add randomness
    if max_score < min_score_threshold and len(lead_coordinates) > max_low_score_solutions:
        lead_coordinates = random.sample(lead_coordinates, max_low_score_solutions)

    print(f"Depth {depth}: Best score {max_score} with {len(lead_coordinates)} lead solutions.")
    print(f"Lead coordinates: {lead_coordinates}")

    # List to store all complete solutions
    complete_solutions = []

    # Step 3: Recurse for each lead solution
    for lead_coord in lead_coordinates:
        # Duplicate the table
        table_copy = table.copy()

        # Get the column name corresponding to the lead coordinate
        col_name = str(lead_coord)

        if col_name in table_copy.columns:
            # Drop the column and associated rows
            rows_to_drop = table_copy.index[table_copy[col_name] == 1].tolist()
            table_copy.drop(columns=[col_name], inplace=True)
            table_copy.drop(index=rows_to_drop, inplace=True)

        print(f"Depth {depth}: Generated table after removing {lead_coord}. Remaining size: {table_copy.shape}")

        # Recur on the modified table
        solutions_from_here = recursive_solver(
            table_copy,
            shape,
            depth + 1,
            max_lead_solutions,
            current_solution + [lead_coord],
            min_score_threshold,
            max_low_score_solutions,
            random_seed
        )

        # Add these solutions to the complete solutions list
        complete_solutions.extend(solutions_from_here)

        # Stop early if we have enough solutions
        if len(complete_solutions) >= max_lead_solutions:
            break

    # Return all complete solutions
    return complete_solutions


# Define the range for x and y
x_min, x_max = 5, 75  # Range for x
y_min, y_max = 0, 70  # Range for y

# Define the grid resolution (step size)
x_step, y_step = 2, 2  # Step size for x and y

# Generate the grid points
x = np.arange(x_min, x_max + x_step, x_step)
y = np.arange(y_min, y_max + y_step, y_step)
x_grid, y_grid = np.meshgrid(x, y)

# Combine grid points into a single array of shape (N, 2), where N is the number of points
grid_points = [(round(xi, 2), round(yi, 2)) for xi in x for yi in y]

# Ensure unique points only
sorted_points_formatted = list(set(grid_points))

# Sort the points for consistent formatting
sorted_points_formatted = sorted(sorted_points_formatted, key=lambda p: (p[0], p[1]))

# Create Dataframe for the detection
detectors_radius = 40
angles = [20, 45, 60, 90, 120, 135, 150]  # List of possible angles

# Create a new list with (x, y, angle) for each point
grid_points_sorted_with_angles = [(x, y, angle) for (x, y) in sorted_points_formatted for angle in angles]

# Write the interceptors and detectors placements to excel
excel_writer = pd.ExcelWriter("placements.xlsx", engine="xlsxwriter")

# Update `original_df_detection` to use this new list
original_df_detection = pd.DataFrame(index=def_points_sorted)

# Create an empty DataFrame
original_df_interception = pd.DataFrame(index=def_points_sorted)

score_inter, original_df_interception  = assign_table(original_df_interception, sorted_points_formatted,
                                                      radius=20, shape=1)
score_detec, original_df_detection  = assign_table(original_df_detection, grid_points_sorted_with_angles,
                                                   radius=detectors_radius, shape=2)

inter_solutions = recursive_solver(
    original_df_interception,
    depth=0,
    max_lead_solutions=150,
    min_score_threshold=5,
    max_low_score_solutions=2
)

detect_solutions = recursive_solver(
    original_df_detection,
    shape=2,
    depth=0,
    max_lead_solutions=300,
    min_score_threshold=10,
    max_low_score_solutions=1
)
placements_df = pd.DataFrame(columns=["Interceptors", "Detectors", "Number of Interceptors", "Number of Detectors"])
de_min_sol = detect_solutions[0]
for sol in detect_solutions:
    if len(sol) < len(de_min_sol):
        de_min_sol = sol
print(f'Best solution is: {de_min_sol}, \nwith {len(de_min_sol)} Detectors')

in_min_sol = inter_solutions[0]
for sol in inter_solutions:
    if len(sol) < len(in_min_sol):
        in_min_sol = sol
print(f'Best solution is: {in_min_sol}, \nwith {len(in_min_sol)} interceptors')

# History assign
placements_df.loc[len(placements_df)] = [
    in_min_sol,
    de_min_sol,
    len(in_min_sol),
    len(de_min_sol)
]
# Save each run as a separate sheet
placements_df.to_excel(excel_writer, sheet_name="Placements", index=False)
excel_writer.close()
