import pandas as pd
import matplotlib.pyplot as plt

# Extracting all relevant data for each stage and num_threads
df = pd.DataFrame(
    [
        (67, "jetson", "CifarSparse", "OMP", 1, "little", 1, 0.184),
        (68, "jetson", "CifarSparse", "OMP", 1, "little", 2, 0.098),
        (69, "jetson", "CifarSparse", "OMP", 1, "little", 3, 0.074),
        (70, "jetson", "CifarSparse", "OMP", 1, "little", 4, 0.054),
        (71, "jetson", "CifarSparse", "OMP", 1, "little", 5, 0.046),
        (72, "jetson", "CifarSparse", "OMP", 1, "little", 6, 0.041),
        (73, "jetson", "CifarSparse", "OMP", 2, "little", 1, 0.102),
        (74, "jetson", "CifarSparse", "OMP", 2, "little", 2, 0.055),
        (75, "jetson", "CifarSparse", "OMP", 2, "little", 3, 0.039),
        (76, "jetson", "CifarSparse", "OMP", 2, "little", 4, 0.032),
        (77, "jetson", "CifarSparse", "OMP", 2, "little", 5, 0.029),
        (78, "jetson", "CifarSparse", "OMP", 2, "little", 6, 0.028),
        (79, "jetson", "CifarSparse", "OMP", 3, "little", 1, 0.146),
        (80, "jetson", "CifarSparse", "OMP", 3, "little", 2, 0.078),
        (81, "jetson", "CifarSparse", "OMP", 3, "little", 3, 0.065),
        (82, "jetson", "CifarSparse", "OMP", 3, "little", 4, 0.05),
        (83, "jetson", "CifarSparse", "OMP", 3, "little", 5, 0.04),
        (84, "jetson", "CifarSparse", "OMP", 3, "little", 6, 0.037),
        (85, "jetson", "CifarSparse", "OMP", 4, "little", 1, 0.077),
        (86, "jetson", "CifarSparse", "OMP", 4, "little", 2, 0.041),
        (87, "jetson", "CifarSparse", "OMP", 4, "little", 3, 0.032),
        (88, "jetson", "CifarSparse", "OMP", 4, "little", 4, 0.026),
        (89, "jetson", "CifarSparse", "OMP", 4, "little", 5, 0.025),
        (90, "jetson", "CifarSparse", "OMP", 4, "little", 6, 0.034),
        (91, "jetson", "CifarSparse", "OMP", 5, "little", 1, 0.06),
        (92, "jetson", "CifarSparse", "OMP", 5, "little", 2, 0.033),
        (93, "jetson", "CifarSparse", "OMP", 5, "little", 3, 0.026),
        (94, "jetson", "CifarSparse", "OMP", 5, "little", 4, 0.022),
        (95, "jetson", "CifarSparse", "OMP", 5, "little", 5, 0.02),
        (96, "jetson", "CifarSparse", "OMP", 5, "little", 6, 0.019),
        (97, "jetson", "CifarSparse", "OMP", 6, "little", 1, 0.058),
        (98, "jetson", "CifarSparse", "OMP", 6, "little", 2, 0.033),
        (99, "jetson", "CifarSparse", "OMP", 6, "little", 3, 0.025),
        (100, "jetson", "CifarSparse", "OMP", 6, "little", 4, 0.022),
        (101, "jetson", "CifarSparse", "OMP", 6, "little", 5, 0.02),
        (102, "jetson", "CifarSparse", "OMP", 6, "little", 6, 0.019),
        (103, "jetson", "CifarSparse", "OMP", 7, "little", 1, 0.042),
        (104, "jetson", "CifarSparse", "OMP", 7, "little", 2, 0.024),
        (105, "jetson", "CifarSparse", "OMP", 7, "little", 3, 0.019),
        (106, "jetson", "CifarSparse", "OMP", 7, "little", 4, 0.017),
        (107, "jetson", "CifarSparse", "OMP", 7, "little", 5, 0.016),
        (108, "jetson", "CifarSparse", "OMP", 7, "little", 6, 0.016),
        (109, "jetson", "CifarSparse", "OMP", 8, "little", 1, 0.027),
        (110, "jetson", "CifarSparse", "OMP", 8, "little", 2, 0.017),
        (111, "jetson", "CifarSparse", "OMP", 8, "little", 3, 0.014),
        (112, "jetson", "CifarSparse", "OMP", 8, "little", 4, 0.013),
        (113, "jetson", "CifarSparse", "OMP", 8, "little", 5, 0.013),
        (114, "jetson", "CifarSparse", "OMP", 8, "little", 6, 0.014),
        (115, "jetson", "CifarSparse", "OMP", 9, "little", 1, 0.002),
        (116, "jetson", "CifarSparse", "OMP", 9, "little", 2, 0.004),
        (117, "jetson", "CifarSparse", "OMP", 9, "little", 3, 0.006),
        (118, "jetson", "CifarSparse", "OMP", 9, "little", 4, 0.007),
        (119, "jetson", "CifarSparse", "OMP", 9, "little", 5, 0.009),
        (120, "jetson", "CifarSparse", "OMP", 9, "little", 6, 0.01),
    ],
    columns=[
        "id",
        "machine_name",
        "application",
        "backend",
        "stage",
        "core_type",
        "num_threads",
        "time_ms",
    ],
)

# Pivot the data for grouped bar plot
pivot_df = df.pivot(index="num_threads", columns="stage", values="time_ms")

# Plot the figure
plt.figure(figsize=(15, 8))
pivot_df.plot(kind="bar", width=0.8, figsize=(15, 8), alpha=0.7)

plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (s)")
plt.title(
    "Execution Time per Stage with Different Thread Counts (Jetson, CifarSparse, OpenMP)"
)
plt.xticks(rotation=0)
plt.legend(title="Stage", loc="upper right", bbox_to_anchor=(1.15, 1))
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Display plot
plt.show()

# Save the plot to a file
plt.savefig("execution_time_per_stage.png", dpi=300)
