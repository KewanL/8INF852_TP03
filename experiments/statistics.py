import csv
import os
import numpy as np

CSV_DIR = "results/csv"

def save_stats(name, best, median, std, feasible_rate):
    os.makedirs(CSV_DIR, exist_ok=True)
    file_path = f"{CSV_DIR}/results.csv"

    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Algorithm", "Best", "Median", "Std", "FeasibleRate"])

        writer.writerow([name, best, median, std, feasible_rate])


def save_history(name, histories):
    os.makedirs(CSV_DIR, exist_ok=True)

    median = np.median(histories, axis=0)
    q1 = np.percentile(histories, 25, axis=0)
    q3 = np.percentile(histories, 75, axis=0)

    file_path = f"{CSV_DIR}/history_{name.lower().replace(' ', '_')}.csv"

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "median", "q1", "q3"])

        for i in range(len(median)):
            writer.writerow([i, median[i], q1[i], q3[i]])