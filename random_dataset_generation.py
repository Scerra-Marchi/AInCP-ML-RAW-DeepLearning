import os
import csv
from datetime import datetime, timedelta
import random

# ---------------- CONFIG ----------------
BASE_DIR = "Dataset"
AHA_DIR = os.path.join(BASE_DIR, "AHA")
WEEK_DIR = os.path.join(BASE_DIR, "WEEK")

SAMPLING_INTERVAL_MS = 12.5  # 80 Hz
START_DATETIME = datetime(2017, 5, 3, 16, 31, 0)

AHA_FILES = 5
AHA_DURATION_MIN = 20

WEEK_FILES = 5
WEEK_DURATION_DAYS = 1

HEADER = ["datetime", "x_D", "y_D", "z_D", "x_ND", "y_ND", "z_ND"]

# Valori medi simili al tuo esempio
BASE_VALUES = {
    "x_D": -0.78,
    "y_D": -0.09,
    "z_D": -0.61,
    "x_ND": 0.83,
    "y_ND": -0.06,
    "z_ND": 0.53,
}

NOISE = 0.01
# ---------------------------------------


def generate_row(current_time):
    return [
        current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
        round(BASE_VALUES["x_D"] + random.uniform(-NOISE, NOISE), 3),
        round(BASE_VALUES["y_D"] + random.uniform(-NOISE, NOISE), 3),
        round(BASE_VALUES["z_D"] + random.uniform(-NOISE, NOISE), 3),
        round(BASE_VALUES["x_ND"] + random.uniform(-NOISE, NOISE), 3),
        round(BASE_VALUES["y_ND"] + random.uniform(-NOISE, NOISE), 3),
        round(BASE_VALUES["z_ND"] + random.uniform(-NOISE, NOISE), 3),
    ]


def generate_file(filepath, duration_seconds):
    num_samples = int(duration_seconds / (SAMPLING_INTERVAL_MS / 1000))
    current_time = START_DATETIME

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for _ in range(num_samples):
            writer.writerow(generate_row(current_time))
            current_time += timedelta(milliseconds=SAMPLING_INTERVAL_MS)


def main():
    os.makedirs(AHA_DIR, exist_ok=True)
    os.makedirs(WEEK_DIR, exist_ok=True)

    # AHA files (20 minuti)
    aha_duration_sec = AHA_DURATION_MIN * 60
    for i in range(1, AHA_FILES + 1):
        filename = f"{i}_AHA_RAW.csv"
        generate_file(os.path.join(AHA_DIR, filename), aha_duration_sec)

    # WEEK files (1 giorno)
    week_duration_sec = WEEK_DURATION_DAYS * 24 * 60 * 60
    for i in range(1, WEEK_FILES + 1):
        filename = f"{i}_week_RAW.csv"
        generate_file(os.path.join(WEEK_DIR, filename), week_duration_sec)

    print("Dataset generato con successo!")


if __name__ == "__main__":
    main()
