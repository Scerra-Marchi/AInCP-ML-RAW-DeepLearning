# Time Series Analysis for the AInCP Project

This repository contains the Time Series Analysis component for the AInCP project. It is a fork incorporating advancements from the bachelor thesis work by Davide Marchi and Giordano Scerra.

## Executable Python Files

The primary executable Python scripts are located in the `Pipeline/` folder:

*   **`main_whole_assessment.py`**: This script executes the complete 10-fold cross-validation pipeline.
    *   Executes a fixed 10-fold cross-validation using the entire dataset.
    *   In each fold, 90% of the data is used for training and the remaining 10% is used for assessment.
    *   Outputs are written under `Pipeline/Iterations/` or `Pipeline/Iterations_debug/`.
    *   Each iteration stores its split in `iteration_data.json` and, when completed, writes `best_estimators_results.csv`, `combined_test_stats.json`, and `Week_stats/predictions_dataframe.csv`.

## `main_whole_assessment.py` Parameters

*   **`--iterations N`**:
    *   Runs the first `N` folds from the fixed 10-fold split.
    *   Useful to compute only missing folds and resume later.

*   **`--reset-iterations`**:
    *   Deletes the selected output folder (`Iterations/` or `Iterations_debug/`) before starting.

*   **`--debug`**:
    *   Runs a much smaller sanity-check version of the pipeline.
    *   Uses `Iterations_debug/` instead of `Iterations/`.
    *   Restricts the methods and the number of samples per fold.

*   **`--purge-pred-caches`**:
    *   Deletes all classifier prediction caches (`pred_cache/`) inside the selected iterations folder before starting.

## Environment Reproducibility

*   **`requirements.txt`**:
    *   Generated with `pip freeze` from the active environment.
    *   Captures the exact pip package state of that environment.

*   **`environment.yml`**:
    *   Generated with `conda env export`.
    *   Captures the full conda environment.

*   To recreate the conda environment:
    *   `conda env create -f environment.yml`

*   To recreate the pip environment:
    *   `pip install -r requirements.txt`

## Important Note

Before running the pipeline, ensure that the `DATA_FOLDER` variable in `Pipeline/main_whole_assessment.py` points to the correct location of your dataset.
