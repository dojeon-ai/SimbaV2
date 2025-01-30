import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import wandb


def collect_runs(entity: str, project_name: str):
    api = wandb.Api()

    name = entity + "/" + project_name
    runs = api.runs(name)

    return runs


def filter_runs(runs, exp_names: List, group_name: Optional[str] = None):
    """
    runs: list of wandb runs
    exp_names: list
    group_name: str
    """
    filtered_runs = []
    for run in tqdm(runs):
        configs = {k: v for k, v in run.config.items() if not k.startswith("_")}

        if len(configs) == 0:
            continue

        run_group_name = configs["group_name"]
        run_exp_name = configs["exp_name"]

        if group_name:
            if run_group_name != group_name:
                continue
        if run_exp_name in exp_names:
            filtered_runs.append(run)

    return filtered_runs


def convert_runs_to_dataframe(
    runs, run_exp_name_to_analysis_exp_name: Optional[Dict] = None
):
    """
    Convert wandb runs to a pandas DataFrame.

    Parameters:
    runs: List of wandb runs
    run_exp_name_to_analysis_exp_name: Optional[Dict]
        Mapping from experiment name recorded in wandb to the name to be used in analysis.
        {exp_name: analysis_name}. If None, the recorded experiment name will be used.

    Returns:
    pandas DataFrame with columns:
        - run: wandb run
        - exp_name: analysis name for the experiafrment
        - summary: summary of the run metrics
        - config: hyperparameters of the run
    """
    analysis_exp_names, summaries, configs = [], [], []
    for run in tqdm(runs):
        # Extract and store the summary metrics of the run
        summaries.append(run.summary._json_dict)

        # Extract and store the configuration, excluding special values that start with '_'
        config = {
            key: value for key, value in run.config.items() if not key.startswith("_")
        }
        configs.append(config)

        # Map the recorded experiment name to the analysis experiment name if mapping is provided
        run_exp_name = config["exp_name"]
        if run_exp_name_to_analysis_exp_name:
            analysis_exp_name = run_exp_name_to_analysis_exp_name.get(
                run_exp_name, run_exp_name
            )
        else:
            analysis_exp_name = run_exp_name
        analysis_exp_names.append(analysis_exp_name)

    runs_df = pd.DataFrame(
        {
            "run": runs,
            "exp_name": analysis_exp_names,
            "summary": summaries,
            "config": configs,
        }
    )

    return runs_df


def filter_by_seeds(runs_df, step_key, seeds: List):
    # Filter out rows where the step_key does not match the specified step
    filtered_df = runs_df[
        runs_df.apply(lambda row: row["config"].get(step_key) in seeds, axis=1)
    ]

    return filtered_df


def convert_wandb_df_to_eval_df(runs_df, metrics: List, n_samples=100000):
    """
    Collect the history of specified metrics from wandb runs.

    Parameters:
    runs_df: pandas DataFrame
        DataFrame containing the wandb runs.
    metrics: list
        List of metric names to collect history for.
    n_samples: int, optional
        Number of samples to fetch from the run history. Default is 100000.

    Returns:
    eval_df: pandas DataFrame with following columns
        exp_name / env_name / seed / metric / env_step / value
    """
    records = []

    for idx in tqdm(range(len(runs_df))):
        run_data = runs_df.iloc[idx]
        exp_name = run_data["exp_name"]
        config = run_data["config"]
        if "env_name" in config:
            env_name = config["env_name"]
        elif "env" in config:
            env_name = config["env"]["env_name"]
        else:
            raise ValueError
        seed = config["seed"]

        run = run_data["run"]
        history = run.history(samples=n_samples)
        steps = history["_step"]

        for metric in metrics:
            if metric in history.columns:
                metric_history = history[metric].dropna()
                for idx, val in metric_history.items():
                    env_step = steps[idx]

                    records.append(
                        {
                            "exp_name": exp_name,
                            "env_name": env_name,
                            "seed": seed,
                            "metric": metric,
                            "env_step": env_step,
                            "value": val,
                        }
                    )

    return pd.DataFrame(records)


def read_and_concatenate_eval_df(directory_path):
    """
    Reads all CSV files in the specified directory and concatenates them into a single DataFrame.

    Args:
        directory_path (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all CSV files' content.
    """
    # Initialize an empty DataFrame for concatenation
    concatenated_df = pd.DataFrame()

    # List all files in the directory
    for file_name in os.listdir(directory_path):
        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is a CSV file
        if file_name.endswith(".csv"):
            # Read the file
            try:
                file_df = pd.read_csv(file_path, index_col=0)

                # Append to the concatenated DataFrame
                concatenated_df = pd.concat(
                    [concatenated_df, file_df], ignore_index=True
                )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    concatenated_df = concatenated_df.reset_index(drop=True)

    return concatenated_df


def save_eval_df(eval_df, directory_path):
    """
    Saves individual subsets of eval_df into separate CSV files named after unique values in the 'exp_name' column.

    Args:
        eval_df (pd.DataFrame): DataFrame containing the data to save.
        directory_path (str): Path to the directory where the files will be saved.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for exp_name in eval_df["exp_name"].unique():
        _eval_df = eval_df[eval_df["exp_name"] == exp_name].reset_index(drop=True)
        file_path = os.path.join(directory_path, f"{exp_name}.csv")
        _eval_df.to_csv(file_path)


def plot_metric_history_per_env(
    eval_df,
    metric: str = "avg_return",
    plot_width: int = 10,
    plot_height: int = 4,
    num_plots_per_row: int = 4,
    num_x_ticks: int = 5,
    x_lim_min: int = 0,
    x_lim_max: int = 2e6,
    y_lim_min: int = 0,
    y_lim_max: int = 1000,
    x_label: str = "env_step (M)",
    y_label: str = "avg_return",
    legend_bbox_to_anchor: tuple = (0.5, 1.15),
):
    experiments = eval_df["exp_name"].unique()
    environments = eval_df["env_name"].unique()

    # Plotting
    num_plots = len(environments)
    cols = min(num_plots_per_row, num_plots)
    rows = (num_plots + cols - 1) // cols  # Calculate number of rows needed

    fig, axs = plt.subplots(
        rows, cols, figsize=(plot_width, plot_height), sharex="col", sharey="row"
    )

    # Flatten axs into a 1D array for easier indexing
    axs = axs.flatten()
    eval_df = eval_df[eval_df["metric"] == metric]

    for i, env in enumerate(environments):
        ax = axs[i]
        env_data = eval_df[eval_df["env_name"] == env]

        for j, exp in enumerate(experiments):
            exp_data = env_data[env_data["exp_name"] == exp]
            if len(exp_data) == 0:
                continue
            grouped_data = exp_data.groupby("env_step")["value"]

            env_steps = grouped_data.mean().index.values
            mean = grouped_data.mean().values
            std_dev = grouped_data.std().values

            # Plot mean history
            ax.plot(env_steps, mean, label=exp)

            # Fill between mean - std_dev and mean + std_dev
            ax.fill_between(
                env_steps,
                mean - std_dev,
                mean + std_dev,
                alpha=0.2,
                label="_nolegend_",  # Exclude from legend
            )

        # Set subplot title and labels
        ax.set_title(env)
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lim_min, y_lim_max)

        # Set x-ticks and multiply by eval_freq
        ticks = np.arange(0, 1.001, 1 / num_x_ticks) * x_lim_max
        ax.set_xticks(ticks)
        ax.set_xticklabels(["{:,.1f}".format(tick / 1e6) for tick in ticks])
        ax.set_xlabel(x_label)

    # Add a shared legend
    fig.legend(
        experiments,
        loc="upper center",
        ncol=num_plots_per_row,
        bbox_to_anchor=legend_bbox_to_anchor,
    )

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def generate_metric_matrix_dict(eval_df, env_step: int, metric_type: str):
    """
    Generates a dictionary of metric matrices from eval_df based on the specified env_step.

    Args:
    - eval_df (pandas DataFrame): DataFrame containing the evaluation data.
    - env_step (int): The specific environment step to extract metrics for.
    - metric_type (str): The metric to collect (e.g., 'avg_return').

    Returns:
    - metric_matrix_dict (dict):
        Keys: experiment names (exp_name).
        Values: 2D numpy arrays of shape (num_envs, num_seeds) with the metrics.
                Each row is an environment; each column is a seed.
    """
    # Filter data for the specified env_step and metric_type
    filtered_df = eval_df[
        (eval_df["env_step"] == env_step) & (eval_df["metric"] == metric_type)
    ]

    # Initialize dictionary to hold the metric matrices
    metric_matrix_dict = {}

    # Get the unique experiment names
    exp_names = filtered_df["exp_name"].unique()

    for exp_name in exp_names:
        # Filter data for the current experiment
        exp_data = filtered_df[filtered_df["exp_name"] == exp_name]

        # Pivot table to have env_name as rows, each seed as a column
        pivot_table = exp_data.pivot_table(
            values="value", index="env_name", columns="seed", aggfunc="first"
        )

        # Convert the pivot table to a numpy array
        metric_matrix = pivot_table.to_numpy()

        # Optionally remove any columns (seeds) that contain NaNs across all envs
        # (If you prefer to keep partial data, you can remove this block,
        #  or adjust it to your preference.)
        nan_cols = np.any(np.isnan(metric_matrix), axis=0)
        metric_matrix = metric_matrix[:, ~nan_cols]

        # Add the metric matrix to the dictionary
        metric_matrix_dict[exp_name] = metric_matrix

    return metric_matrix_dict


def normalize_score_with_random_and_base_score(
    df: pd.DataFrame, random_score_dict: dict, base_score_dict: dict
) -> pd.DataFrame:
    """
    Normalize the 'value' column based on random and base scores.
        normalized_value = (value - random_score) / (base_score - random_score)

    Args:
    - df (pd.DataFrame): DataFrame with 'env_name' and 'value' columns.
    - random_score_dict (dict): Mapping of 'env_name' to random scores.
    - base_score_dict (dict): Mapping of 'env_name' to base scores.

    Returns:
    - pd.DataFrame: DataFrame with an added 'normalized_value' column.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_normalized = df.copy()

    # Define a function to normalize a single value
    def normalize_value(row):
        env_name = row["env_name"]
        value = row["value"]
        base_score = base_score_dict[env_name]
        random_score = random_score_dict[env_name]

        return (value - random_score) / (base_score - random_score)

    # Apply the normalization function to each row
    df_normalized["value"] = df_normalized.apply(normalize_value, axis=1)

    return df_normalized


def normalize_values(df, TASK_SUCCESS_SCORE):
    """
    Normalize the 'value' column in the DataFrame based on the TASK_SUCCESS_SCORE.

    Args:
    - df (pandas.DataFrame): Input DataFrame with columns 'env_name' and 'value'

    Returns:
    - pandas.DataFrame: DataFrame with normalized 'value' column
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_normalized = df.copy()

    # Define a function to normalize a single value
    def normalize_value(row):
        env_name = row["env_name"]
        value = row["value"]
        if env_name in TASK_SUCCESS_SCORE:
            return value / TASK_SUCCESS_SCORE[env_name] * 1000
        else:
            print(
                f"Warning: No normalization score found for environment '{env_name}'. Returning original value."
            )
            raise NotImplementedError

    # Apply the normalization function to each row
    df_normalized["value"] = df_normalized.apply(normalize_value, axis=1)

    return df_normalized