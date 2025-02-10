# Created by Y.Yudin, J.McCullough, D.Groen during SEAVEA hackathon 10.02.2024

import os
import sys
import glob
from errno import EEXIST

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def combine_data(run_base_folder: str = "../sample_flee_output", input_file_name: str = "out.csv") -> pd.DataFrame, int:
    """
    Summary

    For a given run base folder parse all the numbered subfolders for each individual run
    parse the output files into a single Pandas DataFrame
    Rows from each run are now added a new index column is added, containing information on the run ID and time stamp
    For each QoI clumns are added for: standard deviation
    """

    run_folders = [d for d in glob.glob(os.path.join(run_base_folder, "[0-9]*")) if os.path.isdir(d)]

    num_runs = len(run_folders)

    data_combined = pd.DataFrame()

    for folder in run_folders:

        run_number = os.path.basename(folder)
        out_files = glob.glob(os.path.join(folder, input_file_name))

        for file in out_files:

            data = pd.read_csv(file)

            day_number = data.index
            data['index'] = [f"{day}_{run_number}" for day in day_number]
            data_combined = pd.concat([data_combined, data], ignore_index=True)

    # Reset the index to the new 'index' column
    data_combined = data_combined.set_index('index')

    return data_combined, num_runs

def data_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summary

    Add columns to the data frame for the daily mean and standard deviation across different runs of the QoIs for every camp
    """

    data_filtered = data.drop(
        [
            "Total error",
            "refugees in camps (UNHCR)",
            "total refugees (simulation)",
            "raw UNHCR refugee count",
            "refugees in camps (simulation)",
            "refugee_debt",
        ],
        axis=1,
    )

    cols = list(data_filtered.columns.values)
    sim_columns = [col for col in data_filtered.columns if col.endswith('sim')]

    # Calculate the daily mean and standard deviation for columns ending with 'sim'
    mean_per_day = data_filtered.groupby('Day')[sim_columns].mean().add_suffix('_daily_mean')
    std_per_day = data_filtered.groupby('Day')[sim_columns].std().add_suffix('_daily_std')

    # Merge the mean and std values back into the original DataFrame
    data_filtered = data_filtered.merge(mean_per_day, on='Day', how='left')
    data_filtered = data_filtered.merge(std_per_day, on='Day', how='left')

    # Restore the original index
    data_filtered.index = data.index
    #print(data_filtered)
        
    return data_filtered

def plot_camps_uq(data: pd.DataFrame, config, output:str) -> None:
    """
    Summary
    """

    # Plotting function for camps

    # Plot a number of lines representing time evolution of a camp
    # Bold line - mean
    # Thin lines - individual simulation in an ensemble
    # Shaded are - +/- standard deviation

    # data_filtered = data.drop(
    #     [
    #         "Total error",
    #         "refugees in camps (UNHCR)",
    #         "total refugees (simulation)",
    #         "raw UNHCR refugee count",
    #         "refugees in camps (simulation)",
    #         "refugee_debt",
    #     ],
    #     axis=1,
    # )

    cols = list(data.columns.values)

    output = os.path.join(output, "camps")

    mkdir_p(output)

    alpha=0.1

    if "n_sim" in config:
        n_sim = int(config["n_sim"])
    else
        n_sim = 10

    for i in range(len(cols)):

        name = cols[i].split()

        if name[0]=="Date" or name[0]=="Day": # Date, Day is not a camp field.
            continue

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 8)

        plt.xlabel("Days elapsed", fontsize=14)
        plt.ylabel("Number of asylum seekers / unrecognised refugees", fontsize=14)
        plt.title("{}".format(name[0]), fontsize=18)

        # Plotting individual runs
        for run_number in range(n_sim):
            run_data = data[data.index.str.endswith(f"_{run_number}")]
            y1 = run_data["%s sim" % name[0]]
            plt.plot(run_data['Day'], y1, "k-", alpha=alpha)

        # Filter the data to include only the first reading for each run
        data_filtered = data.drop_duplicates(subset='Day')

        y1 = data_filtered["%s sim_daily_mean" % name[0]]
        y2 = data_filtered["%s data" % name[0]]

        (label1,) = plt.plot(data_filtered['Day'], y1, "r", linewidth=5, label="{} simulation +/- STD across {} runs".format(name[0], n_sim))
        
        (label2,) = plt.plot(data_filtered['Day'], y2, "b", linewidth=5, label="{} UNHCR data".format(name[0]))

        # plotting uncertainty
        plt.fill_between(data_filtered['Day'], y1 - data_filtered["%s sim_daily_std" % name[0]], y1 + data_filtered["%s sim_daily_std" % name[0]], color="red", alpha=0.5)

        # formatting
        plt.legend(handles=[label1, label2], loc=0, prop={"size": 14})

        plt.savefig("{}/{}.png".format(output, name[0]), bbox_inches = 'tight')

        plt.clf()

def mkdir_p(mypath: str) -> None:
    """
    Creates a directory. equivalent to using mkdir -p on the command line

    Args:
        mypath (TYPE): Description
    """
    try:
        os.makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else:
            raise