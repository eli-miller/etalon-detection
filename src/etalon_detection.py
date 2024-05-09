import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from tqdm import tqdm
import argparse

# TODO: Refactor this to be a module
import sys

from pldspectrapy import beam_from_log_files

sys.path.append(
    "/Users/elimiller/Documents/1Research/python_project/NNA_data_analysis/nna_tools.py"
)
from nna_tools import create_dcs_dataframe


plt.style.use("eli_default")
plt.switch_backend("macosx")


def extract_noise_floor(abs_residual, percent_of_signal=0.1, method="std"):
    """
        Extract the noise floor from the *Absolute Value*  of a residual signal.
    Finds the middle of the residual signal where we expect no etalons or signal.
    Returns 3x the standard deviation of the middle portion of the residual signal.


    Parameters:
    ----------
    abs_residual : array
        Residual signal.

    percent_of_signal : float
        Percent of the signal to use for extracting the noise floor.

    method : str
        Method for extracting the noise floor. Must be 'std' or 'max'.
        'std' returns 3x the standard deviation of the middle portion of the residual signal.

    Returns
    -------
    noise_floor : float
        Noise floor of the residual signal.
    """

    # Find the middle of the residual signal
    middle = int(len(abs_residual) / 2)
    idx_percent_of_signal = int(len(abs_residual) * percent_of_signal / 2)

    noise_region = abs_residual[
        middle - idx_percent_of_signal : middle + idx_percent_of_signal
    ]

    if method == "std":
        noise_floor = 3 * np.std(noise_region)
    elif method == "max":
        noise_floor = np.max(noise_region)
    else:
        raise ValueError("method must be 'std' or 'max'")

    return noise_floor


def expand_edges(edges, width_multiplier=1.0):
    """
    Grow the edges of a peak by a given width multiplier.

    Parameters:
    ----------
    edges : tuple
        Tuple of left and right edges of a peak. Must be ints.
    width_multiplier : float
        Width multiplier for symmetrically adjusting peak widths.

    Returns
    -------
    new_edges : tuple
        Tuple of left and right edges of the grown peak.
    """

    if not all(isinstance(edge, int) for edge in edges):
        print("WARNING: Edges are not ints. Results may be unexpected.")
    # ensure that inputs are ints
    edges = tuple(map(int, edges))

    left, right = edges
    width = right - left

    new_left = left - width * (width_multiplier - 1.0) / 2
    new_right = right + width * (width_multiplier - 1.0) / 2

    # Convert to ints
    new_left = int(np.floor(new_left))
    new_right = int(np.ceil(new_right))

    new_edges = (new_left, new_right)

    return new_edges


def merge_edges(list_of_edges, distance_threshold):
    if distance_threshold == 0:
        # If the distance_threshold is 0, find the union of overlapping edges
        return merge_overlapping_edges(list_of_edges)

    if not list_of_edges:
        print("Warning: No edges found.")
        return []

    # Merge edges within the specified distance_threshold
    merged_edges = []
    list_of_edges.sort(key=lambda edge: edge[0])  # Sort edges by their start index

    current_edge = list_of_edges[0]
    for i in range(1, len(list_of_edges)):
        if list_of_edges[i][0] - current_edge[1] <= distance_threshold:
            # Edges overlap, merge them
            current_edge = (
                current_edge[0],
                max(list_of_edges[i][1], current_edge[1]),
            )
        else:
            merged_edges.append(current_edge)
            current_edge = list_of_edges[i]

    merged_edges.append(current_edge)  # Add the last merged edge
    return merged_edges


def merge_overlapping_edges(list_of_edges):
    if not list_of_edges:
        return []

    merged_edges = []
    list_of_edges.sort(key=lambda edge: edge[0])  # Sort edges by their start index

    current_edge = list_of_edges[0]
    for i in range(1, len(list_of_edges)):
        if list_of_edges[i][0] <= current_edge[1]:
            # Edges overlap, merge them
            current_edge = (current_edge[0], max(list_of_edges[i][1], current_edge[1]))
        else:
            merged_edges.append(current_edge)
            current_edge = list_of_edges[i]

    merged_edges.append(current_edge)  # Add the last merged edge
    return merged_edges


def plot_etalon_detection(
    residual,
    merged_edges,
    props,
    plot_bl=False,
    title=None,
    label=None,
    fig=None,
    color=None,
    plot_peaks=False,
    plot_abs_residual=False,
    plot_edges=True,
    plot_noise_floor=False,
):
    if fig is None:
        plt.figure()
    elif type(fig) == int:
        plt.figure(fig)
    elif type(fig) == plt.Figure:
        plt.figure(fig.number)
    else:
        raise ValueError("fig must be None, int, or matplotlib.pyplot.Figure")

    # Add peaks to the plot
    plt.plot(residual, label=label)
    color = plt.gca().lines[-1].get_color()

    if plot_abs_residual:
        plt.plot(np.abs(residual))

    if plot_bl:
        # plot the baseline the same color as the residual

        plt.vlines(props["baseline"], -0.01, 0.01, color=color, linestyle="--")

    if plot_edges:
        for edge in merged_edges:
            plt.axvspan(edge[0], edge[1], alpha=0.1, color=color)

    if plot_peaks:
        peaks = props["peaks"]
        plt.plot(peaks, np.abs(residual)[peaks], "x")

    if plot_noise_floor:
        plt.hlines(noise_floor, 0, len(residual), color="black", linestyle="--")

    plt.ylim(-0.01, 0.01)

    # Add major ticks every 100 and minor (without labels) every 25
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(25))

    plt.title(title)


def plot_n_peaks_vs_distance_threshold(props, label=None):
    plt.figure(2, figsize=(4, 5))
    # Plot the number of peaks as a function of distance_threshold
    distance_thresholds = np.arange(0, 200)
    num_peaks = []
    for distance_threshold in distance_thresholds:
        merged_peaks = merge_edges(props["edges"], distance_threshold)
        num_peaks.append(len(merged_peaks))
        num_peaks.append(len(merged_peaks))

    plt.plot(distance_thresholds, num_peaks, label=label)
    plt.ylabel("Number of Notches")
    plt.xlabel("Distance Threshold for merge_edges")


def extract_etalon(summary_df, retro, etalon_loc):
    """Given etalon analysis summary, a specific retro, and an etalon
    location, merge all overlapping etalons,
    find all etalons that overlap the given location and return the left and right edges of the merged etalon.

    Use Case: after getting etalons for a bunch of residuals, we want to summarize these parameters for use in processing.  (This requires doing some manual analysis to figure out where the "clumps" of etalons are)


    Parameters:
    ----------
    summary_df : DataFrame
        DataFrame of etalon analysis summary. Assumes there is a column 'etalons' which is a list of tuples
    retro : str
        Retro name.
    etalon_loc : int
        Etalon location.

    Returns:
    -------
    overlapping_etalons : tuple
        Tuple of left and right edges of the merged and overlapping etalon.
    """

    subset_df = summary_df[summary_df["retro"] == retro]

    # Extract all etalons from the summary_df and put in a big list
    edges = []
    for etalon in subset_df["etalons"].values:
        # etalon_list = ast.literal_eval(etalon) # Fixed by using converter in pd.read_csv
        edges.extend(etalon)

    merged_edges = merge_edges(edges, distance_threshold=0)

    # Find the tuple inside merged_edges that contains the etalon_loc
    overlapping_etalons = []
    for edge in merged_edges:
        if edge[0] <= etalon_loc <= edge[1]:
            overlapping_etalons.append(edge)

    try:
        return overlapping_etalons[0]
    except IndexError:
        print(f"No etalon found at {etalon_loc} for {retro}")
        print(f"Original etalons: {edges}")
        print(f"Merged etalons: {merged_edges}")
        return None


def get_peak_edges(
    residual,
    noise_multiplier=10,
    width_extraction_rel_height=0.75,
    distance_threshold=35,
    recursive_iters=1,
    noise_floor_method="max",
):
    """Find the edges of peaks in a residual signal.

    Parameters
    ----------
    residual : array
        Residual signal.
    noise_multiplier : float
        Multiplier for the noise floor. Default is 10.
    width_extraction_rel_height : float
        Relative height of the peak to use for extracting the width of the peak.
        Used in scipy.signal.peak_widths. Default is 0.75.
    distance_threshold : int
        Maximum distance allowed between edges for merging. Default is 35.
    recursive_iters : int
        Number of times to recursively merge peaks. Default is 1.
    noise_floor_method : str
        Method for extracting the noise floor. Must be 'std' or 'max'.

    Returns
    -------
    merged_edges : list
        List of tuples of merged left and right edges of peaks.
    properties : dict
        Dictionary of properties of the etalon extraction process.


    """
    abs_residual = np.abs(residual)
    noise_floor = extract_noise_floor(abs_residual, method=noise_floor_method)
    peaks, peak_props = signal.find_peaks(
        abs_residual, height=noise_multiplier * noise_floor
    )

    widths, width_heights, left_edges, right_edges = signal.peak_widths(
        np.abs(residual), peaks, rel_height=width_extraction_rel_height
    )

    edges = list(
        zip(np.floor(left_edges).astype(int), np.ceil(right_edges).astype(int))
    )

    merged_edges = merge_edges(edges, distance_threshold=distance_threshold)

    # Turn first detected etalon into baseline
    baseline = merged_edges[0][1]
    merged_edges = merged_edges[1:]

    etalon_std = [np.std(residual[peak[0] : peak[1]]) for peak in merged_edges]

    properties = {
        "noise_floor": noise_floor,
        "peaks": peaks,
        "peak_props": peak_props,
        "widths": widths,
        "width_heights": width_heights,
        "left_edges": left_edges,
        "right_edges": right_edges,
        "edges": edges,
        "baseline": baseline,
        "etalon_size": etalon_std,
    }

    return merged_edges, properties


def initialize_parser():
    parser = argparse.ArgumentParser(description="Summarize etalon detection.")

    parser.add_argument(
        "--r",
        "--residual",
        help="Path to residual *_.csv files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--d",
        "--data",
        help="Path to data files. This can be either the directory containing "
        "the .log files from measurement, "
        "or the directory containing the output files from the fit",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--s",
        "--save_path",
        help="Path to save figures",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--o",
        "--output_path",
        help="Output path for summary csv",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--n",
        "--number_top_files",
        help="Number of top P2P values to run the etalon detection on. Specify 0 to run on all files. Default is 5.",
        type=int,
        default=5,
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    args = initialize_parser()

    path_to_residuals = args.r
    path_to_data = args.d
    save_fig_path = args.s
    output_path = args.o
    number_top_files = args.n

    files = glob.glob(os.path.join(path_to_residuals, "*.csv"))

    list_of_dfs = []

    # This is a hack to use the path to data as either to the .log from measurements
    # or the output files from the fit.
    realtime_results_df = create_dcs_dataframe(
        path_to_data, correct_pathlength=False, drop_na=False
    )

    if number_top_files > 0:
        top_files = realtime_results_df.groupby("ch4retro").apply(
            lambda x: x.nlargest(number_top_files, "dataP2P")
        )
    else:
        top_files = realtime_results_df

    retros = realtime_results_df["ch4retro"].unique()

    print("Processing files...")
    for file in tqdm(files):
        try:
            trimmed_filename = os.path.basename(file).split("_")[0]

            if int(trimmed_filename) not in top_files["fileNames"].values:
                continue

            # Get the beam name from the log file. This is a hacky way to allow for either
            # the log files from the measurement or the output files from the fit.
            # TODO: clean this up
            try:
                beam_name = beam_from_log_files(path_to_data, trimmed_filename)
            except FileNotFoundError as e:
                beam_name_entry = realtime_results_df[
                    realtime_results_df["fileNames"] == int(trimmed_filename)
                ]["ch4retro"]

                if len(beam_name_entry) != 1:
                    raise ValueError(
                        f"Could not find beam name for {trimmed_filename}. Found {len(beam_name_entry)} entries."
                    )
                else:
                    beam_name = beam_name_entry.values[0]

            residual = np.array(pd.read_csv(f"{file}", header=None)).ravel()

            merged_edges, props = get_peak_edges(residual)

            output_info = {
                "filename": trimmed_filename,
                "retro": beam_name,
                "baseline": props["baseline"],
                "etalons": [merged_edges],
                "etalon_size": [props["etalon_size"] / props["noise_floor"]],
            }

            list_of_dfs.append(pd.DataFrame(output_info))

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

        summary = pd.concat(list_of_dfs)

        if output_path is not None:
            # Check if the output path exists and if not, create it
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            summary.to_csv(os.path.join(output_path, "etalon_summary.csv"), index=False)

        if save_fig_path is not None:
            # look up beam name in retros list
            retro_idx = list(retros).index(beam_name)

            plot_etalon_detection(
                residual,
                merged_edges,
                props,
                plot_bl=True,
                label=trimmed_filename,
                fig=retro_idx,
                title=beam_name,
            )
    print("Making plots")
    for fig in tqdm(plt.get_fignums()):
        plt.figure(fig)
        plt.legend()
        plt.xlim([0, 1000])
        # plt.show()

        if save_fig_path is not None:
            plt.savefig(
                os.path.join(save_fig_path, f"etalon_detection_{retros[fig]}.png"),
                dpi=600,
            )


if __name__ == "__main__":
    main()
