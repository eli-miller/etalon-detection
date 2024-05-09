import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm


from pldspectrapy import td_support as td
import nna_tools

plt.style.use("eli_default")


def parse_xml(xml_file_path):
    """
    Parse an XML file and return the root element.
    """
    tree = ET.parse(xml_file_path)
    return tree.getroot()


def extract_values_for_section(root, section_name, tag_names):
    """
    Extract values for a specific section in the XML.

    Args:
        root (Element): The root element of the XML.
        section_name (str): The name of the section to extract.
        tag_names (list): A list of tag names to extract values from.

    Returns:
        dict: A dictionary containing the extracted values.
    """
    section = root.find(section_name)
    if section is None:
        return None

    values = {}
    for tag_name in tag_names:
        element = section.find(tag_name)
        if element is not None:
            values[tag_name] = element.text

    return values


def extract_bl_and_etalons(root, retro_name):
    """
    Extract BL and Etalons values for a specific retro in the XML.

    Args:
        root (Element): The root element of the XML.
        retro_name (str): The name of the retro to extract values for.

    Returns:
        dict: A dictionary containing BL and Etalons values.
    """
    spectrapy = root.find("spectrapy")
    retros = spectrapy.find("retros")

    for retro in retros:
        if retro.tag == retro_name:
            bl = retro.find("bl").text
            etalons = retro.find("etalons").text
            etalons_list = eval(etalons)

            values = {"BL": bl, "Etalons": etalons_list}
            return values

    return None


def plot_residuals(
    axs,
    residual_data,
    y_td_data,
    retro_name,
    filename,
    plot_gray=False,
    plot_legend=False,
):
    # TODO: normalize residuals in plot
    axs[0].plot(y_td_data[filename], label=filename)

    if plot_gray:
        axs[1].plot(
            residual_data[filename],
            alpha=0.1,
            color="gray",
        )
    else:
        axs[1].plot(residual_data[filename])

    axs[0].set_title(retro_name)
    axs[0].set_ylabel("y_data_i")
    axs[0].set_ylim(-0.01, 0.015)
    axs[1].set_ylabel("Residual")
    axs[1].set_ylim(-0.005, 0.005)
    axs[1].set_xlim(0, 1000)

    if plot_legend:
        axs[0].legend(loc="upper right", ncols=2)
    else:
        axs[0].legend().set_visible(False)

    # Add major x ticks at multiples of 100
    axs[1].set_xticks(np.arange(0, 1000, 100))
    # add minor x ticks at multiples of 10
    axs[1].set_xticks(np.arange(0, 1000, 10), minor=True)

    inset = axs[2]
    # Add an inset on the second subplot from x=8000-9000 and y = -.0005 to .0005
    if plot_gray:
        inset.plot(residual_data[filename], alpha=0.1, color="gray")
    else:
        inset.plot(residual_data[filename])
    inset.set_xlim(8400, 8800)
    inset.set_ylim(-0.0005, 0.0005)
    # Move inset y label to the right
    inset.yaxis.set_label_position("right")


def plot_notches(axs, weight):
    n_weigt = len(weight)
    weight_plot = (np.ones(len(weight)) - weight)[0 : int(n_weigt / 2)]

    axs[0].fill_between(
        np.arange(len(weight_plot)),
        weight_plot,
        -weight_plot,
        color="lightblue",
        alpha=0.3,
        zorder=10,
    )
    axs[1].fill_between(
        np.arange(len(weight_plot)),
        weight_plot,
        -weight_plot,
        color="lightblue",
        alpha=0.3,
        zorder=10,
    )


def setup_args():
    parser = argparse.ArgumentParser(description="Plot fit results.")
    parser.add_argument(
        "-r",
        "--results",
        type=str,
        help="Path to the results directory.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        help="Path to save plots.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--grayscale",
        action="store_true",
        help="Plot in grayscale. Disabled by default.",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display the plots.",
        default=False,
    )

    parser.add_argument(
        "-p",
        "--p2p",
        type=float,
        help="Peak to peak value threshold of files to plot.",
        default=0.0,
    )

    parser.add_argument(
        "-l",
        "--legend",
        action="store_true",
        help="Plot legend.",
        default=False,
    )

    return parser.parse_args()


def main():
    """
    Plot residuals and y_td data for each retro in the results directory.

    """
    args = setup_args()

    results_path = args.results
    config_paths = args.config
    save_path = args.save
    plot_gray = args.grayscale
    p2p_threshold = args.p2p
    plot_legend = args.legend

    # Read in config file to get etalon and bl info. RealTime not saving correctly so can't use RealTime.log file.
    config_path = config_paths
    retro_values = {}
    root = parse_xml(config_path)

    output_data_path = os.path.join(results_path, "output", "*RealTime.log")

    output_data = nna_tools.create_dcs_dataframe(
        output_data_path, correct_pathlength=False, drop_na=False
    )  # new retros don't have all info yet.

    # Filter by peak to peak value
    output_data = output_data[output_data["dataP2P"] > p2p_threshold]

    # get retro names from root xml file

    # for retro_name in tqdm.tqdm(output_data.ch4retro.unique()):
    for retro_name in output_data.ch4retro.unique():
        # Catch the case where the retro name is a float (nan)
        if type(retro_name) == float:
            print(f"Skipping retro_name: {retro_name}")
            continue

        filenames = output_data[output_data["ch4retro"] == retro_name]["fileNames"]
        retro_values[retro_name] = extract_bl_and_etalons(root, retro_name)

        try:
            bl = int(retro_values[retro_name]["BL"])
            etalons = retro_values[retro_name]["Etalons"]
        except TypeError:
            print(f"Skipping retro_name: {retro_name}")
            continue

        residual_data = {}
        y_td_data = {}
        fig, axs = plt.subplots(
            2, 1, figsize=(9.6, 5), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
        )

        inset = axs[1].inset_axes([0.6, 0.6, 0.3, 0.3])
        # add the inset axes to the axs object. it is an array so we can't append
        axs = np.append(axs, inset)

        for filename in tqdm(filenames):
            filename = str(filename)

            residual_path = os.path.join(
                results_path, "residuals", filename + "_residual.csv"
            )
            cepstrum_path = os.path.join(results_path, "y_td", f"{filename}_y_td.csv")

            if not os.path.exists(residual_path):
                print(f"Skipping file: {filename}. The residual file does not exist.")
                continue
            if not os.path.exists(cepstrum_path):
                print(f"Skipping file: {filename}. The cepstrum file does not exist.")
                continue

            residual_data[filename] = np.loadtxt(
                residual_path,
                delimiter=",",
            )

            y_td_data[filename] = np.loadtxt(
                cepstrum_path,
                delimiter=",",
            )

            plot_residuals(
                axs,
                residual_data,
                y_td_data,
                retro_name,
                filename,
                plot_gray=plot_gray,
                plot_legend=plot_legend,
            )

        if len(residual_data) == 0:
            plt.close()
            continue

        # This throws errors if the last file didn't exist. Need to rewrite to handle this.
        # Get an entry from residual_data to get the length of the residual data. Don't use the last file.

        residual_length = len(residual_data[list(residual_data.keys())[0]])

        weight = td.weight_func(residual_length, bl, etalons=etalons)

        plot_notches(axs, weight)

        if save_path is not None:
            plt.savefig(os.path.join(save_path, f"{retro_name}_residuals.png"), dpi=600)

        if not args.display:
            plt.close()


if __name__ == "__main__":
    main()
