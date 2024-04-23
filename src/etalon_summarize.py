import json
import os

import pandas as pd
import ast

import etalon_detection
import argparse


def setup_args():
    parser = argparse.ArgumentParser(description="Summarize etalon detections.")
    parser.add_argument(
        "--s",
        "--summary_path",
        help="Path to etalon_summary.csv output by `etalon_detection`",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--e",
        "--etalon_centers_path",
        help="Path to manually created csv summarizing which etalon centers to aggregate",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--o",
        "--output_path",
        help="Path of where to put summary etalon_union.json",
        type=str,
        required=True,
    )

    return parser.parse_args()


def parse_list_of_tuples(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None  # Handle any parsing errors gracefully


def calculate_etalon_union(retro, etalon_centers):
    etalon_union = []
    for etalon_center in etalon_centers:
        etalon_union.append(
            etalon_detection.extract_etalon(
                summary_df=summary, retro=retro, etalon_loc=int(etalon_center)
            )
        )
    return etalon_union


# Specify the converters for the "etalons" and "etalon_size" columns
# TODO: this is all because of dumb ways that I've chosen to store data
#  Refactor so that etalon_centers_df is a dict with keys retro and values lists of etalon_centers
def main():
    args = setup_args()

    summary_path = args.s
    etalon_centers_path = args.e
    save_path = args.o

    converters = {
        "etalons": parse_list_of_tuples,
        "etalon_size": parse_list_of_tuples,
        "etalon_centers": parse_list_of_tuples,
    }

    summary = pd.read_csv(summary_path, converters=converters)

    etalon_centers_df = pd.read_csv(etalon_centers_path, converters=converters)

    config_params = {}

    for retro in etalon_centers_df.retro.unique():
        retro_etalon_centers = etalon_centers_df[etalon_centers_df["retro"] == retro][
            "etalon_centers"
        ].values[0]
        # print(retro_etalon_centers)

        joined_etalons_list = []
        for etalon_center in retro_etalon_centers:
            joined_etalon = etalon_detection.extract_etalon(
                summary_df=summary, retro=retro, etalon_loc=etalon_center
            )

            joined_etalons_list.append(joined_etalon)

        # Add to config_params dictionary.  The first level should be retro name
        # The second level should have keys "etalons" and "bl" (baseline)
        config_params[retro] = {}
        config_params[retro]["etalons"] = joined_etalons_list

        # Pull bl from etalon_centers_df
        config_params[retro]["bl"] = int(
            etalon_centers_df[etalon_centers_df["retro"] == retro]["baseline"].values[0]
        )

    if save_path is not None:
        save_name = "etalon_union.json"
        with open(os.path.join(save_path, save_name), "w") as f:
            json.dump(config_params, f)


if __name__ == "__main__":
    main()
