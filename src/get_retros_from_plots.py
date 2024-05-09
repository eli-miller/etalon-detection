from pldspectrapy import beam_from_log_files
import glob
import os
import argparse

""" 
This script is designed to take a directory of fit plots and a directory of log files and return a dictionary of beam names for each plot.
Additional functionality is added to optionally take in a list of files within the directory and only return beam names for those files.
"""

parser = argparse.ArgumentParser(
    description="Get beam names from fit plots and log files. Chose either directory of plots or a list of plots to process."
)
parser.add_argument("--plots_path", type=str, help="Path to directory of fit plots.")
parser.add_argument("--log_file_path", type=str, help="Path to directory of log files.")
parser.add_argument(
    "--file_list",
    type=str,
    help="text file with List of files to process. If not provided, all files in plots_path will be processed.",
    required=False,
    default=None,
)
parser.add_argument(
    "--save_path",
    type=str,
    help="Path to save beam_dict.txt file to.",
    required=False,
    default=None,
)


args = parser.parse_args()

plots_path = args.plots_path
log_file_path = args.log_file_path
save_path = args.save_path
file_list = args.file_list
print(file_list)

if file_list is not None:
    with open(file_list, "r") as f:
        files = f.read().splitlines()

else:
    files = glob.glob(plots_path + "/*.png")

beam_dict = {}
for file in files:
    filename = os.path.basename(file).split("_")[0]
    try:
        beam_dict[filename] = beam_from_log_files(log_file_path, filename)
    except FileNotFoundError:
        print(f"Could not find log file for {filename}. Skipping.")

# Save beam_dict to a text file. Newline delimited.
if save_path is not None:
    with open(os.path.join(save_path, "beam_dict.txt"), "w") as f:
        for key, value in beam_dict.items():
            f.write("%s:%s\n" % (key, value))

else:
    print(beam_dict)
