import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from cmcrameri import cm

import etalon_detection_analysis.src.plot_fit_results
import pldspectrapy as pld
from etalon_detection_analysis.src import plot_fit_results

# %% File of interest
file_of_interest = 20240415072446
file_of_interest = 20240417150445
computer = "computer_1"

bandwidth = "wideband"
path_to_results = f"/Users/elimiller/UCB-O365/Precision Laser Diagnostics Lab-Data sharing - NNA/{computer}/output/BS_telescope/realtime_output_{bandwidth}"


fig, axs = pld.generate_fit_plots(
    path_to_results, file_of_interest, baseline=600, etalons=[[8500, 8700]]
)

axs[0, 0].set_ylim(-0.005, 0.005)
axs[1, 0].set_ylim(-0.015, 0.015)
fig.suptitle(f"File: {file_of_interest} ({computer} {bandwidth})")
# axs[1,0].set_ylim(-0.1, 0.1)

# %% Etalon and Baseline Summary

computers = ["computer_1", "computer_2"]
bandwidths = ["narrowband", "wideband"]

# baselines = ["narrowband"]
# computers = ["computer_1"]

fig, axs = plt.subplots(
    len(computers), len(bandwidths), figsize=(10, 10), sharex=True, sharey="col"
)

if len(computers) > 1 or len(bandwidths) > 1:
    axs = axs.flatten().T
else:
    axs = [axs]


for i, (computer, baseline) in enumerate(itertools.product(computers, bandwidths)):
    path = f"/Users/elimiller/Documents/1Research/python_project/etalon_detection_analysis/output/bs_telescope/{computer}_{baseline}_top_P2P/etalon_summary.csv"
    data = pd.read_csv(path)
    data.sort_values("retro", inplace=True)

    sns.boxplot(x="retro", y="baseline", data=data, ax=axs[i])
    sns.stripplot(
        x="retro",
        y="baseline",
        data=data,
        jitter=True,
        alpha=0.5,
        color="black",
        ax=axs[i],
    )
    axs[i].set_title(f"{computer} {baseline}")
    axs[i].xaxis.set_tick_params(rotation=90)

fig.suptitle("Baseline Extraction Comparison")
plt.tight_layout()
# %% Comparing Results
import nna_tools


path = "/Users/elimiller/UCB-O365/Precision Laser Diagnostics Lab-Data sharing - NNA/computer_1/output/BS_telescope/realtime_output_narrowband_tuned_etalons/output"

df = nna_tools.create_dcs_dataframe(path, correct_pathlength=False, drop_na=False)
nna_tools.plot_concentrations(df, ebar=False)


# %%
plt.figure()
# plot boxplot and stripplot of ch4_1_scaled values by retro
sns.boxplot(x="ch4retro", y="ch4_1_scaled", data=df, color="white")
sns.stripplot(
    x="ch4retro",
    y="ch4_1_scaled",
    data=df,
    jitter=True,
    alpha=0.5,
    hue=df.index,
    legend=False,
)
# %%
sns.boxplot(y="ch4retro", x="dataP2P", data=df)
sns.stripplot(y="ch4retro", x="dataP2P", data=df, jitter=True, alpha=0.5)
# Turn x labels
# plt.xticks(rotation=90)
plt.tight_layout()


# %%

# Grouping by ch4retro, find the top 5 files with the highest dataP2P values
top_files = df.groupby("ch4retro").apply(lambda x: x.nlargest(5, "dataP2P"))


# Filter data (the etalon results) by the top_files (from the output processing)


sns.stripplot(
    y="ch4retro", x="baseline", data=df_filtered, jitter=False, color="red", marker="x"
)


# %%
import nna_tools

list_of_dfs = []
for i, (computer, bandwidth) in enumerate(itertools.product(computers, bandwidths)):
    path = f"/Users/elimiller/UCB-O365/Precision Laser Diagnostics Lab-Data sharing - NNA/{computer}/output/BS_telescope/realtime_output_{bandwidth}/output"
    df = nna_tools.create_dcs_dataframe(path, correct_pathlength=False, drop_na=False)
    df["computer"] = computer
    df["bandwidth"] = bandwidth
    list_of_dfs.append(df)

df = pd.concat(list_of_dfs)


computer = "computer_1"
df_filtered = df.query(f"computer == '{computer}' ")


# Make a list of the value oc ch4retro sorted by their mean value of point1
retro_order = list(df_filtered.groupby("ch4retro")["point1"].mean().sort_values().index)

fields_to_plot = ["ch4_1_scaled", "co2_1_scaled", "h2o_1_scaled"]

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

for i, field in enumerate(fields_to_plot):
    sns.scatterplot(
        x=df_filtered.index,
        y=field,
        hue="ch4retro",
        data=df_filtered,
        style="bandwidth",
        hue_order=retro_order,
        ax=axs[i],
        # legend=False,
    )


axs[2].legend().remove()
axs[1].legend().remove()
axs[0].legend().remove()
# make the legend face white and no transparency
# plt.legend(facecolor="white", framealpha=1)
# turn x labels 45 degrees
plt.xticks(rotation=45)
fig.suptitle(f"pre-tuning comparison ({computer})")

# %% absorbance snr vs peak2peak
plt.figure(figsize=(7, 5))

sns.scatterplot(
    x="dataP2P",
    y="absSNR",
    data=df,
    # hue="ch4retro",
    hue="igs_accepted",
)

# Add line to the log log plot and show the equation


# make log log
plt.xscale("log")
plt.yscale("log")
plt.grid(which="major", linestyle="-", linewidth="0.5", color="black")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")

# move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title="IGS Accepted")

plt.title("Computer 1 (narrowband)")


# %%
from etalon_detection_analysis.src.plot_fit_results import (
    parse_xml,
    extract_bl_and_etalons,
)

config_path = "/Users/elimiller/UCB-O365/Precision Laser Diagnostics Lab-Data sharing - NNA/computer_1/output/BS_telescope/realtime_output_narrowband_tuned_etalons/master_configurations_beamsplitter_narrowband_tuned_etalons.xml"
retros = df.ch4retro.unique()
root = parse_xml(config_path)

retro_values = {}
for retro in retros:
    retro_values[retro] = extract_bl_and_etalons(root, retro)

retro_values_df = pd.DataFrame(retro_values).T.dropna()
# rename the columns with the lowercase version of the column name
retro_values_df.columns = retro_values_df.columns.str.lower()
retro_values_df["num_etalons"] = retro_values_df.etalons.apply(len)

# make a plot that shades between the etalons. y axis just zero to one and x axis 0-1000
plt.figure(figsize=(10, 5))
for i, (retro_name, row) in enumerate(retro_values_df.iterrows()):
    for etalon in row.etalons:
        plt.fill_between(
            [etalon[0], etalon[1]],
            i,
            i + 1,
            alpha=0.5,
            label=f"{etalon} etalon {etalon}",
        )
        # add the retro name as the y label
    plt.text(0, i + 0.5, retro_name, fontsize=8)
