import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from cmcrameri import cm

computers = ["computer_1", "computer_2"]
baselines = ["narrowband", "wideband"]

# baselines = ["narrowband"]
# computers = ["computer_1"]

fig, axs = plt.subplots(
    len(computers), len(baselines), figsize=(10, 10), sharex=True, sharey="col"
)

if len(computers) > 1 or len(baselines) > 1:
    axs = axs.flatten().T
else:
    axs = [axs]


for i, (computer, baseline) in enumerate(itertools.product(computers, baselines)):
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
# %%
import nna_tools

path = "/Users/elimiller/UCB-O365/Precision Laser Diagnostics Lab-Data sharing - NNA/computer_1/output/BS_telescope/realtime_output_narrowband/output"

df = nna_tools.create_dcs_dataframe(path, correct_pathlength=False, drop_na=False)
nna_tools.plot_concentrations_byretro(df)

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
