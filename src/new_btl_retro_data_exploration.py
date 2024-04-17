import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('eli_default')
plt.switch_backend('macosx')
import numpy as np
import seaborn as sns
import nna_tools
import cmcrameri.cm as cm
import warnings


# Suppress FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

filepath = "/Users/elimiller/UCB-O365/Precision Laser Diagnostics Lab-Data sharing - NNA/computer_2/output/*.log"
data = nna_tools.create_dcs_dataframe(filepath, correct_pathlength=False)
# # Filter to only keep after specified date
data = data[data["datetime"] > pd.to_datetime("2023-09-1")]

ec_path = "/Users/elimiller/Documents/1Research/python_project/NNA_data_analysis/etalon_detection_analysis/data/*.txt"
ec_data = nna_tools.create_ec_dataframe(ec_path)

#%%
# nna_tools.plot_concentrations(data, threshold=.1)

# Between 11/8/23 and 11/9/23, apply a .describe grouped by retro for methane, water, and co2 concentrations
# subset the data to the date range
data_for_stats = data[(data.index > pd.to_datetime("2023-11-8 20:00")) & (data.index < pd.to_datetime("2023-11-9"))]

# Group by retro and compute the .describe
stats_df = data_for_stats.groupby('ch4retro')[['ch4_1_scaled', 'co2_1_scaled', 'h2o_1_scaled','pathlength']].describe(percentiles=[0.5])
stats_df.to_csv(r"/Users/elimiller/Downloads/new_retro_stats.csv")

# Plot the mean ch4 concentration by pathlength for each retro from stats using a multi index
plt.scatter(stats_df['pathlength']['mean'], stats_df['ch4_1_scaled']['mean'])
plt.scatter(stats_df['pathlength']['mean'], stats_df['co2_1_scaled']['mean'])
plt.scatter(stats_df['pathlength']['mean'], stats_df['h2o_1_scaled']['mean'])

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Assume that the stats DataFrame is already defined

fig, ax = plt.subplots(1, 3)

plots = ['ch4_1_scaled', 'co2_1_scaled', 'h2o_1_scaled']

for i, plot in enumerate(plots):
    sns.regplot(x=stats_df['pathlength']['mean'], y=stats_df[plot]['mean'], ax=ax[i])

    # Perform a linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        stats_df['pathlength']['mean'], stats_df[plot]['mean'])

    # Generate the equation of the line
    line_eq = "y = {:.2f}x + {:.2f}".format(slope, intercept)

    # Annotate the plot with the equation
    ax[i].annotate(line_eq, xy=(0.1, 0.9), xycoords='axes fraction',
                   fontsize=12, color="red")

    ax[i].set_title(f'Regression plot for {plot}')

plt.tight_layout()
plt.show()


#%%
# Use pairplot to plot the relationship between ch4_1_scaled, co2_1_scaled, and h2o_1_scaled from stats_df
# Only do this for the mean values

# Only keep stats_df of entries with count greater than 3
stats_df = stats_df[stats_df['ch4_1_scaled']['count'] > 3]

mean_stats_df = stats_df.xs('mean', level=1, axis=1)
median_stats_df = stats_df.xs('50%', level=1, axis=1)

sns.pairplot(mean_stats_df, corner=True)
# TItle the plot
plt.suptitle('Mean values')

sns.pairplot(median_stats_df, corner=True)
plt.yticks(rotation=0)
plt.suptitle('Median values')
#%%

# sns.pairplot(data=data_retros[['ch4_1_scaled', 'co2_1_scaled', 'h2o_1_scaled', 'ch4retro']], hue='ch4retro', plot_kws={'alpha': 0.1, 'edgecolor': None}, corner=True)

# %%
# Subset retros to eliminate those that only were measred a few times.
# This helps with plot color allocation
retros = ["NorthDock",
          "AboveNorthDock"]  # , "WestDock", "AboveWestDock", "AboveEC"]

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10),
                        gridspec_kw={'height_ratios': [.5, 1.5, 1]})

ax_for_plot_concentration = axs[-2::]
data_retros = data[data["ch4retro"].isin(retros)]

nna_tools.plot_concentrations(data_retros, field='ch4_1_scaled',
                              threshold=0.05, wind_kws={'hue_norm': (0, 5)},
                              ax=ax_for_plot_concentration)

plt.tight_layout()

sns.scatterplot(data=data_retros, x=data_retros.index,
                y='temperature_evolution',
                # hue='ch4retro',
                ax=axs[0],
                edgecolor=None,
                )
#
# sns.scatterplot(data=data_retros, x=data_retros.index,
#                 y='pressure_evolution',
#                 # hue='ch4retro',
#                 ax=axs[1],
#                 edgecolor=None,
#                 )

axs[0].set_ylabel('Temp (C)')

fig.suptitle('Incorrect T&P affects fit results')
axs[1].set_ylabel('CH4 (ppm)')

# despine all axes
for ax in axs:
    sns.despine(ax=ax)

# Remove bottom subplot legend
axs[2].legend().remove()
plt.tight_layout()

#%%
plt.figure()
data['pathlength_minus_mean'] = data.groupby('ch4retro')['pathlength'].transform(lambda x: x - x.mean())

sns.lineplot(data=data, x=data.index, y='pathlength_minus_mean', hue='ch4retro', marker='o')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
# %%

# Remove all data with ch4_1_error values greater than 0.05

data_retros = data_retros[data_retros["ch4_1_error"] < 0.025]
rolling_period = 10


# subtract the mean ch4_1_scaled value for each retro
data_retros['ch4_1_scaled_zero_meaned'] = data_retros.groupby('ch4retro')[
    'ch4_1_scaled'].transform(lambda x: x - x.median())


# Compute the ch4_1_scaled rolling mean and subtract it from the ch4_1_scaled for each retro
data_retros['ch4_1_scaled_rm'] = data_retros.groupby('ch4retro')[
    'ch4_1_scaled_zero_meaned'].transform(lambda x: x.rolling(rolling_period).mean())


data_retros['ch4_1_scaled_rm_sub'] = data_retros['ch4_1_scaled'] - data_retros[
    'ch4_1_scaled_rm']
data_retros['ch4_1_scaled_rm_sub_rm'] = data_retros.groupby('ch4retro')[
    'ch4_1_scaled_rm_sub'].transform(lambda x: x.rolling(rolling_period).mean())


fig, axs = nna_tools.plot_concentrations(data_retros, field='ch4_1_scaled',
                                         threshold=0.05, wind_kws={'hue_norm': (0, 5)})

# set x axis limits
axs[0].set_xlim(pd.to_datetime("2023-11-13"), pd.to_datetime("2023-11-18 10:00"))

fig, axs = nna_tools.plot_concentrations(data_retros, field='ch4_1_scaled_rm',
                                         threshold=0.05, wind_kws={'hue_norm': (0, 5)})
axs[0].set_xlim(pd.to_datetime("2023-11-13"), pd.to_datetime("2023-11-18 10:00"))
axs[0].set_ylim(-.15, .25)

# %% Figure out where the bias in concentration is coming from
# For each retro, compute the difference between mean ch4_1_scaled

# Compute the mean ch4_1_scaled for each retro

fields = ['ch4_1_scaled', 'co2_1_scaled', 'h2o_1_scaled']

data_retros.groupby('ch4retro')[fields].mean()

# %%
# Compute the difference between the AboveNorthDock and NorthDock ch4_1_scaled rolling mean.
# Will need to interpolate to a common time index

# Choose relevant columns to keep in the data

# Get the AboveNorthDock data
data_AN = data_retros[data_retros["ch4retro"] == "AboveNorthDock"][
    ['ch4_1_scaled_rm']].dropna().sort_index()

# Get the NorthDock data
data_N = data_retros[data_retros["ch4retro"] == "NorthDock"][
    ['ch4_1_scaled_rm']].dropna().sort_index()

# Interpolate the AboveNorthDock data to the NorthDock time index
data_AN_interp = data_AN.reindex(data_N.index)

# Compute the difference between the AboveNorthDock and NorthDock ch4_1_scaled rolling mean.

# Plot the difference
plt.figure()
plt.plot(data_AN_interp.index, data_AN_interp['ch4_1_scaled_rm'])

# %%
# Print etalon parameters for AboveNorthDock and the timestamp

# Get the AboveNorthDock data
data_AN = data[data["ch4retro"] == "AboveNorthDock"]

# Get the AboveNorthDock etalon parameters
data_AN[['etalons_original']].value_counts()

# %%
retro_pairs = [["NorthDock", "AboveNorthDock"], ["WestDock", "AboveWestDock"]]
fields = ["ch4_1_scaled", "co2_1_scaled", "h2o_1_scaled"]

# Plot retro pairs together for each field over time (index)

fig, axs = plt.subplots(len(fields), len(retro_pairs), figsize=(10, 6),
                        sharex=True, sharey='row')

for i_retro_pair, retro_pair in enumerate(retro_pairs):
    data_retro_pair = data[data["ch4retro"].isin(retro_pair)]

for i_field, field in enumerate(fields):
    sns.scatterplot(data=data_retro_pair,
                    x=data_retro_pair.index,
                    y=field,
                    hue='ch4retro',
                    ax=axs[i_field, i_retro_pair])

# Turn x axis labels 45 degrees

plt.tight_layout()

# %%

# Plot dataP2P values with boxplot and jittered observations for each retro. Log y axis

plt.figure()

field = 'ch4_1_scaled'

sns.boxplot(data=data_retros, x='ch4retro', y=field, showfliers=False)
sns.stripplot(data=data_retros, x='ch4retro', y=field, color='black',
              alpha=0.5)

# Add title with date range, rounding to the hour
plt.title(
    f"{field} values from {data_retros.index[0].round('H')} to {data_retros.index[-1].round('H')}")

# %%

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10),
                        gridspec_kw={'height_ratios': [1, 1, 1, .3]})
# fig2, axs2 = plt.subplots()

# Plot ch4_1_scaled vs time for each retro. Include error bars
for i, retro in enumerate(
        data.ch4retro.unique()):  # errorbar doesnt work with hue
    data_retro = data[data["ch4retro"] == retro]
axs[0].errorbar(data_retro.index, data_retro["ch4_1_scaled"],
                yerr=data_retro["res_sdev"], fmt='o', label=retro,
                markersize=2)

# sns.regplot(data=data_retro, x='WSP_2D_avg', y='ch4_1_scaled',
#             ax=axs2, label=retro)

axs[0].set_ylabel("ch4_1_scaled")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# axs2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.lineplot(data=data, x=data.index, y='dataP2P',
             hue='ch4retro',
             ax=axs[3]
             )

# Remove white markers on points
sns.scatterplot(data, x=data.index, y='WDR_avg', hue='WSP_2D_avg',
                palette=cm.devon_r, hue_norm=(0, 5),
                ax=axs[1], edgecolor=None, s=10)

# Plot the EC wind data on the same plot as the DCS wind data
sns.scatterplot(data=ec_data, x=ec_data.index, y='wind_dir', ax=axs[1],
                hue='wind_speed', palette=cm.grayC_r, hue_norm=(0, 5),
                edgecolor=None, s=10)

axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=2)

axs[0].scatter(ec_data.index, ec_data['ch4_mole_fraction'], s=10, color='black')

# Add second y axis for EC ch4_mole_fraction and plot the ch4_flux
axs[2].scatter(ec_data.index, ec_data['ch4_flux'], s=10, color='C1')
axs[2].set_ylabel('ch4_flux')
# Add zero line
axs3.axhline(0, color='C1', linestyle='--')

# Turn x axis labels 45 degrees
for label in axs[-1].get_xticklabels():
    label.set_rotation(45)

# %%

# Plot a rolling standard deviation of ch4_1_scaled for each retro
plt.figure()

# Subset data after 11-8-23
data = data[data.index > pd.to_datetime("2023-11-8")]
# Remove data with ch4retro value coutns less than 100
data = data.groupby('ch4retro').filter(lambda x: len(x) > 100)

for retro in data.ch4retro.unique():
    data_retro = data[data["ch4retro"] == retro]
plt.scatter(data_retro.index, data_retro["ch4_1_scaled"].rolling(10).std(),
            label=retro)

plt.ylabel("ch4_1_scaled rolling std")

plt.legend()

# %%

problem_retros = pd.read_csv(
    "/Users/elimiller/Documents/1Research/python_project/NNA_data_analysis/etalon_detection_analysis/doc/after_rt_etalon_files").values
# Turn this df into a list of strings
problem_retros = [retro[0] for retro in problem_retros]
print(problem_retros)
# Strip off the '_plot' from the retro names and convert to an int
problem_retros = [int(retro[:-5]) for retro in problem_retros]

# Find the entries in data that have a problem retro as the fileNames value
problem_data = data[data["fileNames"].isin(problem_retros)]

print(problem_data.ch4retro)

# %%

import plotly.express as px
