import xarray as xr 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
from glob import glob

dset = xr.open_dataset('energy_sources_gfs.nc')

germany = dset.sel(lat=slice(54.8, 47.5), lon=slice(6.3, 15.)).mean(dim=('lat','lon'), skipna=True).interpolate_na(dim='time')

sns.set()
fig = plt.figure(1, figsize=(16,6))

boxplot = sns.boxplot(pd.to_datetime(germany.time.values), germany.wind_power.T, fliersize=2)

plt.gca().set_xticklabels([tm.strftime('%Y-%m-%d \n %H UTC') for tm in pd.to_datetime(germany.time.values)],
 rotation=50)
plt.gca().set_ylabel("Wind Power [MW]",fontsize=14)
plt.gca().yaxis.grid(True)
plt.gca().xaxis.grid(True, color='gray', linewidth=0.2)
for ind, label in enumerate(boxplot.get_xticklabels()):
    if ind % 3 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)

plt.savefig('boxplot_wind_power.pdf', bbox_inches='tight')        
plt.clf()

sns.set()
fig = plt.figure(1, figsize=(16,6))

boxplot = sns.boxplot(pd.to_datetime(germany.time.values), germany['2m_temperature'].T, fliersize=2)

plt.gca().set_xticklabels([tm.strftime('%Y-%m-%d \n %H UTC') for tm in pd.to_datetime(germany.time.values)],
 rotation=50)
plt.gca().set_ylabel("2m Temperature [C]",fontsize=14)
plt.gca().yaxis.grid(True)
plt.gca().xaxis.grid(True, color='gray', linewidth=0.2)
for ind, label in enumerate(boxplot.get_xticklabels()):
    if ind % 3 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.savefig('2mt.pdf', bbox_inches='tight')