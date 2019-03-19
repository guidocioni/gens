# coding: utf-8
debug = True 
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np # Import the Numpy package
from datetime import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.gridspec as gridspec
from glob import glob
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from geopy.geocoders import Nominatim

diri='/scratch/local1/m300382/gens/grib/'
diri_images='/scratch/local1/m300382/gens/'

fileslist=sorted(glob(diri+"*.nc"))
datasets = [xr.open_dataset(files) for files in fileslist]
# Merging should take care automatically of solving every conflict in the dimensions
merged = xr.concat(datasets, 'ens_member').squeeze()

merged['t']= merged['t'] - 273.15
n_pert=merged['ens_member'].values

time=pd.to_datetime(merged['time'].values)

t_850hpa_point={}
tot_prec_point={}
snow_point={}
cities = ["Milano","Roma","Palermo"]
geolocator = Nominatim(user_agent='gefs')
for city in cities:
    loc = geolocator.geocode(city)
    t_850hpa_point[city] = merged['t'].sel(lon=loc.longitude, lat=loc.latitude, method='nearest').interpolate_na(dim='time')
    tot_prec_point[city] = merged['tp'].sel(lon=loc.longitude, lat=loc.latitude, method='nearest').interpolate_na(dim='time')
    snow_point[city]     = merged['csnow'].sel(lon=loc.longitude, lat=loc.latitude, method='nearest').interpolate_na(dim='time')

sns.set(style="white")
var_to_plot=t_850hpa_point
var2_to_plot=tot_prec_point

for city_to_plot in cities:
    fig = plt.figure(1, figsize=(9,6))
    ax = plt.gca()
    plt.plot(time, var_to_plot[city_to_plot].T,lw=1)
    plt.plot(time, var_to_plot[city_to_plot].T[:,np.where(n_pert==0)[0]],'-',lw=2.5,color='black')
    plt.ylabel("850 hPa temperature [C]")
    plt.ylim(-25, 20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle='-', linewidth='0.3', color='black')
    
    ax2 = ax.twinx()
    ax2.plot(time,var2_to_plot[city_to_plot].T,'--',lw=1.5)
    ax2.set_ylabel("Precipitation [mm]")
    ax2.set_ylim(0, 50)
    
    plt.title("GEFS forecast for "+city_to_plot+" | Run "+time[0].strftime("%Y%m%d %H"))
    fig.autofmt_xdate()
    
    if debug:
        plt.show(block=True)
    else:
        plt.savefig(diri_images+"spaghetti_"+city_to_plot, dpi=100, bbox_inches='tight') 

    plt.clf()