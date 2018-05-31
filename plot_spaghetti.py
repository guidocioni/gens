# coding: utf-8

from netCDF4 import Dataset
import netCDF4
import matplotlib.pyplot as plt
import numpy as np # Import the Numpy package
from datetime import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab
import matplotlib.cm as colors
from glob import glob
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from geopy.geocoders import Nominatim

diri='/scratch/m/m300382/temp/gefs-plots/grib/'

t_850hpa=[]
tot_prec=[]
n_pert=[]

for fname in glob(diri+"*.nc"):
    nc=Dataset(fname)
    t_850hpa.append(nc.variables['T'][:][:,0,:,:]-273.15)
    tot_prec.append(nc.variables['tp'][:])
    if (fname[fname.find(".nc")-2:fname.find(".nc")]) =="00": #this is the control run
        n_pert.append(0)
    else:
        n_pert.append(nc.variables['T_2M'].realization)

n_pert=np.array(n_pert)
t_850hpa=np.array(t_850hpa)
tot_prec=np.array(tot_prec)
lon=nc.variables['lon'][:]
lat=nc.variables['lat'][:]
time_var = nc.variables['time']
dtime = netCDF4.num2date(time_var[:],time_var.units)

t_850hpa_point={}
tot_prec_point={}
#cities = [["Milano",45.464211,9.191383],["Roma",41.8933203,12.4829321],["Palermo",38.1112268,13.3524434]]
cities = ["Milano","Roma","Palermo"]
#for (city,latitude,longitude) in cities:
geolocator = Nominatim()
for city in cities:
    loc = geolocator.geocode(city)
    t_850hpa_point[city] = t_850hpa[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]
    tot_prec_point[city] = tot_prec[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]

sns.set(style="white")
var_to_plot=t_850hpa_point
var2_to_plot=tot_prec_point

for city_to_plot in cities:
    fig = plt.figure(1, figsize=(9,6))
    plt.plot_date(dtime,var_to_plot[city_to_plot].T,'-',lw=1)
    plt.plot_date(dtime,var_to_plot[city_to_plot].T[:,np.where(n_pert==0)[0]],'-',lw=2.5,color='black')
    plt.ylabel("850 hPa temperature [C]")
    plt.ylim(-25, 20)
    
    plt.gca().minorticks_on()
    plt.gca().grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.gca().grid(which='minor', linestyle=':', linewidth='0.3', color='black')
    
    ax2 = plt.gca().twinx()
    ax2.plot_date(dtime,var2_to_plot[city_to_plot].T,'--',lw=1.5)
    ax2.set_ylabel("Precipitation [mm]")
    ax2.set_ylim(0, 50)
    
    plt.title("GEFS forecast for "+city_to_plot+" | Run "+dtime[0].strftime("%Y%m%d %H"))
    fig.autofmt_xdate()
    
    plt.savefig("spaghetti_"+city_to_plot, dpi=150, bbox_inches='tight')
    plt.clf()