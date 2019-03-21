debug = False 
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import xarray as xr
import metpy.calc as mpcalc
from utils import *
import sys

folder_images='/home/mpim/m300382/gens/'

# Get the projection as system argument from the call so that we can 
# span multiple instances of this script outside
if not sys.argv[1:]:
    print('City not defined, falling back to default (Hamburg)')
    cities = ['Hamburg']
else:    
    cities=sys.argv[1:]

dset = xr.open_mfdataset(input_files, concat_dim='ens_member').squeeze()
dset = dset.metpy.parse_cf()
time = pd.to_datetime(dset['time'].values)
# Array needed for the box plot
pos = np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

sns.set(style="white")

for city in cities:
    print('Producing meteogram for %s' % city)
    lon, lat = get_city_coordinates(city)
    dset_city =  dset.sel(lon=lon, lat=lat, method='nearest').interpolate_na(dim='time')
    # Recover units which somehow are deleted by interpolate_na,
    # no idea why....
    dset_city['t'].attrs['units'] = 'K'
    dset_city['t'].metpy.convert_units('degC')

    fig = plt.figure(1, figsize=(9,6))
    ax = plt.gca()
    plt.plot(time, dset_city['t'].T, lw=1)
    plt.plot(time, dset_city['t'].sel(ens_member=0),'-', lw=2.5, color='black')
    plt.ylabel("850 hPa temperature [C]")
    plt.ylim(-10, 15)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.3', color='black')
    ax.grid(which='minor', linestyle='-', linewidth='0.1', color='black')
    ax.set_xlim(time[0], time[-1])
    ax.fill_between(time, dset_city['t'].min(dim='ens_member'), dset_city['t'].max(dim='ens_member'),
                     facecolor='gray', alpha=0.3)

    # ax2 = ax.twinx()
    # ax2.plot(time, dset_city['tp'].T,'-', lw=1)
    # ax2.set_ylabel("Precipitation [mm]")
    # ax2.set_ylim(0, 50)
    
    plt.title("GEFS forecast for "+city+" | Run "+(time[0]-np.timedelta64(6,'h')).strftime('%Y%m%d %H UTC'))
    fig.autofmt_xdate()
    
    if debug:
        plt.show(block=True)
    else:
        plt.savefig(folder_images+"spaghetti_"+city, dpi=100, bbox_inches='tight') 

    plt.clf()
