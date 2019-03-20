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

print('Starting script to plot meteograms')

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

nrows=4
ncols=1
sns.set(style="white")

for city in cities:
    print('Producing meteogram for %s' % city)
    lon, lat = get_city_coordinates(city)
    dset_city =  dset.sel(lon=lon, lat=lat, method='nearest').interpolate_na(dim='time')
    # Recover units which somehow are deleted by interpolate_na,
    # no idea why....
    dset_city['2t'].attrs['units'] = 'K'
    dset_city['t'].attrs['units'] = 'K'
    dset_city['10u'].attrs['units'] = 'm/s'
    dset_city['10v'].attrs['units'] = 'm/s'
    dset_city['2t'].metpy.convert_units('degC')
    dset_city['t'].metpy.convert_units('degC')
    wind_speed = mpcalc.wind_speed(dset_city['10u'],dset_city['10v']).to('kph')

    fig = plt.figure(1, figsize=(9,10))
    ax1=plt.subplot2grid((nrows,ncols), (0,0))
    ax1.set_title("GEFS meteogram for "+city+" | Run "+(time[0]-np.timedelta64(6,'h')).strftime('%Y%m%d %H UTC'))
    
    bplot=ax1.boxplot(dset_city['2t'].values, patch_artist=True,
                      showfliers=False, positions=pos, widths=3)
    for box in bplot['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')

    ax1.plot(pos, dset_city['2t'].mean(axis=0), 'r-', linewidth=1)
    ax1.set_ylabel("2m Temp. [C]",fontsize=8)
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True, color='gray', linewidth=0.2)
    ax1.tick_params(axis='y', which='major', labelsize=8)
    ax1.tick_params(axis='x', which='both', bottom=False)

    ax2=plt.subplot2grid((nrows,ncols), (1,0))
    bplot_rain=ax2.boxplot(dset_city['tp'].values, patch_artist=True,
                      showfliers=False, positions=pos, widths=3)
    for box in bplot_rain['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')
        
    ax2.plot(pos, dset_city['tp'].mean(axis=0), 'r-', linewidth=1)
    ax2.set_ylim(bottom=0)
    ax2b = ax2.twinx()
    ax2b.plot(pos, dset_city['csnow'].mean(axis=0)*100., '*',color='purple')
    ax2b.set_ylabel("Snow probability",fontsize=8)
    ax2b.set_ylim(10, 100)
    ax2.yaxis.grid(True)
    ax2.set_ylabel("Precipitation [mm]",fontsize=8)
    ax2.xaxis.grid(True, color='gray', linewidth=0.2)
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax2b.tick_params(axis='y', which='major', labelsize=8)

    ax3=plt.subplot2grid((nrows,ncols), (2,0))
    bplot_wind=ax3.boxplot(wind_speed.T, patch_artist=True,
                      showfliers=False, positions=pos, widths=3)
    for box in bplot_wind['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')
    ax3.plot(pos, np.mean(wind_speed, axis=0), 'r-', linewidth=1)

    ax3.yaxis.grid(True)
    ax3.set_ylabel("Wind speed [km/h]",fontsize=8)
    ax3.tick_params(axis='y', which='major', labelsize=8)
    ax3.set_ylim(bottom=0)
    ax3.xaxis.grid(True, color='gray', linewidth=0.2)

    ax4=plt.subplot2grid((nrows,ncols), (3,0))
    ax4.plot(time, dset_city['t'].values.T, '-',linewidth=0.8)
    ax4.set_xlim(time[0],time[-1])
    ax4.set_ylabel("850 hPa Temp. [C]",fontsize=8)
    ax4.tick_params(axis='y', which='major', labelsize=8)
    ax4.yaxis.grid(True)
    ax4.xaxis.grid(True)
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax4.xaxis.set_major_formatter(DateFormatter('%d %b %Y'))

    ax4.annotate('Grid point %3.1fN %3.1fE' % (dset_city.lat, dset_city.lon),
                         xy=(0.7, -0.7), xycoords='axes fraction', color="gray")

    fig.subplots_adjust(hspace=0.1)
    fig.autofmt_xdate()

    if debug:
        plt.show(block=True)
    else:
        plt.savefig(folder_images+"meteogram_"+city, dpi=100, bbox_inches='tight')   
    plt.clf()
