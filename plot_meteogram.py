from netCDF4 import Dataset
import netCDF4
import matplotlib.pyplot as plt
import numpy as np # Import the Numpy package
import matplotlib.gridspec as gridspec
from glob import glob
from geopy.geocoders import Nominatim
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import  DateFormatter
import xarray as xr

plt.switch_backend('agg')

diri='/scratch/local1/m300382/gens/grib/'
diri_images='/scratch/local1/m300382/gens/'
cities = ["Milano","Roma","Palermo","Hamburg","Pisa","Storrs"]

n_pert=[]

fileslist=sorted(glob(diri+"*.nc"))
datasets = [xr.open_dataset(files) for files in fileslist]
# Merging should take care automatically of solving every conflict in the dimensions
merged = xr.concat(datasets, 'ens_member')

# Need to find a way to do this slicing using dimension names...
t_2m=merged['2t'][:,:,0,:,:]-273.15
t_850hpa=merged['t'][:,:,0,:,:]-273.15
tot_prec=merged['tp'][:]
snow=merged['csnow'][:]
wind_speed_10m=(merged['10u'][:,:,0,:,:]**2+merged['10v'][:,:,0,:,:]**2)**(0.5)

n_pert=merged['ens_member'].values

wind_speed_10m=3.6*wind_speed_10m
lon=np.where(merged['lon'][:] >=180,merged['lon'][:]-360, merged['lon'][:] )
lat=merged['lat'].values
dtime = merged['time'].values

t_2m_point={}
t_850hpa_point={}
tot_prec_point={}
snow_point={}
wind_speed_10m_point={}
geolocator = Nominatim()
for city in cities:
    loc = geolocator.geocode(city)
    ilat=np.argmin(abs(lat-loc.latitude))
    ilon=np.argmin(abs(lon-loc.longitude))

    t_2m_point[city] = t_2m[:,:,ilat,ilon].interpolate_na(dim='time')
    t_850hpa_point[city] = t_850hpa[:,:,ilat,ilon].interpolate_na(dim='time')
    tot_prec_point[city] = tot_prec[:,:,ilat,ilon].interpolate_na(dim='time')
    snow_point[city] = snow[:,:,ilat,ilon].interpolate_na(dim='time')
    wind_speed_10m_point[city] = wind_speed_10m[:,:,ilat,ilon].interpolate_na(dim='time')

nrows=4
ncols=1
sns.set(style="white")

for city_to_plot in cities:
    loc = geolocator.geocode(city_to_plot)
    ilat=np.argmin(abs(lat-loc.latitude))
    ilon=np.argmin(abs(lon-loc.longitude))
    
    time=pd.to_datetime(dtime)
    pos = np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    fig = plt.figure(1, figsize=(9,10))
    ax1=plt.subplot2grid((nrows,ncols), (0,0))
    ax1.set_title("GEFS meteogram for "+city_to_plot+" | Run "+fileslist[0][fileslist[0].find('_2')+1:fileslist[0].find('_00')])
    
    bplot=ax1.boxplot(t_2m_point[city_to_plot].T, patch_artist=True,
                      showfliers=False, positions=pos, widths=3)
    for box in bplot['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')

    ax1.plot(pos, np.mean(t_2m_point[city_to_plot], axis=0), 'r-', linewidth=1)
    ax1.set_ylabel("2m Temp. [C]",fontsize=8)
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True, color='gray', linewidth=0.2)
    ax1.tick_params(axis='y', which='major', labelsize=8)
    ax1.tick_params(axis='x', which='both', bottom=False)

    ax2=plt.subplot2grid((nrows,ncols), (1,0))
    bplot_rain=ax2.boxplot(tot_prec_point[city_to_plot].T, patch_artist=True,
                      showfliers=False, positions=pos, widths=3)
    for box in bplot_rain['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')
        
    ax2.plot(pos, np.mean(tot_prec_point[city_to_plot], axis=0), 'r-', linewidth=1)
    ax2.set_ylim(bottom=0)
    ax2b = ax2.twinx()
    ax2b.plot(pos, np.mean(snow_point[city_to_plot]*100, axis=0), '*',color='purple')
    ax2b.set_ylabel("Snow probability",fontsize=8)
    ax2b.set_ylim(10, 100)
    ax2.yaxis.grid(True)
    ax2.set_ylabel("Precipitation [mm]",fontsize=8)
    ax2.xaxis.grid(True, color='gray', linewidth=0.2)
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax2b.tick_params(axis='y', which='major', labelsize=8)

    ax3=plt.subplot2grid((nrows,ncols), (2,0))
    bplot_wind=ax3.boxplot(wind_speed_10m_point[city_to_plot].T, patch_artist=True,
                      showfliers=False, positions=pos, widths=3)
    for box in bplot_wind['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')
    ax3.plot(pos, np.mean(wind_speed_10m_point[city_to_plot], axis=0), 'r-', linewidth=1)

    ax3.yaxis.grid(True)
    ax3.set_ylabel("Wind speed [km/h]",fontsize=8)
    ax3.tick_params(axis='y', which='major', labelsize=8)
    ax3.set_ylim(bottom=0)
    ax3.xaxis.grid(True, color='gray', linewidth=0.2)

    ax4=plt.subplot2grid((nrows,ncols), (3,0))
    ax4.plot_date(time, t_850hpa_point[city_to_plot][:,:].T, '-',linewidth=0.8)
    ax4.set_xlim(dtime[0],dtime[-1])
    ax4.set_ylabel("850 hPa Temp. [C]",fontsize=8)
    ax4.tick_params(axis='y', which='major', labelsize=8)
    ax4.yaxis.grid(True)
    ax4.xaxis.grid(True)
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax4.xaxis.set_major_formatter(DateFormatter('%d %b %Y'))

    ax4.annotate('Grid point %3.1fN %3.1fE' % (lat[ilat], lon[ilon]), xy=(0.7, -0.7), xycoords='axes fraction', color="gray")

    fig.subplots_adjust(hspace=0.1)
    fig.autofmt_xdate()
    plt.savefig(diri_images+"meteogram_"+city_to_plot, dpi=100, bbox_inches='tight')
#     plt.show()
    plt.clf()
