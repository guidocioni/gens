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

diri='/scratch/local1/m300382/gens/grib/'
diri_images='/scratch/local1/m300382/gens/'
cities = ["Frankfurt am Main"]

t_2m=[]
t_850hpa=[]
tot_prec=[]
snow=[]
u_10m=[]
v_10m=[]
n_pert=[]

for fname in glob(diri+"*.nc"):
    nc=Dataset(fname)
    t_2m.append(nc.variables['2t'][:][:,0,:,:]-273.15)
    t_850hpa.append(nc.variables['t'][:][:,0,:,:]-273.15)
    tot_prec.append(nc.variables['tp'][:])
    snow.append(nc.variables['csnow'][:])
    u_10m.append(nc.variables['10u'][:][:,0,:,:])
    v_10m.append(nc.variables['10v'][:][:,0,:,:])
    if (fname[fname.find(".nc")-2:fname.find(".nc")]) =="00": #this is the control run
        n_pert.append(0)
    else:
        n_pert.append(nc.variables['2t'].realization)

n_pert=np.array(n_pert)
t_2m=np.array(t_2m)
t_850hpa=np.array(t_850hpa)
tot_prec=np.array(tot_prec)
snow=np.array(snow)
wind_speed_10m=np.sqrt(np.array(u_10m)**2+np.array(v_10m)**2)*3.6
lon=np.where(nc.variables['lon'][:] >=180,nc.variables['lon'][:]-360, nc.variables['lon'][:] )
lat=nc.variables['lat'][:]
time_var = nc.variables['time']
dtime = netCDF4.num2date(time_var[:],time_var.units)

t_2m_point={}
t_850hpa_point={}
tot_prec_point={}
snow_point={}
wind_speed_10m_point={}
geolocator = Nominatim()
for city in cities:
    loc = geolocator.geocode(city)
    t_2m_point[city] = t_2m[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]
    t_850hpa_point[city] = t_850hpa[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]
    tot_prec_point[city] = tot_prec[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]
    snow_point[city] = snow[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]
    wind_speed_10m_point[city] = wind_speed_10m[:,:,np.argmin(abs(lat-loc.latitude)),np.argmin(abs(lon-loc.longitude))]

nrows=4
ncols=1
gridspec.GridSpec(nrows,ncols)
fig = plt.figure(1, figsize=(9,8))
sns.set(style="white")

for city_to_plot in cities:
    ax1=plt.subplot2grid((nrows,ncols), (0,0))
    ax1.set_title("GEFS meteogram for "+city_to_plot+" | Run "+dtime[0].strftime("%Y%m%d %H"))
    bplot=ax1.boxplot(t_2m_point[city_to_plot],patch_artist=True,showfliers=False)
    for box in bplot['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')

    xaxis=np.arange(1,np.shape(dtime)[0]+1,1)
    ax1.plot(xaxis, np.mean(t_2m_point[city_to_plot], axis=0), linewidth=1,color='red')
    ax1.set_ylabel("2m Temp. [C]",fontsize=8)
    ax1.yaxis.grid(True)
    ax1.tick_params(axis='y', which='major', labelsize=8)

    ax2=plt.subplot2grid((nrows,ncols), (1,0))
    bplot_rain=ax2.boxplot(tot_prec_point[city_to_plot],patch_artist=True,showfliers=False)
    for box in bplot_rain['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')
    ax2.plot(xaxis, np.mean(tot_prec_point[city_to_plot], axis=0), linewidth=1,color='red')
    ax2.set_ylim(bottom=0)
    ax2b = ax2.twinx()
    ax2b.plot(xaxis, np.mean(snow_point[city_to_plot]*100, axis=0), '*', linewidth=1,color='purple')
    ax2b.set_ylabel("Snow probability",fontsize=8)
    ax2b.set_ylim(10, 100)
    ax2.yaxis.grid(True)
    ax2.set_ylabel("Precipitation [mm]",fontsize=8)
    ax2.tick_params(axis='y', which='major', labelsize=8)
    ax2b.tick_params(axis='y', which='major', labelsize=8)

    ax3=plt.subplot2grid((nrows,ncols), (2,0))
    bplot_wind=ax3.boxplot(wind_speed_10m_point[city_to_plot],patch_artist=True,showfliers=False)
    for box in bplot_wind['boxes']:
        box.set(color='LightBlue')
        box.set(facecolor='LightBlue')
    ax3.plot(xaxis, np.mean(wind_speed_10m_point[city_to_plot], axis=0), linewidth=1,color='red')

    ax3.yaxis.grid(True)
    ax3.set_ylabel("Wind speed [km/h]",fontsize=8)
    ax3.tick_params(axis='y', which='major', labelsize=8)
    ax3.set_ylim(bottom=0)
    ax4=plt.subplot2grid((nrows,ncols), (3,0))
    ax4.plot_date(dtime, t_850hpa_point[city_to_plot][:,:].T, '-',linewidth=0.8)
    ax4.set_xlim(dtime[0],dtime[-1])
    ax4.set_ylabel("850 hPa Temp. [C]",fontsize=8)
    ax4.tick_params(axis='y', which='major', labelsize=8)
    ax4.yaxis.grid(True)
    ax4.xaxis.grid(True)
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax4.xaxis.set_major_formatter(DateFormatter('%d %b %Y'))

    fig.subplots_adjust(hspace=0.1)
    fig.autofmt_xdate()
    plt.show()
    plt.clf()
