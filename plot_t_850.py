import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # Import the Basemap toolkit
import numpy as np # Import the Numpy package
from datetime import datetime
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from glob import glob
import xarray as xr
import utils
import pandas as pd

diri='/scratch/local1/m300382/gens/grib/'
diri_images='/scratch/local1/m300382/gens/'

fileslist=sorted(glob(diri+"*.nc"))
datasets = [xr.open_dataset(files) for files in fileslist]
# Merging should take care automatically of solving every conflict in the dimensions
merged = xr.concat(datasets, 'ens_member')
t_850hpa=merged['t'][:,:,0,:,:]-273.15
time = pd.to_datetime(merged['time'].values)

cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

lon2d, lat2d = np.meshgrid(merged['lon'], merged['lat'])

# Compute arrays to plot just once

t_850_std=t_850hpa.std(axis=0)
t_850_mean=t_850hpa.mean(axis=0)
t_850_std=np.ma.masked_less_equal(t_850_std, 1)

# Truncate colormap
cmap = plt.get_cmap('gist_stern_r')
new_cmap = utils.truncate_colormap(cmap, 0., 0.9)

levels=(-15, -10, -5, 0, 5, 10)
# levels_std=np.linspace(0, round(t_850hpa.std(axis=0).max()), 16)
levels_std=(0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 15)

# Euro-Atlantic plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="euroatlantic", labels=True)
m.shadedrelief(scale=0.4, alpha=0.8)

first = True
for i, date in enumerate(time):
    c = m.contour(lon2d, lat2d, t_850_mean[i,:,:], extend='both', levels=levels, latlon=True, 
        colors='black', linestyles='solid')
    cs = m.contourf(lon2d, lat2d,t_850_std[i,:,:], extend='both', levels=levels_std,
                    cmap=new_cmap, latlon=True)
    
    labels=plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)
    plt.title('GEFS forecast for %s' % date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    utils.annotation(plt.gca(), text='www.guidocioni.it', loc='lower right')
    
    if first: # Apparently it only needs to be added once...
        plt.colorbar(cs, orientation='horizontal', label='Standard deviation [C]', pad=0.03, fraction=0.04)
    plt.savefig(diri_images+'euratl/t_850_%s.png' % cum_hour[i],
                dpi=utils.dpi_resolution, bbox_inches='tight')
    # This is needed to have contour which not overlap
    for coll in c.collections: 
        plt.gca().collections.remove(coll)
    for coll in cs.collections: 
        plt.gca().collections.remove(coll)
    for label in labels:
        label.remove()
    first=False
       
# Northern-Hemisphere plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="nh", labels=False)
m.shadedrelief(scale=0.3, alpha=0.8)

first = True
for i, date in enumerate(time):
    c = m.contour(lon2d, lat2d, t_850_mean[i,:,:], extend='both', levels=levels, latlon=True,
                  colors='black', linestyles='solid')
    cs = m.contourf(lon2d, lat2d,t_850_std[i,:,:], extend='both', levels=levels_std,
                    cmap=new_cmap, latlon=True)
    
    labels=plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)
    plt.title('GEFS forecast for %s' % date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    utils.annotation(plt.gca(), text='www.guidocioni.it', loc='lower right')
    
    if first: # Apparently it only needs to be added once...
        plt.colorbar(cs, orientation='horizontal', label='Standard deviation [C]', pad=0.03, fraction=0.04)
    plt.savefig(diri_images+'nh/t_850_%s.png' % cum_hour[i],
                dpi=utils.dpi_resolution, bbox_inches='tight')
    # This is needed to have contour which not overlap
    for coll in c.collections: 
        plt.gca().collections.remove(coll)
    for coll in cs.collections: 
        plt.gca().collections.remove(coll)
    for label in labels:
        label.remove()
    first=False