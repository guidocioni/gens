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
gph_500=merged['gh'][:,:,0,:,:]
time = pd.to_datetime(merged['time'].values)

cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

lon2d, lat2d = np.meshgrid(merged['lon'], merged['lat'])

# Compute arrays to plot just once

gph_500_std=gph_500.std(axis=0)
gph_500_mean=gph_500.mean(axis=0)
gph_500_std=np.ma.masked_less_equal(gph_500_std, 20)

# Truncate colormap
cmap = plt.get_cmap('Greys')
new_cmap = utils.truncate_colormap(cmap, 0.1, 0.9)

levels=(4600., 5000., 5200., 5400., 5600., 5700., 5800.)
levels_std=np.linspace(0, round(gph_500_std.max()), 16)

# Euro-Atlantic plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="euroatlantic", labels=True)
m.shadedrelief(scale=0.4, alpha=0.8)

first = True
for i, date in enumerate(time):
    c = m.contour(lon2d, lat2d, gph_500_mean[i,:,:], extend='both', levels=levels, latlon=True, cmap='plasma')
    cs = m.contourf(lon2d, lat2d, gph_500_std[i,:,:], extend='both', levels=levels_std,
                    cmap=new_cmap, latlon=True)
    
    labels=plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)
    plt.title('GEFS forecast for %s' % date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    utils.annotation(plt.gca(), text='www.guidocioni.it', loc='lower right')
    
    if first: # Apparently it only needs to be added once...
        plt.colorbar(cs, orientation='horizontal', label='Standard deviation [gpm]', pad=0.05, fraction=0.05)
    plt.savefig(diri_images+'euratl/gph_500_%s.png' % cum_hour[i],
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
    c = m.contour(lon2d, lat2d, gph_500_mean[i,:,:], extend='both', levels=levels, latlon=True, cmap='plasma')
    cs = m.contourf(lon2d, lat2d, gph_500_std[i,:,:], extend='both', levels=levels_std,
                    cmap=new_cmap, latlon=True)
    
    labels=plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)
    plt.title('GEFS forecast for %s' % date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    utils.annotation(plt.gca(), text='www.guidocioni.it', loc='lower right')
    
    if first: # Apparently it only needs to be added once...
        plt.colorbar(cs, orientation='horizontal', label='Standard deviation [gpm]', pad=0.05, fraction=0.05)
    plt.savefig(diri_images+'nh/gph_500_%s.png' % cum_hour[i],
                dpi=utils.dpi_resolution, bbox_inches='tight')
    # This is needed to have contour which not overlap
    for coll in c.collections: 
        plt.gca().collections.remove(coll)
    for coll in cs.collections: 
        plt.gca().collections.remove(coll)
    for label in labels:
        label.remove()
    first=False

