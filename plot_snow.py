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

snow=merged['csnow']
time = pd.to_datetime(merged['time'].values)

cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

lon2d, lat2d = np.meshgrid(merged['lon'], merged['lat'])

prob_snow=snow.mean(axis=0)*100.
prob_snow=np.ma.masked_less_equal(prob_snow, 5)

# Truncate colormap
cmap = plt.get_cmap('gist_stern_r')
new_cmap = utils.truncate_colormap(cmap, 0, 0.9)

# Euro-Atlantic plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="euroatlantic", labels=True)
m.shadedrelief(scale=0.4, alpha=0.8)

first = True 
for i, date in enumerate(time):
    cs = m.contourf(lon2d, lat2d, prob_snow[i,:,:], levels=np.linspace(0,100,11),
                cmap=new_cmap, latlon=True)

    plt.title('Snow probability (ensemble mean) | '+date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    utils.annotation(plt.gca(), text='www.guidocioni.it', loc='lower right')
    
    if first: # Apparently it only needs to be added once...
        plt.colorbar(cs, orientation='horizontal', label='Probability [%]',fraction=0.046, pad=0.04)
    plt.savefig(diri_images+'euratl/prob_snow_%s.png' % cum_hour[i],
                dpi=utils.dpi_resolution, bbox_inches='tight')
        # This is needed to have contour which not overlap
    for coll in cs.collections: 
        plt.gca().collections.remove(coll)
    first=False

plt.close('all')
       
# Northern-Hemisphere plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="nh", labels=False)
m.shadedrelief(scale=0.4, alpha=0.8)

first = True 
for i, date in enumerate(time):
    cs = m.contourf(lon2d, lat2d, prob_snow[i,:,:], levels=np.linspace(0,100,11),
                cmap=new_cmap, latlon=True)

    plt.title('Snow probability (ensemble mean) | '+date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    utils.annotation(plt.gca(), text='www.guidocioni.it', loc='lower right')
    
    if first: # Apparently it only needs to be added once...
        plt.colorbar(cs, orientation='horizontal', label='Probability [%]',fraction=0.046, pad=0.04)
    plt.savefig(diri_images+'nh/prob_snow_%s.png' % cum_hour[i],
                dpi=utils.dpi_resolution, bbox_inches='tight')
        # This is needed to have contour which not overlap
    for coll in cs.collections: 
        plt.gca().collections.remove(coll)
    first=False

plt.close('all')
