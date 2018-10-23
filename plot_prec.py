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

tot_prec=merged['tp']
time = pd.to_datetime(merged['time'].values)

cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

lon2d, lat2d = np.meshgrid(merged['lon'], merged['lat'])

# Truncate colormap
cmap = plt.get_cmap('gist_stern_r')
new_cmap = utils.truncate_colormap(cmap, 0, 0.9)

levs_prec = np.linspace(0, 100, 51)

# Probabilites plot 
thresholds = [50.]

# Euro-Atlantic plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="euroatlantic", labels=True)

first = True 
for threshold in thresholds:
    probabilities = (np.sum(tot_prec > threshold, axis=0)/float(tot_prec.shape[0]))*100.
    
    for i, date in enumerate(time):
        cs = m.contourf(lon2d, lat2d, tot_prec[0,i,:,:], levels=levs_prec,
                        cmap=new_cmap, extend="both", latlon=True)
        c = m.contour(lon2d, lat2d, probabilities[i,:,:], np.linspace(0,100,5), latlon=True)

        labels=plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)
        plt.title('CTRL prec. & Probability tot. prec. > '+str(int(threshold))+' mm | '+date.strftime('%d %b %Y at %H UTC'))
        utils.annotation_run(plt.gca(), time)
        utils.annotation(plt.gca(), text='GEFS', loc='upper left')
        if first: # Apparently it only needs to be added once...
            plt.colorbar(cs, orientation='horizontal', label='Probability [%]',fraction=0.046, pad=0.04)
        plt.savefig(diri_images+'euratl/prob_prec_%s_%s.png' % (int(threshold), cum_hour[i]),
                    dpi=utils.dpi_resolution, bbox_inches='tight')
            # This is needed to have contour which not overlap
        for coll in c.collections: 
            plt.gca().collections.remove(coll)
        for label in labels:
            label.remove()
        first=False

plt.close('all')
       
# Northern-Hemisphere plots
fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="nh", labels=False)

first = True 
for threshold in thresholds:
    probabilities = (np.sum(tot_prec > threshold, axis=0)/float(tot_prec.shape[0]))*100.
    
    for i, date in enumerate(time):
        cs = m.contourf(lon2d, lat2d, tot_prec[0,i,:,:], levels=levs_prec,
                        cmap=new_cmap, extend="both", latlon=True)
        c = m.contour(lon2d, lat2d, probabilities[i,:,:], np.linspace(0,100,5), latlon=True)

        labels=plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)
        plt.title('CTRL prec. & Probability tot. prec. > '+str(int(threshold))+' mm | '+date.strftime('%d %b %Y at %H UTC'))
        utils.annotation_run(plt.gca(), time)
        utils.annotation(plt.gca(), text='GEFS', loc='upper left')
        if first: # Apparently it only needs to be added once...
            plt.colorbar(cs, orientation='horizontal', label='Probability [%]',fraction=0.046, pad=0.04)
        plt.savefig(diri_images+'nh/prob_prec_%s_%s.png' % (int(threshold), cum_hour[i]),
                    dpi=utils.dpi_resolution, bbox_inches='tight')
            # This is needed to have contour which not overlap
        for coll in c.collections: 
            plt.gca().collections.remove(coll)
        for label in labels:
            label.remove()
        first=False
