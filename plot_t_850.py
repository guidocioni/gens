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
dtime = pd.to_datetime(merged['time'].values)

lon2d, lat2d = np.meshgrid(merged['lon'], merged['lat'])

fig = plt.figure(figsize=(10,10))
m = utils.get_projection(projection="euroatlantic", labels=True)

levels=(-15, -10, -5, 0, 5, 10)

first = True
for i, date in enumerate(dtime):
    cs = m.contourf(lon2d, lat2d, t_850hpa.std(axis=0)[-1,:,:], extend='both', cmap='Greys', latlon=True)
    c = m.contour(lon2d, lat2d, t_850hpa.mean(axis=0)[-1,:,:], extend='both', levels=levels, latlon=True)
    
    plt.gca().clabel(c, c.levels, inline=True, fmt='%d' , fontsize=10)

    plt.title(' | '+date.strftime('%d %b %Y at %H UTC'))
    utils.annotation_run(plt.gca(), time)
    utils.annotation(plt.gca(), text='GEFS', loc='upper left')
    if first: # Apparently it only needs to be added once...
        plt.colorbar(orientation='horizontal', label='Standard deviation [C]', pad=0.1, fraction=0.05)
    plt.savefig(folder_images+'t_850_%s.png' % cum_hour[i],
                dpi=dpi_resolution, bbox_inches='tight')
    first=False