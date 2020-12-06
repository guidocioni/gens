debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import xarray as xr 
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os 
from utils import *
import sys
from matplotlib.colors import from_levels_and_colors
import seaborn as sns

# The one employed for the figure name when exported 
variable_name = 't_850'

print('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can 
# span multiple instances of this script outside
if not sys.argv[1:]:
    print('Projection not defined, falling back to default (euratl, nh)')
    projections = ['euratl','nh']
else:    
    projections=sys.argv[1:]

def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset = xr.open_mfdataset(input_files, concat_dim='ens_member', combine='nested').squeeze()
    dset = dset.metpy.parse_cf()

    t_850 = dset['t'].load() - 273.15
    t_850_std=t_850.std(axis=0, skipna=True)
    t_850_mean=t_850.mean(axis=0, skipna=True)
    t_850_std=np.ma.masked_less_equal(t_850_std, 1)

    lon, lat = get_coordinates(dset)
    lon2d, lat2d = np.meshgrid(lon, lat)

    time = pd.to_datetime(dset.time.values)
    cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    levels_std=(0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 15)
    levels_temp=(-15, -10, -5, 0, 5, 10)

    cmap, norm = get_colormap_norm("rain_acc", levels_std)
    
    for projection in projections:# This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax  = plt.gca()        
        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)
        img=m.shadedrelief(scale=0.4, alpha=0.8)
        img.set_alpha(0.8)

        # All the arguments that need to be passed to the plotting function
        args=dict(m=m, x=x, y=y, ax=ax, cmap=cmap, norm=norm,
                 t_850_std=t_850_std, levels_std=levels_std, t_850_mean=t_850_mean,
                 levels_temp=levels_temp, time=time, projection=projection, cum_hour=cum_hour)
        
        print('Pre-processing finished, launching plotting scripts')
        if debug:
            plot_files(time[-2:-1], **args)
        else:
            # Parallelize the plotting by dividing into chunks and processes 
            dates = chunks(time, chunks_size)
            plot_files_param=partial(plot_files, **args)
            p = Pool(processes)
            p.map(plot_files_param, dates)

def plot_files(dates, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for date in dates:
        # Find index in the original array to subset when plotting
        i = np.argmin(np.abs(date - args['time'])) 
        # Build the name of the output image
        filename = subfolder_images[args['projection']]+'/'+variable_name+'_%s.png' % args['cum_hour'][i]

        cs = args['ax'].contourf(args['x'], args['y'], args['t_850_std'][i], extend='both', cmap=args['cmap'],
                                    norm=args['norm'], levels=args['levels_std'])
        
        c = args['ax'].contour(args['x'], args['y'], args['t_850_mean'][i], levels=args['levels_temp'],
                             colors='black', linewidths=1., linestyles='solid')

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f' , fontsize=7)

        an_fc = annotation_forecast(args['ax'],args['time'][i])
        an_var = annotation(args['ax'], '$T$ at 850 hPa (mean and std.)' ,loc='lower left', fontsize=7)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Standard deviation ',fraction=0.046, pad=0.04)
        
        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        
        
        remove_collections([cs, c, labels, an_fc, an_var, an_run])

        first = False 

if __name__ == "__main__":
    main()
