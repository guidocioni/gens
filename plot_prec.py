import xarray as xr
from matplotlib.colors import from_levels_and_colors
import sys
from functools import partial
from multiprocessing import Pool
import pandas as pd
import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
import matplotlib.pyplot as plt

from utils import *
debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')


# The one employed for the figure name when exported
variable_name = 'prob_snow'

print('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    print('Projection not defined, falling back to default (euratl, nh)')
    projections = ['euratl', 'nh']
else:
    projections = sys.argv[1:]


def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset = xr.open_mfdataset(
        input_files, concat_dim='ens_member', combine='nested')\
            .squeeze()\
            .chunk({'ens_member':1, 'time':1,'lat':180,'lon':360})
    dset = dset.metpy.parse_cf()

    prob_snow = (dset['csnow'].mean(axis=0) * 100.).compute()
    prob_snow = np.ma.masked_less_equal(prob_snow, 5.)
    prob_rain = (dset['crain'].mean(axis=0) * 100.).compute()
    prob_rain = np.ma.masked_less_equal(prob_rain, 5.)

    lon, lat = get_coordinates(dset)
    lon2d, lat2d = np.meshgrid(lon, lat)

    time = pd.to_datetime(dset.time.values)
    cum_hour = np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

    levels = np.linspace(30, 100, 8)

    cmap_snow, norm_snow = get_colormap_norm("snow", levels)
    cmap_rain, norm_rain = get_colormap_norm("rain", levels)

    for projection in projections:  # This works regardless if projections is either single value or array
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.gca()
        m, x, y = get_projection(lon2d, lat2d, projection, labels=True)
        img = m.shadedrelief(scale=0.4, alpha=0.8)
        img.set_alpha(0.8)

        # All the arguments that need to be passed to the plotting function
        args = dict(m=m, x=x, y=y, ax=ax, cmap_snow=cmap_snow, norm_snow=norm_snow,
                    cmap_rain=cmap_rain, norm_rain=norm_rain,
                    prob_snow=prob_snow, prob_rain=prob_rain, levels=levels,
                    time=time, projection=projection, cum_hour=cum_hour)

        print('Pre-processing finished, launching plotting scripts')
        if debug:
            plot_files(time[1:2], **args)
        else:
            # Parallelize the plotting by dividing into chunks and processes
            dates = chunks(time, chunks_size)
            plot_files_param = partial(plot_files, **args)
            p = Pool(processes)
            p.map(plot_files_param, dates)


def plot_files(dates, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for date in dates:
        # Find index in the original array to subset when plotting
        i = np.argmin(np.abs(date - args['time']))
        # Build the name of the output image
        filename = subfolder_images[args['projection']] + \
            '/'+variable_name+'_%s.png' % args['cum_hour'][i]

        cs_rain = args['ax'].contourf(args['x'], args['y'], args['prob_rain'][i], extend='max', cmap=args['cmap_rain'],
                                      norm=args['norm_rain'], levels=args['levels'], alpha=0.8)
        cs_snow = args['ax'].contourf(args['x'], args['y'], args['prob_snow'][i], extend='max', cmap=args['cmap_snow'],
                                      norm=args['norm_snow'], levels=args['levels'], alpha=0.8)

        an_fc = annotation_forecast(args['ax'], args['time'][i])
        an_var = annotation(
            args['ax'], 'Snow probability (ens. mean)', loc='lower left', fontsize=7)
        an_run = annotation_run(args['ax'], args['time'])

        if first:
            x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size = 0.15, 0.05, 0.3, 0.02
            x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.05, 0.3, 0.02

            ax_cbar = plt.gcf().add_axes(
                [x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size])
            ax_cbar_2 = plt.gcf().add_axes(
                [x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size])
            cbar_snow = plt.gcf().colorbar(cs_snow, cax=ax_cbar, orientation='horizontal',
                                           label='Snow')
            cbar_rain = plt.gcf().colorbar(cs_rain, cax=ax_cbar_2, orientation='horizontal',
                                           label='Rain')
            cbar_snow.ax.tick_params(labelsize=8)
            cbar_rain.ax.tick_params(labelsize=8)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs_rain, cs_snow, an_fc, an_var, an_run])

        first = False


if __name__ == "__main__":
    main()
