debug = False 
if not debug:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from utils import *
import sys
import matplotlib.gridspec as gridspec

dset = xr.open_mfdataset(input_files, concat_dim='ens_member', combine='nested').squeeze()
dset = dset.metpy.parse_cf()
time = pd.to_datetime(dset['time'].values)
# Array needed for the box plot
pos = np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

myDates= ["2019-03-23 00:00", "2019-03-26 00:00", "2019-03-29 00:00"]
cities=['Milano','Roma','Palermo']
var_to_plot='2t'

nrows=np.shape(myDates)[0]
rows=np.arange(start=0, stop=nrows, step=1)
gridspec.GridSpec(nrows,1)

for city in cities:
    lon, lat = get_city_coordinates(city)
    dset_city =  dset.sel(lon=lon, lat=lat, method='nearest').interpolate_na(dim='time')
    # Recover units which somehow are deleted by interpolate_na,
    # no idea why....
    dset_city['t'].attrs['units'] = 'K'
    dset_city['t'].metpy.convert_units('degC')
    dset_city['2t'].attrs['units'] = 'K'
    dset_city['2t'].metpy.convert_units('degC')

    fig = plt.figure(1, figsize=(10, 6))

    sns.set(style="white", palette="muted", color_codes=True)
    sns.set_context(rc={"lines.linewidth": 0.5})
    plt.rc_context({'axes.edgecolor':'grey', 'xtick.color':'grey', 'ytick.color':'grey'})
    plt.suptitle("GEFS PDF forecast for "+city+" | Run "+(time[0]-np.timedelta64(6,'h')).strftime('%Y%m%d %H UTC'))

    for row in rows:
        ds = dset_city[var_to_plot].sel(time=myDates[row], method='nearest')
        if row == rows[0]:
            ax1=plt.subplot2grid((nrows,1), (row,0))
        else: 
            plt.subplot2grid((nrows,1), (row,0), sharex=ax1)
        
        ax2 = sns.distplot(ds, hist=False, kde_kws={"shade":True})
        
        ax3 = ax2.twinx()
        ax3.get_xaxis().set_visible(False)
        sns.boxplot(x=ds, ax=ax3)
        sns.despine()
        ax3.set(ylim=(-4, 6))
        at = AnchoredText(myDates[row], prop=dict(size=10), frameon=True,loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        plt.gca().add_artist(at)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
        if row != rows[-1]:
            ax2.get_xaxis().set_visible(False)
        else: 
            ax2.get_xaxis().set_visible(True)
            ax2.set_xlabel('Temperatura a 2 metri [C]')

    if debug:
        plt.show(block=True)
    else:
        plt.savefig('gefs_pdf_%s.png' % city, dpi=100, bbox_inches='tight') 