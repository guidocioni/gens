import numpy as np
import pandas as pd
import xarray as xr
import datetime
from glob import glob

def wind_power(wind, a, b, c):
    return a/(1+np.exp(-b*(wind+c)))

diri='/scratch/local1/m300382/gens/grib/'
fileslist=sorted(glob(diri+"*.nc"))

# merged = xr.open_mfdataset(fileslist, concat_dim='ens_member').squeeze().interpolate_na(dim='time')

datasets = [xr.open_dataset(files) for files in fileslist]
merged = xr.concat(datasets, 'ens_member').squeeze()

wind = (merged['10u']**2+merged['10v']**2)**(0.5)
temperature = merged['2t'] - 273.15

# Using the parameters obtained by the fit
power = wind_power(wind, 2500.,  0.85, -7.5)

# outdset = xr.merge([wind.to_dataset(name='Wind Power'), temperature.to_dataset(name='2m Temperature')])

outdset = xr.Dataset({
         'wind_power': (['ens_member', 'time', 'lat', 'lon'],  power.values, {'units' : 'MW'} ),
         '2m_temperature':(['ens_member', 'time', 'lat', 'lon'],  temperature.values, {'units' : 'C'} )
         },
         coords={'time': merged.time, 'ens_member': np.arange(1,merged.dims['ens_member']+1,1) , 'lat':merged.lat.values, 'lon':merged.lon.values},
         attrs={'creation date': datetime.datetime.now().strftime("%d %b %Y at %H:%M"),
                'author' : 'Guido Cioni (guido.cioni@mpimet.mpg.de)',
                'description' : 'Wind power prediction'})

outdset.to_netcdf('energy_sources_gfs.nc')