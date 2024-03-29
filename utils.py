# Configuration file for some common variables to all script 
from mpl_toolkits.basemap import Basemap  # import Basemap matplotlib toolkit
import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as colors
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd
from matplotlib.colors import from_levels_and_colors
import seaborn as sns
import requests
import json
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

apiKey = os.environ['MAPBOX_KEY']
apiURL_places = "https://api.mapbox.com/geocoding/v5/mapbox.places"

if "HOME_FOLDER" in os.environ:
    home_folder = os.environ['HOME_FOLDER']
else:
    home_folder = os.path.dirname(os.path.realpath(__file__))

# Output folder for images
if 'MODEL_DATA_FOLDER' in os.environ:
    folder = os.environ['MODEL_DATA_FOLDER']
else:
    folder = '/home/ekman/ssd/guido/gens/'

folder_images = folder
input_files=folder+'grib_gefs_*.nc' 
chunks_size = 10 
processes = 4
figsize_x = 10 
figsize_y = 8

# Options for savefig
options_savefig={
    'dpi':100,
    'bbox_inches':'tight',
    'transparent':True
}

# Dictionary to map the output folder based on the projection employed
subfolder_images={
    'euratl' : folder_images+'euratl',
    'nh' : folder_images+'nh'   
}
# Number of ensemble members
n_members_ensemble = 20

def get_coordinates(dataset):
    """Get the lat/lon coordinates from the dataset and convert them to degrees.
    TODO. Implement a check for coordinates to make sure that they are defined
    between -180 and 180, and not 0 to 360."""
    # We have to return an array otherwise Basemap 
    # will complain
    return(dataset['lon'].values, dataset['lat'].values)


def get_city_coordinates(city):
    # First read the local cache and see if we already downloaded the city coordinates
    if os.path.isfile(home_folder + '/cities_coordinates.csv'):
        cities_coords = pd.read_csv(home_folder + '/cities_coordinates.csv',
                                    index_col=[0])
        if city in cities_coords.index:
            return cities_coords.loc[city].lon, cities_coords.loc[city].lat
        else:
            # make the request and append to the file
            url = "%s/%s.json?&access_token=%s" % (apiURL_places, city, apiKey)
            response = requests.get(url)
            json_data = json.loads(response.text)
            lon, lat = json_data['features'][0]['center']
            to_append = pd.DataFrame(index=[city],
                                     data={'lon': lon, 'lat': lat})
            to_append.to_csv(home_folder + '/cities_coordinates.csv',
                             mode='a', header=False)

            return lon, lat
    else:
        # Make request and create the file for the first time
        url = "%s/%s.json?&access_token=%s" % (apiURL_places, city, apiKey)
        response = requests.get(url)
        json_data = json.loads(response.text)
        lon, lat = json_data['features'][0]['center']
        cities_coords = pd.DataFrame(index=[city],
                                     data={'lon': lon, 'lat': lat})
        cities_coords.to_csv(home_folder + '/cities_coordinates.csv')

        return lon, lat


def get_projection(lon, lat, projection="euratl", countries=True, labels=True):
    """Create the projection in Basemap and returns the x, y array to use it in a plot"""
    if projection=="euratl":
        m = Basemap(projection='mill', llcrnrlon=-50, llcrnrlat=30, urcrnrlon=30, urcrnrlat=70,resolution='i')
        if labels:
            m.drawparallels(np.arange(-90.0, 90.0, 10.), linewidth=0.2, color='white',
                labels=[True, False, False, True], fontsize=7)
            m.drawmeridians(np.arange(0.0, 360.0, 10.), linewidth=0.2, color='white',
                labels=[True, False, False, True], fontsize=7)
            
    elif projection=="nh":
        m = Basemap(projection="npstere", lon_0=20, lat_0=90, boundinglat=30,\
               rsphere=6378160.0, ellps="WGS84",k_0=0.9330127018922193)
        if labels:
            m.drawparallels(np.arange(-90.0, 90.0, 10.), linewidth=0.2, color='white',
                labels=[True, False, False, True], fontsize=7)
            m.drawmeridians(np.arange(0.0, 360.0, 10.), linewidth=0.2, color='white',
                labels=[True, False, False, True], fontsize=7)

    m.drawcoastlines(linewidth=0.5, linestyle='solid', color='black', zorder=5)
    if countries:
        m.drawcountries(linewidth=0.5, linestyle='solid', color='black', zorder=5)

    x, y = m(lon,lat)
    return(m, x, y)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Annotation run, models 
def annotation_run(ax, time, loc='upper right'):
    at = AnchoredText('Run %s'% (time[0]-np.timedelta64(6,'h')).strftime('%Y%m%d %H UTC'), 
                      prop=dict(size=8), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    ax.add_artist(at)
    return(at)

def annotation_forecast(ax, time, loc='upper left',fontsize=8):
    """Put annotation of the forecast time."""
    at = AnchoredText('Forecast for %s' % time.strftime('%A %d %b %Y at %H UTC'), 
                       prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    ax.add_artist(at)
    return(at)

def annotation(ax, text, loc='upper right',fontsize=8):
    """Put a general annotation in the plot."""
    at = AnchoredText('%s'% text, prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    ax.add_artist(at)
    return(at)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_colormap(cmap_type):
    """Create a custom colormap."""
    if cmap_type == "winds":
      colors_tuple = pd.read_csv('/home/mpim/m300382/gens/cmap_winds.rgba').values 
    elif cmap_type == "temp":
      colors_tuple = pd.read_csv('/home/mpim/m300382/gens/cmap_temp.rgba').values
         
    cmap = colors.LinearSegmentedColormap.from_list(cmap_type, colors_tuple, colors_tuple.shape[0])
    return(cmap)

def get_colormap_norm(cmap_type, levels):
    """Create a custom colormap."""
    if cmap_type == "rain":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette("Blues", n_colors=len(levels)),
                                                    extend='max')
    elif cmap_type == "snow":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette("PuRd", n_colors=len(levels)),
                                                    extend='max')
    elif cmap_type == "snow_discrete":    
        colors = ["#DBF069","#5AE463","#E3BE45","#65F8CA","#32B8EB",
                    "#1D64DE","#E97BE4","#F4F476","#E78340","#D73782","#702072"]
        cmap, norm = from_levels_and_colors(levels, colors, extend='max')
    elif cmap_type == "rain_acc":    
        cmap, norm = from_levels_and_colors(levels, sns.color_palette('gist_stern_r', n_colors=len(levels)),
                         extend='max')

    return(cmap, norm)

def remove_collections(elements):
    """Remove the collections of an artist to clear the plot without
    touching the background, which can then be used afterwards."""
    for element in elements:
        try:
            for coll in element.collections: 
                coll.remove()
        except AttributeError:
            try:
                for coll in element:
                    coll.remove()
            except ValueError:
                print('WARNING: Collection is empty')
            except TypeError:
                element.remove() 
        except ValueError:
            print('WARNING: Collection is empty')


def divide_axis_for_cbar(ax, width="45%", height="2%", pad=-3, adjust=0.05):
    '''Using inset_axes, divides axis in two to place the colorbars side to side.
    Note that we use the bbox explicitlly with padding to adjust the position of the colorbars
    otherwise they'll come out of the axis (don't really know why)'''
    ax_cbar = inset_axes(ax,
                         width=width,
                         height=height,
                         loc='lower left',
                         borderpad=pad,
                         bbox_to_anchor=(adjust, 0., 1, 1),
                         bbox_transform=ax.transAxes
                         )
    ax_cbar_2 = inset_axes(ax,
                           width=width,
                           height=height,
                           loc='lower right',
                           borderpad=pad,
                           bbox_to_anchor=(-adjust, 0., 1, 1),
                           bbox_transform=ax.transAxes
                           )

    return ax_cbar, ax_cbar_2
