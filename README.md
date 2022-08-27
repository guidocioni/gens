# gens
Download and plot GENS (GFS Ensemble) data 

In the following repository I include a fully-functional suite of scripts needed to download and merge data from GENS.
The main script to be called (possibly through cronjob) is `get_grib.run`. There, the current run version is determined, and files are downloaded from the GFS server.
CDO is used to merge the files. At the end of the process one NETCDF file with all the selected variables is created for every ensemble member (including the control run).

Subsequently some plotting routines written in Python are called. For the moment only meteogram output is used. A python notebook is included to show how to read and process the data.

In the following some description of the files:

* `downloader.py` : `python` script needed to retrieve the files of the run
* `plot_gefs.ipynb` : Jupyter notebook that shows how to process and plot the data. 
* `plot_meteogram.py` : Script to plot meteograms. Include functionality to retrieve lat/lon coordinates using `nominatim`. 
* `plot_spaghetti.py` : Script to plot spaghetti.
