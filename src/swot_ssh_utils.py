#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for manipulating SWOT SSH data.

Author: Jinbo Wang
Date: First version: 01.22.2023

Dependencies:
    numpy
    scipy
    pylab
    xarray
    pyresample
    json
    s3fs
    requests
    h5netcdf
"""

import numpy as np
import scipy.signal as signal
import pylab as plt
import xarray as xr
import numpy as np
import requests
import json


def plot_a_segment(ax,lon,lat,dat,title='',
           vmin=-0.5,vmax=0.5):
    """
    Plot a segment of SWOT data on a map.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot the data on.
        lon (numpy.ndarray): The longitude data.
        lat (numpy.ndarray): The latitude data.
        dat (numpy.ndarray): The data to plot.
        title (str): Optional. The title of the plot.
        vmin (float): Optional. The minimum value of the colorbar.
        vmax (float): Optional. The maximum value of the colorbar.
        
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    proj = ccrs.PlateCarree()
    extent = [np.nanmin(lon),
                np.nanmax(lon),
                np.nanmin(lat),
                np.nanmax(lat)]
              
    # Add the scatter plot
    lon=lon.flatten()
    lat=lat.flatten()
    im=ax.scatter(lon, lat, c=dat, s=1, cmap=plt.cm.bwr,
               transform=ccrs.PlateCarree(),
              vmin=vmin,vmax=vmax )
    # Add the colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                        pad=0.02, aspect=40, shrink=0.8,
                        extend='both' , label='meter')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.RIVERS)
    gl=ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    ax.set_extent(extent, proj)
    ax.set_title(title)
    return

def list_podaac_data(print_on_screen=True, title_keyword_include=None,keyword_exclude=None):
    """
    Find a list all of the PO.DAAC datasets available in the Common Metadata Repository (CMR).

    Returns:
        list: A list of dictionaries containing dataset information. Aslo prints the list to the screen.

    """
    import requests
    import pandas as pd
    import json

    BASE_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"

    # Parameters for the CMR query
    params = {
        'provider': 'POCLOUD',
        'page_size': 2000,
        'page_num': 1,
    }
    
    # Send a request to CMR
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()  # Check for any response errors
    
    data = response.json()
    entries = data['feed']['entry']
    
    datasets = []

    for entry in entries:
        datasets.append({
            'concept-id': entry['id'],
            'short_name': entry.get('short_name', 'N/A'),
            'title': entry['title'],
            'abstract': entry['summary'],
        })
    df = pd.DataFrame(datasets)

    if title_keyword_include!=None:
        df = df[df['title'].str.contains(title_keyword_include, case=False, regex=False)]

    if keyword_exclude!=None:
        if type(keyword_exclude)==str:
            df = df[~df['title'].str.contains(keyword_exclude, case=False, regex=False)]
        if type(keyword_exclude)==list:
            for excl in keyword_exclude:
                df = df[~df['title'].str.contains(excl, case=False, regex=False)]

    if print_on_screen:
        print(df)
    return df

def init_S3FileSystem():
    """
    This function initializes an S3FileSystem object for accessing SWOT data on the PO.DAAC S3 server.
    It requires the .netrc file in the user's home directory to contain the following lines:
        machine urs.earthdata.nasa.gov
            login <uid>
            password <password>
    The <uid> and <password> are the Earthdata login credentials. They can be obtained by registering at https://urs.earthdata.nasa.gov if you don't have them already.

    Returns:
        s3fs.S3FileSystem: An S3FileSystem object for accessing SWOT data on the PO.DAAC S3 server.

        Example usage:
            s3sys = init_S3FileSystem()
            # filename on AWS S3. You need to be on an AWS cloud computer from the US-West-2 region to access the data.
            # You do not need this function if you are using non-AWS cloud computers.
            fn = 'podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_1.0/SWOT_L2_LR_SSH_Unsmoothed_406_013_20230120T185229_20230120T194335_PIA1_01.nc' 
            ds = xr.open_dataset(s3sys.open(fn, mode='rb'), engine='h5netcdf')

            # the following code will list all files in the directory
            file_list = s3sys.glob('podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_1.0/')
            
    """
    import requests,s3fs
    creds = requests.get('https://archive.swot.podaac.earthdata.nasa.gov/s3credentials').json()
    s3 = s3fs.S3FileSystem(anon=False,
                           key=creds['accessKeyId'],
                           secret=creds['secretAccessKey'], 
                           token=creds['sessionToken'])
    return s3

def get_s3_urls(short_name, bounding_box, time_range):
    """
    Search granule S3 filenames
    
    Parameters:
        short_name: str #unique identifier for a NASA dataset
        bounding_box: str #bounding box in the format of 'lon_min,lat_min,lon_max,lat_max'
        time_range: str #time range in the format of 'start_time,end_time', e.g., 2022-01-01T00:00:00Z,2022-01-31T23:59:59Z
    
    Returns:
        list: A list of S3 filenames matching the search criteria.

    Example usage:

    get_s3_urls("SENTINEL-1A", "-10,-10,10,10", "2022-01-01T00:00:00Z,2022-01-31T23:59:59Z")
    """
    
    base_url = "https://cmr.earthdata.nasa.gov/search/granules"

    params = {
        "short_name": short_name,
        "bounding_box": bounding_box,
        "temporal": time_range,
        "page_size": 2000,  # number of items per request, can be increased up to 2000
    }

    headers = {
        "Accept": "application/json",
    }

    response = requests.get(base_url, params=params, headers=headers)

    names=[]
    if response.status_code == 200:
        granules = response.json()['feed']['entry']
        for granule in granules:
            names.append(granule['title'])  # or any other field you are interested in
    else:
        print(f"Request failed, status code: {response.status_code}")

    return names


def get_mss22(longitude, latitude):
    """
    Interpolate Mean Sea Surface 2022 (MSS22) data onto the input longitude and latitude grid.

    Args:
        longitude (numpy.ndarray): The grid of longitudes.
        latitude (numpy.ndarray): The grid of latitudes.

    Returns:
        numpy.ndarray: The interpolated MSS22 data on the input grid.
    """

    from scipy.interpolate import RectBivariateSpline as rbs
    from scipy.interpolate import griddata
    from xarray import open_dataset
    import numpy as np

    # Calculate the min and max values for longitude and latitude
    lonmin = np.nanmin(longitude)
    lonmax = np.nanmax(longitude)
    latmin = np.nanmin(latitude)
    latmax = np.nanmax(latitude)

    # Create a mask for missing data in the input longitude and latitude grids
    m = np.isnan(longitude * latitude).flatten()

    # Open the MSS22 dataset
    mss = open_dataset('/mnt/flow/swot/MSS/MSS_2022/mss_cls22_updated.grd')

    # Select the MSS22 data within the range of input longitude and latitude
    mss_z = mss['z'].sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))

    # Check if there are any missing values in the selected MSS22 data
    if np.isnan(mss_z.data).sum() > 0:
        lonm, latm = np.meshgrid(mss_z.lon.data, mss_z.lat.data)
        lonm = lonm.flatten()
        latm = latm.flatten()

        # Mask for missing values in the selected MSS22 data
        mm = ~np.isnan(mss_z.data).flatten()

        # Interpolate the MSS22 data onto the input grid using griddata
        aa = griddata((latm[mm], lonm[mm]), mss_z.data.flatten()[mm], (latitude.flatten()[~m], longitude.flatten()[~m]))
    else:
        # Interpolate the MSS22 data onto the input grid using RectBivariateSpline
        aa = rbs(mss_z.lat, mss_z.lon, mss_z.data).ev(latitude.flatten()[~m], longitude.flatten()[~m])

    # Initialize an array for interpolated MSS22 data
    mss22 = np.ones((longitude.size))

    # Assign missing values and interpolated data to the MSS22 array
    mss22[m] = np.nan
    mss22[~m] = aa

    return mss22.reshape(longitude.shape)


class SSH_L2:
    """
    A class for handling Level 2 Sea Surface Height (SSH) data.
    """
    import sys
    def __init__(self):
        """
        Initialize the SSH_L2 object.
        """
        return

    def load_data(self, fn, s3sys=None, lat_bounds=[]):
        """
        Load the Level 2 SSH data from a file or a URL.

        Args:
            fn (str): Filename or URL to the data file.
            lat_bounds (list): Optional. A list with two elements indicating
                the lower and upper latitude bounds to subset the data.
        Returns:
            bool: True if the data is successfully loaded, False otherwise.
        """

        # Load unsmoothed data
        if 'Unsmoothed' in fn:
            # Open datasets from 'podaac' source or local file
            if 'podaac' in fn:
                if s3sys==None:
                    sys.exit("Please include a pre-existing s3sys object in the input")
                ddl = xr.open_dataset(s3sys.open(fn, mode='rb'), group='left', engine='h5netcdf')
                ddr = xr.open_dataset(s3sys.open(fn, mode='rb'), group='right', engine='h5netcdf')
            else:
                ddl = xr.open_dataset(fn, group='left', engine='netcdf4')
                ddr = xr.open_dataset(fn, group='right', engine='netcdf4')

            # Assign datasets to self.left and self.right
            self.left = ddl 
            self.right = ddr
            print('Load unsmoothed data to self.left and self.right')

            # Subset data between specified latitude bounds
            if len(lat_bounds) == 2:
                print('Subset data between latitude bounds ', lat_bounds)
                self.left = self.subset(self.left, lat_bounds)
                self.right = self.subset(self.right, lat_bounds)
            return 

        # Load data from 'podaac' source or local file
        if 'podaac' in fn:
            if s3sys==None:
                sys.exit("Please include a pre-existing s3sys object in the input")
            dd = xr.open_dataset(s3sys.open(fn, mode='rb'), engine='h5netcdf')
        else:
            dd = xr.open_dataset(fn, engine='netcdf4')

        # Subset data between specified latitude bounds
        if len(lat_bounds) == 2:
            dd = self.subset(dd, lat_bounds)
            # Error in reading the file
            if type(dd) == type([]): 
                return False

        # Extract variable type from the filename
        var_type = fn.split('/')[-1].split('_')[4]

        # Set the variable as an attribute of the object
        setattr(self, var_type, dd)

        return True

    
    def subset(self,data_in,lat_bounds):
        lat=np.nanmean(data_in['latitude'].data,axis=-1)
        lat=np.where(np.isnan(lat),100,lat)
        l0,l1=lat_bounds
        i0=np.where(np.abs(lat-l0)==np.abs(lat-l0).min())[0][0]
        i1=np.where(np.abs(lat-l1)==np.abs(lat-l1).min())[0][0]
        if i0>i1:i0,i1=i1,i0
        if i0==i1:
            return []
        #sub = data_in.sel(num_lines=slice(i0, i1))
        # Subset all variables that share the latitude dimension
        subset_vars = {}
        for varname, var in data_in.data_vars.items():
            if var.dims==2:
                subset_vars[varname] = var[i0:i1,:]
            else:
                subset_vars[varname] = var[i0:i1]
        # Combine the subset variables into a new dataset
        subset_data = xr.Dataset(subset_vars, attrs=data_in.attrs)
        return subset_data
    def remove_phase_bias(self,data_in):
        anomaly,bias=fit_phase_bias(data_in)
        self.anomaly=anomaly
        self.phase_bias=bias
        return
        
class unsmoothed:    
    def __init__(self, fn):
        self.file_name = fn

    def load_data(self, lat_range=[], lon_range=[]):
        """
        Load unsmoothed SSH (Sea Surface Height) data from files, and return the combined data from left and right swaths.

        Parameters
        ----------
        fn : str
            File name or path to the netCDF file containing the SSH data.
        lat_range : list, optional
            A list with two elements representing the minimum and maximum latitude for filtering data. Defaults to an empty list (no filtering).
        lon_range : list, optional
            A list with two elements representing the minimum and maximum longitude for filtering data. Defaults to an empty list (no filtering).

        Returns
        -------
        lat : numpy.ndarray
            Combined latitude data from both left and right swaths.
        lon : numpy.ndarray
            Combined longitude data from both left and right swaths.
        ssh : numpy.ndarray
            Combined unsmoothed SSH data from both left and right swaths.
        """
        
        fn = self.file_name
        
        if 'podaac' in fn:
            ddl = xr.open_dataset(s3sys.open(fn, mode='rb'), group='left', engine='netcdf4')
            ddr = xr.open_dataset(s3sys.open(fn, mode='rb'), group='right', engine='netcdf4')
        else:
            ddl = xr.open_dataset(fn, group='left', engine='netcdf4')
            ddr = xr.open_dataset(fn, group='right', engine='netcdf4')

        self.left = ddl 
        self.right = ddr
     
        return
    
def filter_butter(data,cutoff,fs, btype,filter_order=4,axis=0):
    """filter signal data using butterworth filter.

    Parameters
    ==================
    data: N-D array
    cutoff: scalar
        the critical frequency
    fs: scalar
        the sampling frequency
    btype: string
        'low' for lowpass, 'high' for highpass, 'bandpass' for bandpass
    filter_order: scalar
        The order for the filter
    axis: scalar
        The axis of data to which the filter is applied

    Output
    ===============
    N-D array of the filtered data

    """

    import numpy as np
    from scipy import signal

    if btype=='bandpass':
            normal_cutoff=[cutoff[0]/0.5/fs,cutoff[1]/0.5/fs]
    else:
        normal_cutoff=cutoff/(0.5*fs)  #normalize cutoff frequency
    b,a=signal.butter(filter_order, normal_cutoff,btype)
    y = signal.filtfilt(b, a, data, axis=axis)

    return y

def compute_geostrophic_velocity(ssh, lat0, dx, dy):
    # Constants
    g = 9.81  # Gravity (m/s^2)
    omega = 7.2921e-5  # Earth's angular velocity (rad/s)

    # Step 1: Compute the horizontal gradients of SSH
    dssh_dx = np.gradient(ssh, axis=1) / dx
    dssh_dy = np.gradient(ssh, axis=0) / dy

    # Step 2: Calculate the Coriolis parameter (f) at the given latitude (lat0)
    f = 2 * omega * np.sin(np.deg2rad(lat0))

    # Step 3: Calculate the geostrophic velocity components
    u = -g / f * dssh_dy
    v = g / f * dssh_dx

    return u, v



def resample_swath_data(lon, lat, ssh, lon_out, lat_out, sigma=1000, radius_of_influence = 6000):
    """
    Resample swath data onto a uniform grid using Pyresample.
    
    Parameters
    ----------
    ssh : array_like
        2D array of swath data with shape (na, nc).
    lat : array_like
        2D array of latitude values with shape (na, nc).
    lon : array_like
        2D array of longitude values with shape (na, nc).
    lat_out : array_like
        2D array of latitude values for the output grid.
    lon_out : array_like
        2D array of longitude values for the output grid.

    Returns
    -------
    data_out : array_like
        2D array of resampled swath data with shape (ny, nx), where ny and nx
        are the dimensions of the output grid.
    """
    from pyresample import geometry, kd_tree, utils
    import pyresample as pr

    # Define input and output grids
    input_grid = geometry.SwathDefinition(lons=lon, lats=lat)
    output_grid = geometry.GridDefinition(lons=lon_out, lats=lat_out)
    # Some parameters for the Gaussian resampling
    #sigma = 2000  # Standard deviation of the Gaussian function in meters
    #radius_of_influence = 3 * sigma  # Search radius to find input data points

    # Resample the data using Gaussian weighting
    data_out = kd_tree.resample_gauss(
        input_grid, ssh, output_grid,
        radius_of_influence=radius_of_influence,
        sigmas=sigma,
        fill_value=None,
        with_uncert=False
    )

    return data_out


def identify_outliers_iqr(arr, size=3, threshold=1.5):
    """
    Identify outliers in a 2D array using the local median and interquartile range method.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array of data.
    size : int, optional
        Size of the window used to compute the local median, default is 3.
    threshold : float, optional
        Threshold for identifying outliers, defaults to 1.5.

    Returns
    -------
    outlier_mask : numpy.ndarray
        A boolean array of the same shape as the input array, where True
        indicates the position of an outlier.

    Usage
    -----
    To identify outliers in a 2D array `arr`, call this function with the
    input array `arr`, the desired window size for computing the local
    median (default is 3), and the threshold for identifying outliers
    (default is 1.5). The function returns a boolean array `outlier_mask`
    indicating the positions of the outliers.
    """
    import numpy as np
    from scipy.ndimage import median_filter
    # Calculate the local median using a median filter
    local_median = median_filter(arr, size=size)

    # Compute the local interquartile range (IQR)
    q1 = np.percentile(arr, 25, interpolation='midpoint')
    q3 = np.percentile(arr, 75, interpolation='midpoint')
    iqr = q3 - q1

    # Identify the outliers
    lower_bound = local_median - threshold * iqr
    upper_bound = local_median + threshold * iqr
    outlier_mask = (arr < lower_bound) | (arr > upper_bound)

    return outlier_mask




def distance_between_points(lon0, lons, lat0, lats):
    """
    Calculate the distance between two points on the Earth's surface
    using the haversine formula.
    
    Parameters:
    lon0 (float): The longitude of the first point in decimal degrees
    lons (float): The longitudes of the second point(s) in decimal degrees.
                  This can be a single value or an array-like object.
    lat0 (float): The latitude of the first point in decimal degrees
    lats (float): The latitudes of the second point(s) in decimal degrees.
                  This can be a single value or an array-like object.
    
    Returns:
    float or numpy.ndarray: The distance(s) between the two points in meters.
    
    """
    import numpy as np
    
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0
        
    # phi = 90 - latitude
    phi1 = lat0*degrees_to_radians
    phi2 = lats*degrees_to_radians
    dphi = phi1-phi2
    
    # theta = longitude
    theta1 = lon0*degrees_to_radians
    theta2 = lons*degrees_to_radians
    dtheta=theta1-theta2
    
    # The haversine formula
    co = np.sqrt(np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dtheta/2.0)**2)
    arc = 2* np.arcsin(co)
    dist = arc*6371.0e3
    
    return dist


def karin_error_correction(din):
    """
    Applies a correction to a segment of the level-2 KaRIn data to remove
    systematic errors in the ssha_karin_2 variable.

    Parameters
    ----------
    din : xarray.Dataset
        A segment of the level-2 KaRIn data in xarray Dataset format.

    Returns
    -------
    xarray.Dataset
        The input data with the ssha_karin_2 variable corrected for systematic
        errors and two new variables added to the Dataset: ssha_karin_2_corrected
        and ssha_karin_2_var. The ssha_karin_2_corrected variable is the corrected
        version of the ssha_karin_2 variable, while the ssha_karin_2_var variable
        is the standard deviation of the corrected ssha_karin_2 variable over the
        data segment.
    """
    
    ssh = din['ssha_karin_2']  # extracts the ssha_karin_2 variable
    lat, lon = din['latitude'], din['longitude']  # extracts the latitude and longitude variables

    # Subtract mean of ssha_karin_2 along its axis
    ssh -= ssh.mean(axis=0)

    # Flatten and remove NaNs
    m = np.isfinite(ssh)
    mf = m.data.flatten()

    # Fit a 2D surface to the ssha_karin_2 variable using latitude and longitude
    da, dm = fit2Dsurf(lon.data.flatten()[mf], lat.data.flatten()[mf], ssh.data.flatten()[mf], kind='linear')

    # Initialize a new array for the corrected ssha_karin_2 variable
    dd = np.zeros_like(ssh.data.flatten())

    # Apply the correction to the ssha_karin_2 variable
    dd[mf] = (da * 1).flatten()
    ssh.values = dd.reshape(ssh.shape)

    # Add the corrected ssha_karin_2 and its standard deviation to the input data segment
    din['ssha_karin_2_corrected'] = ssh
    din['ssha_karin_2_var'] = ssh.std(axis=0)
    
    return din
    
    
def fit_bias(ssh, cross_track_distance,
                   tol=1e-4,order=2,
                   iter_max=20,
                   remove_along_track_polynomial=False,
                   check_bad_point_threshold=0.6):
    """
    Parameters
    ----------
    ssh : xarray.DataArray
        A 2D array of SSH data.
    cross_track_distance : xarray.DataArray
        A 2D array of cross-track distances 
    tol : float, optional
        Tolerance for the phase bias correction, by default 1e-4.
    order : int, optional
        Order of the polynomial to fit to the phase bias, by default 2.
    remove_along_track_polynomial : bool, optional
        Flag to remove the along-track polynomial from the SSH data, by default False.
    check_bad_point_threshold : float, optional
        Threshold for checking the percentage of bad points, 
        larger than which the fitting will be skipped, by default 0.6.

    The two arrays must have the same shape. Missing values are filled with NaNs.

    Returns
    -------
    xarray.DataArray
        A 2D array of the phase bias.

    """

    from scipy.optimize import leastsq
    import numpy as np

    #if remove_along_track_mean:
    #    ssh -= np.nanmean(ssh,axis=0,keepdims=True)

    def err(cc,x0,p,order):
        if order == 2:
            a,b,c=cc 
            return p - (a + b*x0 + c*x0**2)
        if order == 3:
            a,b,c,d=cc
            return p - (a + b*x0 + c*x0**2 + d*x0**3)

    def surface(cc,x0,order):
        if order==2:
            a,b,c=cc 
            return a + b*x0 + c*x0**2
        if order==3:
            a,b,c,d=cc
            return a + b*x0 + c*x0**2 + d*x0**3
        
    def get_anomaly(ssha,distance,order):
        msk=np.isfinite(ssha.flatten())
        if msk.sum()<ssha.size*check_bad_point_threshold:
            return np.zeros_like(ssha)
        x=distance
        xf=x.flatten()[msk]
        pf=ssha.flatten()[msk]

        cc = [0.0]*(order+1)
        coef = leastsq(err,cc,args=(xf,pf,order))[0]
        anomaly = err(coef,x,ssha,order)
        
        return anomaly 
    
    cdis = np.nanmean(cross_track_distance,axis=0)/1e3
    
    
    m1=cdis>0
    m2=cdis<0

    
    ano = np.where(np.isfinite(ssh),np.zeros((ssh.shape)),np.nan)
    ano[:,m1]=get_anomaly(ssh[:,m1],cross_track_distance[:,m1],order)
    ano[:,m2]=get_anomaly(ssh[:,m2],cross_track_distance[:,m2],order)

    for i in range(iter_max//2):
        ano[:,m1]=get_anomaly(ano[:,m1],cross_track_distance[:,m1],order)
        ano[:,m2]=get_anomaly(ano[:,m2],cross_track_distance[:,m2],order)
        #ano = np.where(np.abs(ano)>6*np.nanstd(ano),np.nan,ano)
    for i in range(iter_max//2):
        ano[:,m1]=get_anomaly(ano[:,m1],cross_track_distance[:,m1],order)
        ano[:,m2]=get_anomaly(ano[:,m2],cross_track_distance[:,m2],order)
        ano = np.where(np.abs(ano-np.nanmean(ano))>5*np.nanstd(ano),np.nan,ano)

    ano = np.where(np.isnan(ssh),np.nan,ano)
    #mm = m1|m2
    #ano[:,~mm]=np.nan
    
    if remove_along_track_polynomial:
        y = np.arange(ssh.shape[0])[:,np.newaxis]*np.ones_like(ssh)
        ano = fit_along_track_polynomial(y,ano)

    return ano

def fit_along_track_polynomial(y,din):
    """
    Computes the best-fit 2D surface of the form
    p = a + by + cy^2 + d y^3
    
    The best-fit surface is determined by minimizing the sum 
    of squared residuals between the functional surface and the input data.

    Parameters
    ----------
    y : numpy.ndarray
        A 2D array or a list of y-coordinates.
    p : numpy.ndarray
        A 2D array or a list of data values on (y) grid.

    Returns

    
    """
    
    from scipy.optimize import leastsq
    import numpy as np

    def err(cc,y0,p):
        a,b,c,d,e=cc
        return p - (a + b*y0 + c*y0**2 + d*y0**3+e*y0**4)

    def surface(cc,y0):
        a,b,c,d,e=cc
        return a + b*y0 + c*y0**2 + d*y0**3 + e*y0**4

    msk=np.isfinite(din.flatten())
    if msk.sum()<din.size/3:
        return np.zeros_like(din)*np.nan
    yf=y.flatten()[msk]
    dd=din.flatten()[msk]
    cc = [1e-4,1e-6,1e-10,1e-10,1e-10]

    coef = leastsq(err,cc,args=(yf,dd))[0]

    anomaly = err(coef,y,din) #mean surface

    return anomaly

def fit2Dsurf(x,y,p,kind='linear'):
    """
    Computes the best-fit 2D surface of the form
    p = a + bx + cy + dx^2 + ey^2 + fxy (quadratic) or
    p = a + bx + cy (linear) depending on the value of kind. 
    The best-fit surface is determined by minimizing the sum 
    of squared residuals between the functional surface and the input data.

    Parameters
    ----------
    x : numpy.ndarray
        A 2D array or a list of x-coordinates.
    y : numpy.ndarray
        A 2D array or a list of y-coordinates.
    p : numpy.ndarray
        A 2D array or a list of data values on (x,y) grid.
    kind : str, optional
        The type of surface to fit to the data. Can be either 'linear'
        or 'quadratic'. Default is 'linear'.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The computed anomaly and mean surfaces. The anomaly surface is 
        given by the difference between the input data and the mean surface.
        The mean surface is computed by evaluating the best-fit 2D surface
        at the input coordinates.
    """
    
    from scipy.optimize import leastsq
    import numpy as np

    def err(c,x0,y0,p):
        if kind=='linear':
            a,b,c=c
            return p - (a + b*x0 + c*y0 )
        if kind=='quadratic':
            a,b,c,d,e,f=c
            return p - (a + b*x0 + c*y0 + d*x0**2 + e*y0**2 + f*x0*y0)

    def surface(c,x0,y0):
        if kind=='linear':
            a,b,c=c
            return a + b*x0 + c*y0 
        if kind=='quadratic':
            a,b,c,d,e,f=c
            return a + b*x0 + c*y0 + d*x0**2 + e*y0**2 + f*x0*y0

    
    aa=x*y*p
    msk=np.isfinite(aa.data.flatten())
    #x=x.flatten()[msk];y=y.flatten()[msk];p=p.flatten()[msk]
    
    #dpdy = (np.diff(p,axis=0)/np.diff(y,axis=0)).mean()
    #dpdx = (np.diff(p,axis=1)/np.diff(x,axis=1)).mean()
    dpdx=(p.max()-p.min())/(x.max()-x.min())
    dpdy=(p.max()-p.min())/(y.max()-y.min())
    xf=x.data.flatten()[msk]
    yf=y.data.flatten()[msk]
    pf=p.data.flatten()[msk]

    
    if kind=='linear':
        c = [pf.mean(),dpdx,dpdy]
    if kind=='quadratic':
        c = [pf.mean(),dpdx,dpdy,1e-22,1e-22,1e-22]

    coef = leastsq(err,c,args=(xf,yf,pf))[0]
    vm = surface(coef,x,y) #mean surface
    va = p - vm #anomaly
    
    return va,vm


def along_track_distance(lats,lons):
    """
    Calculates the along-track distance between adjacent grid points 
    on the surface of the Earth using the Haversine formula. 

    Parameters
    ----------
    lats : numpy.ndarray
        A 2D array of latitudes in degrees.
    lons : numpy.ndarray
        A 2D array of longitudes in degrees.

    Returns
    -------
    numpy.ndarray
        A 2D array of along-track distances in meters. The values in the 
        resulting array correspond to the distance between adjacent grid 
        points in the input latitude and longitude arrays.
    """
    
    from geopy.distance import geodesic
    nj,ni=lats.shape
    dis=np.zeros_like(lats)
    for j in range(1,nj):
        for i in range(ni):
            dis[j,i]=geodesic((lats[j,i],lons[j,i]),(lats[j-1,i],lons[j-1,i])).m
    return dis


def interp_alongtrack(ssha,lats,lons,dx=250):
    """
    Interpolate along-track data to a constant distance step.

    Parameters
    ----------
    ssha : numpy array
        Array of data to be interpolated.
    lats : numpy array
        Array of latitudes.
    lons : numpy array
        Array of longitudes.
    dx : int, optional
        Distance step for interpolation, by default 250.

    Returns
    -------
    numpy array
        Interpolated data.
    """
    # Get the shape of the lats array
    nj, ni = lats.shape
    
    # Calculate along-track distance for each point and cumulative distance along track
    dis = along_track_distance(lats, lons).cumsum(axis=0)
    
    # Find the minimum value of the cumulative distance along track
    ext = dis[-1, :].min()
    
    # Create a new array of evenly spaced distances
    new_j = np.arange(0, ext, dx)
    
    # Create a new array to hold the interpolated values
    new_ssha = np.zeros((new_j.size, ni))
    
    # Loop through each cross-swath track
    for i in range(ni):
        # Find the indices of non-nan values
        msk = ~np.isnan(ssha[:, i])
        
        # If more than half of the values are non-nan, interpolate
        if msk.sum() > nj/2:
            new_ssha[:, i] = np.interp(new_j, dis[:, i][msk], ssha[:, i][msk])
            
    return new_j, new_ssha



def find_closest_segment(fn, lat0, ddeg=4, do_correction=False):
    """
    Finds the closest segment to a given point (lat0, lon0) within a given radius of degrees.

    Parameters
    ----------
    fn : str
        L2_LR_SSH filename.
    lat0 : float
        Latitude of the point of interest.
    ddeg : float, optional
        Distance in degrees within which to search for the closest segment, by default 4.
    do_correction : bool, optional
        Flag to run karin_error_correction function on the dataset, by default False.

    Returns
    -------
    tuple or xarray Dataset
        If do_correction is False, a tuple of indices of the start and end of the closest segment; 
        if do_correction is True, an xarray Dataset of the closest segment with Karin error correction applied.
    """
    import xarray as xr
    
    # Open the dataset from the file
    d = xr.open_dataset(fn, mode='r')
    
    # Get the latitude and longitude arrays
    lats, lons = d['latitude'].values, d['longitude'].values
        
    if 'cross_track_distance' not in list(d.keys()):
        ctd = distance_between_points(lats[:,1:],lats[:,:-1],lons[:,1:],lons[:,:-1])
        cross_track_distance=np.r_[0,ctd.mean(axis=0)]/1e3
        print(cross_track_distance.shape)
        dd = {'cross_track_distance':xr.DataArray(cross_track_distance,dims=('num_pixels',))}
    else:
        cross_track_distance=d['cross_track_distance'].mean(axis=0)
        dd={'cross_track_distance':cross_track_distance}
        
    # Check if latitude nadir and longitude nadir arrays exist
    try:
        lats_nadir, lons_nadir = d['latitude_nadir'].values, d['longitude_nadir'].values
    except:
        lats_nadir, lons_nadir = np.nanmean(lats, axis=1), np.nanmean(lons, axis=1)
    
    # Get the number of rows and columns in the latitude and longitude arrays
    nj, ni = lats.shape
    
    # Create a boolean mask for the latitude nadir array based on the given latitude and radius
    mask = (lats_nadir > lat0 - ddeg) & (lats_nadir < lat0 + ddeg)
    
    # Check if there are any True values in the mask
    if mask.sum() == 0:
        del d
        return None
    else:
        
        for key in d.keys():
            # Copy the data from the dataset into a new dictionary using the boolean mask
            try:
                dd[key] = d[key][mask, :].copy()
            except:
                dd[key] = d[key][mask].copy()
        del d
        
        # Check if Karin error correction is required
        if do_correction:
            # Apply the Karin error correction function to the new dataset and return the result
            return karin_error_correction(xr.Dataset(dd))
        else:
            # Return a tuple of indices of the start and end of the closest segment
            return dd['idx_segment_start'], dd['idx_segment_end']

def along_track_spectrum(fn, lat0, varname='ssha_karin_2', ddeg=4, dx=250, do_correction=True):
    """
    Compute the along-track spectrum of sea surface height anomalies at a specified latitude.

    Args:
        fn (str): File name of the dataset to be analyzed.
        lat0 (float): Latitude at which to compute the along-track spectrum.
        varname (str, optional): Name of the variable to be analyzed (default 'ssha_karin_2').
        ddeg (float, optional): Degree of latitude to be considered for selecting the data segment (default 4).
        dx (float, optional): Along-track sampling distance in meters (default 250).
        do_correction (bool, optional): Whether to perform an additional correction to the selected data segment (default True).

    Returns:
        xarray.Dataset: A dataset containing the following variables:
            - freq: wavenumber frequency [cycles/km]
            - psd: power spectral density [m^2/cycles/km]
            - psd_lownoise: average power spectral density over the 30 lowest noise pixels
            - psd_highnoise: average power spectral density over the 10 highest noise pixels
            - psd_mean: average power spectral density over the pixels within the range 10-60 km from the track
            - ssha_interp: sea surface height anomalies interpolated along the track
            - cross_track_distance: distance from the track [km]
    """

    import numpy as np
    from scipy import signal
    import xarray as xr

    # Find the closest data segment around the specified latitude
    dd = find_closest_segment(fn, lat0, ddeg, do_correction)
    lat, lon = dd['latitude'].values, dd['longitude'].values

    # Compute cross-track distance if not already present in the dataset
    if 'cross_track_distance' not in list(dd.keys()):
        ctd = distance_between_points(lat[:, 1:], lat[:, :-1], lon[:, 1:], lon[:, :-1])
        cross_track_distance = np.r_[0, ctd.mean(axis=0)] / 1e3
    else:
        cross_track_distance = dd['cross_track_distance'].mean(axis=0)

    # Interpolate sea surface height anomalies along the track
    newy, ssha = interp_alongtrack(dd[varname], lat, lon, dx)
    nn, ni = ssha.shape

    # Calculate the power spectral density using Welch's method
    ff, ss = signal.welch(ssha[:, 20] * 100, nfft=nn, nperseg=nn, noverlap=0, detrend='linear', window='hanning', fs=1e3 / dx)
    ss = np.zeros((ff.size, ni))

    for i in range(ni):
        if ssha[:, i].sum() != 0:
            ff, ss[:, i] = signal.welch(ssha[:, i] * 100, nfft=nn, nperseg=nn, noverlap=0, detrend='linear', window='hanning', fs=1e3 / dx)

    # Create the output dataset with the computed variables
    msk = ff < 1 / 2.
    num_pixels = range(ni)

    ff = xr.DataArray(data=ff[msk], name='freq', dims=('wavenumber'))
    ss = xr.DataArray(data=ss[msk, :], name='psd', dims=('wavenumber','num_pixels'))
    cross_track_distance = xr.DataArray(data=cross_track_distance, name='cross_track_distance', dims=('num_pixels'))
    ssha = xr.DataArray(data=ssha, name='ssha', dims=('y', 'num_pixels'), coords={'y': newy / 1e3, 'num_pixels': num_pixels})
    #ssha['y'].attrs['units'] = 'meters'
    
    noise = ss[ff > 1 / 10, :].mean(axis=0).data
    msk = noise != 0
    index_low = np.argsort(noise[msk])

    # Calculate the low noise power spectral density
    ss_low = ss[:, msk][:, index_low[:30]].mean(axis=-1)

    # Calculate the high noise power spectral density
    ss_high = ss[:, msk][:, index_low[-10:]].mean(axis=-1)

    # Calculate the mean power spectral density within the range of 10-60 km from the track
    dis = np.abs(cross_track_distance / 1e3)
    msk = (dis > 10) & (dis < 60)
    ss_mean = ss[:, msk].mean(axis=-1)

    # Create the output dataset with the computed variables
    dout = xr.Dataset({'freq': ff, 'psd': ss,
                       'psd_lownoise': ss_low,
                       'psd_highnoise': ss_high,
                       'psd_mean': ss_mean,
                       '%s_interp' % varname: ssha,
                       'cross_track_distance': cross_track_distance})

    return dout

def plot_spectrum(spec,ax,varns=['psd_mean'],params=None):
    """
    Plot the spectrum data on the given axis.
    
    Parameters:
    spec (xarray.Dataset): output from along_track_spectrum. Example:
        <xarray.Dataset>
        Dimensions:              (wavenumber: 900, num_pixels: 69, y: 1799)
        Coordinates:
          * y                    (y) float64 0.0 0.25 0.5 0.75 ... 449.0 449.2 449.5
        Dimensions without coordinates: wavenumber, num_pixels
        Data variables:
            freq                 (wavenumber) float64 0.0 0.002223 ... 1.997 1.999
            psd                  (wavenumber, num_pixels) float64 0.0 0.0 ... 0.0
            ssha_karin_2_interp  (y, num_pixels) float64 0.0 0.0 0.0653 ... 1.026 0.0
    ax (matplotlib.axes._subplots.AxesSubplot): the figure axis
    sel_pixels (list): a list of index for selecting the pixels
    params (tuple): p['label','lw','color']
    """
    p={'label':'','lw':2,'color':'k'}
    if params!=None:
        for key in params.keys():
            p[key]=params[key]
    
    for var in varns:
        ax.loglog(spec['freq'],spec[var],
              label=p['label'],lw=p['lw'],color=p['color'])
    
    plt.minorticks_on()
    plt.grid(True,which='both')
    msk=(spec['freq']<1/4)
    
    plt.ylim(spec['psd_mean'][msk].min()*0.8,spec['psd_mean'][msk].max()*1.2)
    return ax

def fit_spectrum_slope(k, y, k_range=(1/300,1/70), ax=None):
    """
    Parameters:
    -----------
    k: array
        The array of wave number
    y: array
        The array of y
    k_range: tuple
        The range of wave number to be considered for fitting
    ax: axis
        The axis on which the fitted curve should be plotted 
    Returns:
    --------
    slope: float
        The slope of the best fit curve of the form y = a + b*log10(k)
    
    """
    import numpy as np
    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a + x*b
    msk=(k>k_range[0])&(k<=k_range[1])
    popt, pcov = curve_fit(func, np.log10(k[msk]), np.log10(y[msk]))
    print('spectrum slope for ',k_range,' is ',popt[1])
    if ax!=None:
        ax.loglog(k[msk],10**popt[0]*k[msk]**popt[1],'--',color='gray',lw=3)
    ax.set_xlabel('Wavenumber (cpkm)')
    ax.set_ylabel('Power Spectral Density (cm$^2$/cpkm)')
    return popt[1]

def plot_science_requirement(ax,k):
    """
    Plot the SWOT science requirement in a log-log plot.

    Parameters:
    -----------
    ax: matplotlib.axes.Axes object
        Axes object to plot the scientific requirement.
    k: array_like
        Array of wavenumber frequency [cycles/km] to be plotted.

    Returns:
    --------
    None
    """
    ax.loglog(k,2+0.00125*k**(-2),'g',lw=3)
    ax.vlines(1/15,1e1,1e3,color='gray')
    ax.text(1/15,1e3,'15 km',fontsize=16)
    return

def plot_mooring_locations(ax,num=11):
    import pandas as pd
    moorings= pd.read_csv('mooring_location.txt', skiprows=1)
    #print(moorings)
    loc=moorings.to_numpy()
    ax.scatter(loc[:num,2],loc[:num,1],s=60,c='b',marker='^',zorder=1)
    return loc


def calculate_mean_and_bias(ssh,epoch=60000):
    """
    Compute anomalies in the SSH data using a neural network model.
    
    Parameters:
    - ssh (numpy.ndarray): Input SSH data with shape (time, num_lines, num_pixels).
    - epoch (int, optional): Number of training epochs. Default is 60000.
    
    Returns:
    - model (torch.nn.Module): Trained neural network model.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    #SSH has shape of (time, num_lines, num_pixels). This program will create a mean and anomaly field
    # Create x tensor
    nt, nx, nl = ssh.shape
    x = np.arange(nx)/nx
    x_torch = torch.tensor(x.reshape(1,-1, 1), dtype=torch.float32)
    y_torch = torch.tensor(np.nan_to_num(ssh), dtype=torch.float32)

    # Define the function f(x, a) and the model
    class ConstantModel(nn.Module):
        def __init__(self, nt, nl):
            super(ConstantModel, self).__init__()
            self.a = nn.Parameter(torch.randn(nt,1,nl)*1e-3) #bias
            self.b = nn.Parameter(torch.randn(1,nx,nl)*1e-3) #mean SSH (including smallscale geiod and mesoscale mean
        def forward(self,):
            return self.a  + self.b  # Shape: (nt, nx, nl)

    # Initialize the model, loss function, and optimizer
    model = ConstantModel(nt, nl)

    optimizer = optim.SGD(model.parameters(), lr=2)

    def masked_rmse_loss(y_true, y_pred, mask):
        """Calculate the masked RMS difference."""
        #print(y_true.shape,y_pred.shape)
        diff = y_true - y_pred
        masked_diff = diff * mask
        return torch.sqrt(torch.sum(masked_diff ** 2) / torch.sum(mask))

    # Create a mask for non-NaN values in the original ssh data
    mask = torch.tensor(np.isnan(ssh)==0, dtype=torch.float32)

    # Train the model
    num_epochs = epoch
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model()
        loss = masked_rmse_loss(y_torch, outputs, mask)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for this epoch
        if np.mod(epoch,6000)==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} meters') 
    #select good passes and iterate the mean
    bias0 = model.a.detach().numpy().copy() #cross-swath bias
    time_mean = model.b.detach().numpy().copy() #time mean
    
    return bias0,time_mean