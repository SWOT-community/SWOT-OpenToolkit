#!/usr/bin/env python
"""
Script to remove the residual phase-bias (cross-swath error) from the SWOT KaRIn Sea Surface Height Anomaly (ssha_karin_2).

Pytorch.NN is used, but it is not necessary. Scipy.optimization.leastsq should do the trick too. 

The program estimated a field ssha_mean(num_lines,num_pixels) and phase_bias(timec, 1, num_pixels) that minimize the variances of ssha_karin_2(timec,num_lines,num_pixels)

The time mean should include the three-month mean of the oceanography circulation SSH as well as the MSS. 

This script will probably become obsolete after the project re-process the SWOT_LR_L2_SSH data product. 

Author: Jinbo Wang
Date: 04.05.2023
"""

import numpy as np
import xarray as xr
import swot_ssh_utils as swot_ssh
import argparse


def process_twice(fn_in):
    """
    Process the SSH data twice to refine the results.
    
    Parameters:
    - fn_in (str): Input filename.

    Returns:
    - dout (xarray.Dataset): Output dataset with the following new variables in addition to the original variables:
        - ssha_karin_2_2 (xarray.DataArray): Optimized SSH anomaly.
        - bias_phase (xarray.DataArray): Optimized cross-swath bias.
        - time_mean (xarray.DataArray): Optimized time mean.
    """

    d = xr.open_dataset(fn_in)
    ssh = d['ssha_karin_2']+ d['internal_tide_hret'] - 2*d['sea_state_bias_cor_2']
    ssh=np.where(d['ssha_karin_2_qual']==0,ssh.data,np.nan)

    a0,b0=swot_ssh.calculate_mean_and_bias(ssh,epoch=60000) # The model converged at 50000 epoch.
    
    ssh -= (a0+b0) 
    ssh[np.abs(ssh)>ssh.std()*5]=np.nan
    # run it again after removing the mean and bias, to refine the results
    a1,b1=swot_ssh.calculate_mean_and_bias(ssh,epoch=60000) # The model converged at 50000 epoch.
    
    ssh-=(a1+b1)
    ssh[np.abs(ssh)>ssh.std()*5]=np.nan
    
    bias_phase=xr.DataArray(data=a0+a1,dims=('timec','num_lines_1','num_pixels'),
                            attrs={'note':'optimized cross-swath bias, created by get_anomaly.py'})
    time_mean=xr.DataArray(data=b0+b1,dims=('timec_1','num_lines','num_pixels'),
                            attrs={'note':'optimized time mean, created by get_anomaly.py'})
        
    dout=d.copy()

    dout['ssha_karin_2_2'] = xr.DataArray(data=ssh,dims=('timec','num_lines','num_pixels'),
                                          attrs={'note':'ssha_karin_2_1 - bias_phase - time_mean, created by get_anomaly.py'})
    dout['bias_phase'] = bias_phase
    dout['time_mean'] = time_mean
    print(dout)
    return dout

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_filename', type=str, default='', help='The filename of the subset created by create_datacube.py')
    args = parser.parse_args()

    #fn_in=/mnt/flow/swot/Jinbo_Analysis/subset_california/data_cube_Expert_013_california.nc
    fn_in = args.subset_filename
    dout=process_twice(fn_in)
    dout.to_netcdf(fn_in.replace('.nc','.ssha.nc'))