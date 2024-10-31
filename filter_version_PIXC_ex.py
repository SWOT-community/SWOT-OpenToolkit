"""
Takes a list of directories of SWOT PIXC data, filters
them to find the best version of each granule, and writes
out a JSON containing the best files for each directory.

Currently built for PGC0 and PIC0 versions

Usage:
    python3 filterVersionRiverSP.py

Authors: Fiona Bennitt and Elisa Friedmann
Date: 2024-10-31
"""

import json
import os

import pandas as pd

def filterVersionPIXC(directories, outpath):
    """
    Reads in all filenames, sorts them, and retrieves the best version/
    counter for each file (e.g. PGC0 over PIC0, PIC0_03 over PIC0_02).
    Writes out a json with the filenames for filtering upon read in
    to the outpath directory.

    Parameters:
    directories (list): List of directories to search for best files.
    outpath (string): Where to save the output JSON.
    
    Returns:
    None
    """
    
    # Get all file names from directories
    for directory in directories:
        # List to store all file paths
        files = []
        for file in os.listdir(directory):
            files.append(file)
    
        print(f"There are {str(len(files))} original files in directory.")

        # Make DataFrame of filenames
        granules = pd.DataFrame({'files': files})
        granules['cycle'] = granules['files'].str.slice(16, 19)
        granules['pass'] = granules['files'].str.slice(20, 23)
        granules['tile'] = granules['files'].str.slice(24, 28)
        granules['version'] = granules['files'].str.slice(-10, -6)
        granules['counter'] = granules['files'].str.slice(-5, -3)    

        # Sort the files
        granules = granules.sort_values(by=['cycle', 'pass', 'tile', 'version', 'counter'],
                                        ascending=[True, True, True, True, False])    

        # Keep only the best version of each granule
        granules = granules.drop_duplicates(subset=['cycle', 'pass', 'tile'],
                                            keep='first')    

        # Extract the file names of files passing the test
        best_files = list(granules['files'])

        print(f"There are {str(len(best_files))} best files in directory.")


        # Split filepath for naming json
        pieces = dirs[0].split('/')

        # Write out best files as json
        with open(os.path.join(outpath, pieces[5] + '_filtered.json'), 'w', encoding='utf-8') as f:
            json.dump(best_files, f)

        print(f"Wrote out the unique and most recently processed {str(len(best_files))} files to {outpath}{pieces[5]}_filtered.json")

# Directories to filter    
dirs = ['/path_to_data_download/']
# Outpath for json
out = '/path_out/'

filterVersionPIXC(directories=dirs, outpath=out)