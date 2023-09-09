"""
Find the SWOT satellite fly-by time through a bounding box. 
The time will have uncertainty of tens of seconds to minutes depending on 
the size of the bounding box.

Make sure download the two shapefile associated with the SWOT orbit data from the aviso website. You can use the included script to download the data.

    bash download_swot_orbit_data.sh

Usage:
    python find_swot_timing_science.py -sw_corner -130.0 35.0 -ne_corner -125.0 40.0 -output_filename /tmp/test.png

Author: Jinbo Wang
Date: 2021-07-21
"""

import geopandas as gpd
import pandas as pd
import shapely.geometry as geometry
from datetime import timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box, LineString, Point
import numpy as np
import os,argparse

#download the two shapefile associated with the SWOT orbit data from the aviso website
if os.path.exists("sph_science_nadir.zip") is False or os.path.exists("sph_science_swath.zip") is False:
    print("Please download the two shapefiles associated with the SWOT orbit data from the aviso website using the included script.\n")
    print("bash download_swot_orbit_data.sh\n")
    exit()

#parse the command line arguments
parser = argparse.ArgumentParser(description='Find the SWOT satellite fly-by time through a bounding box.')
parser.add_argument('-sw_corner', type=float, nargs=2, help='The southwest corner of the bounding box [lon, lat]')
parser.add_argument('-ne_corner', type=float, nargs=2, help='The northeast corner of the bounding box [lon, lat]')
parser.add_argument('-output_filename', type=str, help='The filename of the figure to save. e.g. test.png')
args = parser.parse_args()
sw_corner = args.sw_corner
ne_corner = args.ne_corner
output_filename = args.output_filename

#check the command line arguments
if sw_corner is None or ne_corner is None or output_filename is None:
    parser.print_help('Example: \n python find_swot_timing_science.py -sw_corner -130.0 35.0 -ne_corner -125.0 40.0 -output_filename /tmp/test.png')
    exit()
    
def find_time(ID_PASS_value, extended_bbox):
    
    """
    Find the index of the first point in the combined LineString (for the given ID_PASS) 
    that intersects with the extended bounding box.

    Parameters:
    - ID_PASS_value (int): The ID_PASS value for which to find the intersecting point.
    - extended_bbox (Polygon): The extended bounding box to check intersection.

    Returns:
    - int or None: Index of the first intersecting point, or None if no intersection is found.
    """
    nadir_data = gpd.read_file("sph_science_nadir.zip")

    def join_linestrings(group):
        """Join LineStrings in the order they appear in the file."""
        if len(group) == 1:
            return group.iloc[0].geometry
        else:
            # Combine LineStrings
            return LineString([pt for line in group.geometry for pt in line.coords])

    def index_of_first_point_inside_bbox(line, bbox):
        """Returns the index of the first point of the LineString that falls inside the bounding box."""
        for index, point in enumerate(line.coords):
            if bbox.contains(Point(point)):
                return index
        return 0 # If no point falls inside the bounding box
    
    # Filtering the GeoDataFrame for rows with the given ID_PASS_value
    subset_gdf = nadir_data[nadir_data["ID_PASS"] == ID_PASS_value]
    
    # Joining LineStrings if there are multiple rows with the same ID_PASS
    combined_geometry = join_linestrings(subset_gdf)
    
    # Finding the index of the first point inside the extended bounding box
    index = index_of_first_point_inside_bbox(combined_geometry, extended_bbox)

    time_str=subset_gdf['START_TIME'].iloc[0]
    _, day_num, time_str = time_str.split(" ", 2)
    days = int(day_num)
    time_parts = list(map(int, time_str.split(":")))
    delta = timedelta(days=days-1, hours=time_parts[0], minutes=time_parts[1], seconds=time_parts[2]+index*30) 
    return delta+pd.Timestamp("2023-07-21T05:33:45.768") 

# Load the shapefile
gdf_karin = gpd.read_file("sph_science_swath.zip")
# define the bounding box
bbox = geometry.box(sw_corner[0], sw_corner[1], ne_corner[0], ne_corner[1])
# extend the bounding box by 0.2 degree
extended_bbox = box(bbox.bounds[0] - 0.2, bbox.bounds[1] - 0.2, bbox.bounds[2] + 0.2, bbox.bounds[3] + 0.2) #for nadir data

# Filter the GeoDataFrame for rows that intersect with the extended bounding box
overlapping_segments = gdf_karin[gdf_karin.intersects(bbox)]

# Testing the subroutine with a sample ID_PASS value
#test_id_pass = 35
#find_time(test_id_pass, gdf_nadir, extended_bbox)

# Compute the index of the first point inside the bounding box for each intersecting segment
tt=[]
for a in overlapping_segments['ID_PASS']:
    tt.append(find_time(a, extended_bbox))
# Add the index as a new column
overlapping_segments["TIME"] = tt 

# Calculate passing times for four cycles
cycle_period = timedelta(days=20.864549)
overlapping_segments["TIME_0"] = overlapping_segments["TIME"] 
overlapping_segments["TIME_1"] = overlapping_segments["TIME"] + cycle_period
overlapping_segments["TIME_2"] = overlapping_segments["TIME"] + (2 * cycle_period)
overlapping_segments["TIME_3"] = overlapping_segments["TIME"] + (3 * cycle_period)

# Print the results on screen
print(overlapping_segments[["ID_PASS", "TIME_0", "TIME_1", "TIME_2", "TIME_3"]])

# Visualization of the results on a map using Cartopy and saving the figure to output_filename
# Set up the figure and axes with n by 3 subplots
n_segments = len(overlapping_segments)
n_rows = (n_segments + 2) // 3
xlim = (sw_corner[0], ne_corner[0])
ylim = (sw_corner[1], ne_corner[1])

def plotit(index, ax, row):
    # Set the extent of the map
    ax.set_extent([sw_corner[0], ne_corner[0], sw_corner[1], ne_corner[1]])  # Set extent based on your data's region

    # Add high-resolution coastline
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.LAND, edgecolor='none', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, edgecolor='none', facecolor='lightblue')
    # Plot the shapefile data
    #gdf.plot(ax=ax, transform=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Disable longitude labels at the top
    gl.right_labels = False  # Disable latitude labels on the right

    # Plot the overlapping segments with light green color
    overlapping_segments.loc[[index]].plot(ax=ax, edgecolor="blue", facecolor="lightgreen")
    ax.set_title(f"Pass: {row['ID_PASS']}")
    
    # Annotate with cycle, pass number, and the four lines of date and time
    annotation_text = f"Time 0: {row['TIME_0'].strftime('%Y-%m-%d %H:%M:%S')}\n" \
                      f"Time 1: {row['TIME_1'].strftime('%Y-%m-%d %H:%M:%S')}\n" \
                      f"Time 2: {row['TIME_2'].strftime('%Y-%m-%d %H:%M:%S')}\n" \
                      f"Time 3: {row['TIME_3'].strftime('%Y-%m-%d %H:%M:%S')}"
    
    ax.annotate(annotation_text, xy=(0.05, 0.1), xycoords='axes fraction', backgroundcolor='white')

fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.ravel()

# Plot each segment in a separate subplot
for idx, (index, row) in enumerate(overlapping_segments.iterrows()):
    ax = axes[idx]
    plotit(index, ax, row)

# Hide any remaining unused subplots
for idx in range(n_segments, n_rows * 3):
    axes[idx].axis('off')

plt.tight_layout()

if output_filename is not None:
    plt.savefig(output_filename)
