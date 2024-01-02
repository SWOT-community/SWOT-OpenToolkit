# SWOT-OpenToolkit

⚠️ An open repository of community-contributed codes for processing SWOT data. Official project algorithms are not included.

The current code focuses on dealing with the KaRIn during the fast-repeat phase. The following image is used to quick search the pass numbers that are relavent to regional interests. 

![Passes over the calval period](media/calval_passes_locations.png)


## Get started 

 - [A Handbook on level-2 SSH products](docs/chap0_get_started.md)
 - [How to contribute](contrib/CONTRIBUTING.md)
---
## Quick Examples 

1. [Remove cross-swath bias in 2km-resolution ssha_Karin_2](examples/remove_crossswath_bias.ipynb). The following is an example output. 

    <img src="media/figures/ssha_karin_2_california.png" alt="Alt Text" width="200">

1. [Sea ice and iceberg in sig0 at 250m resolution](examples/unsmoothed_sea_ice_250m.ipynb).

   <img src="media/figures/Unsmoothed_sig0_images/SWOT_L2_LR_SSH_Unsmoothed_486_005_20230409T233402_20230410T002508_PIA1_01.png" alt="sig0 over sea ice" width="200">
   <img src="media/figures/worldview/snapshot-2023-04-09T00_00_00Z.png" alt="sig0 over sea ice" width="200">

1. [Identify the pass number and timing of the science orbit over a region](src/find_swot_passes_science.py).

     Run the program as follows:

   ```
      python find_swot_timing_science.py -sw_corner -130.0 35.0 -ne_corner -125.0 40.0 -output_filename /tmp/test.png
   ```

      You will get something like the following figure. It contains the pass number, the satellite passing time (UTC) and the associated visualization. 

   <img src="media/figures/science_orbit_timing_example_quebec.png" alt="Alt Text" width="200">

1. [Plot the unsmoothed SSH (250m posting) near coast](https://github.com/SWOT-community/SWOT-OpenToolkit/blob/main/examples/unsmoothed_coastal.ipynb) 

   <img src="media/figures/unsmoothed_SF_coast.png" alt="unsmoothed SSH" width="200">

## Additional Resources:
- **Consider visiting the NASA [PO.DAAC Cookbook: SWOT Chapter](https://podaac.github.io/tutorials/quarto_text/SWOT.html) for more data resources and tutorials related to both the hydrology and oceanography SWOT communities.**
- **Product description documents for every SWOT collection in the table [here](https://podaac.jpl.nasa.gov/SWOT?tab=datasets-information).**
