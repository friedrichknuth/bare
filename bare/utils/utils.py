import pandas as pd
import subprocess
from subprocess import Popen, PIPE, STDOUT
import sys
import os
from distutils.spawn import find_executable
import multiprocessing
from functools import partial
from osgeo import gdal

import bare.io
import bare.geospatial


def create_overview(img_file_name, scale=8):
    # TODO
    # - Execute with run_command function for realtime output.
    '''
    Function to generate overviews using gdaladdo. gdal must be installed to environment.
    Default overview scale is 8. Can specify multiple overview scales at once, e.g. scale = 8 16 32 64.
    '''
    
    if find_executable('gdaladdo') is None:
        print('gdaladdo command not found. Please make sure gdal is installed to your environment. Cannot generate overview.')
        sys.exit(1)
    
    overview_file = img_file_name+'.ovr'
    
    if os.path.isfile(overview_file):
        print(overview_file, 'already exists.')
        return
        
    else:
        # build call
        call = 'gdaladdo -ro -r average --config COMPRESS_OVERVIEW LZW --config BIGTIFF_OVERVIEW YES'.split()
        call.append(img_file_name)
        call.append(str(scale))
    
        # run command and print to standard out
        subprocess.check_output(call)

def parallel_create_overview(image_list, scale=8, threads=8):
    # TODO
    # - Use os.cpu_count() to determine optimal thread count based on machine, if reasonable approach.
    '''
    Function to run create_overview() in parallel. Threads set to 8 by default.
    '''
    print("Using",threads,'threads to create overviews.')
    
    # set up threads pool
    p = multiprocessing.Pool(threads)
    
    # create partial function with scale input
    func = partial(create_overview, str(scale))
    
    # iterate over image list
    p.map(create_overview, image_list)




def generate_corner_coordinates(image_file_name, 
                                camera_file, 
                                reference_dem, 
                                verbose=False):
    # TODO
    # - Make reference_dem optional and download coarse global DEM for target area if not supplied.
    # - Add cam_gen options --height-above-datum approximated_value and --datum WGS84 to accomodate DEM with potential holes in it,or insufficient extent.
    """
    Function to generate corner coordinates using cam_gen. Continuous reference DEM for full coverage area must be supplied to approximate footprint.
    """
 
    image_file_base_name = os.path.splitext(image_file_name)[0]
    extension = os.path.splitext(camera_file)[-1]
    out_cam = image_file_base_name + '_cam_gen.tsai'
    gcp_file = image_file_base_name + '.gcp'
    
    if not os.path.isfile(gcp_file):
        print("Running ASP cam_gen to calculate image footprint on ground from input camera file and reference DEM.")
        print('Assuming corner coordinates derived from reference DEM are in EPSG 4326.')
        
        if extension == '.tsai':
            call = ['cam_gen', image_file_name, 
                    '--reference-dem', reference_dem, 
                    '-o', out_cam, 
                    '--gcp-file', gcp_file, 
                    '--sample-file', camera_file,
                    '--input-camera', camera_file]
                
        elif extension == '.xml':
            call = ['cam_gen', image_file_name, 
                    '--camera-type', 'opticalbar',
                    '--reference-dem', reference_dem, 
                    '-o', out_cam, 
                    '--gcp-file', gcp_file, 
                    '--sample-file', camera_file,
                    '--input-camera', camera_file]
      
        run_command(call, verbose=verbose)
        return gcp_file
    else:
        print(gcp_file, 'already exists.')
        print('Using',gcp_file, 'to generate footprint.')
        return gcp_file


def run_command(command, verbose=False):
    
    p = Popen(command,
              stdout=PIPE,
              stderr=STDOUT,
              shell=False)
    
    while p.poll() is None:
        line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
        if verbose == True:
            print(line)


def download_srtm(LLLON,LLLAT,URLON,URLAT,
                  output_dir='./data/reference_dems/',
                  verbose=True):
    # TODO
    # - Add function to determine extent automatically from input cameras
    # - Make geoid adjustment and converstion to UTM optional
    import elevation
    
    run_command(['eio', 'selfcheck'], verbose=verbose)
    print('Downloading SRTM DEM data...')

    hsfm.io.create_dir(output_dir)

    cache_dir=output_dir
    product='SRTM3'
    dem_bounds = (LLLON, LLLAT, URLON, URLAT)

    elevation.seed(bounds=dem_bounds,
                   cache_dir=cache_dir,
                   product=product,
                   max_download_tiles=999)

    tifs = glob.glob(os.path.join(output_dir,'SRTM3/cache/','*tif'))
    
    vrt_file_name = os.path.join(output_dir,'SRTM3/cache/srtm.vrt')
    
    call = ['gdalbuildvrt', vrt_file_name]
    call.extend(tifs)
    run_command(call, verbose=verbose)

    
    ds = gdal.Open(vrt_file_name)
    vrt_subset_file_name = os.path.join(output_dir,'SRTM3/cache/srtm_subset.vrt')
    ds = gdal.Translate(vrt_subset_file_name,
                        ds, 
                        projWin = [LLLON, URLAT, URLON, LLLAT])
                        
    
    # Adjust from EGM96 geoid to WGS84 ellipsoid
    adjusted_vrt_subset_file_name_prefix = os.path.join(output_dir,'SRTM3/cache/srtm_subset')
    call = ['dem_geoid','--reverse-adjustment', vrt_subset_file_name, '-o', adjusted_vrt_subset_file_name_prefix]
    run_command(call, verbose=verbose)
    
    adjusted_vrt_subset_file_name = adjusted_vrt_subset_file_name_prefix+'-adj.tif'

    # Get UTM EPSG code
    epsg_code = hsfm.geospatial.wgs_lon_lat_to_epsg_code(LLLON, LLLAT)
    
    # Convert to UTM
    utm_vrt_subset_file_name = os.path.join(output_dir,'SRTM3/cache/srtm_subset_utm_geoid_adj.tif')
    call = 'gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -dstnodata -9999 -r cubic -t_srs EPSG:' + epsg_code
    call = call.split()
    call.extend([adjusted_vrt_subset_file_name,utm_vrt_subset_file_name])
    run_command(call, verbose=verbose)
    
    return utm_vrt_subset_file_name