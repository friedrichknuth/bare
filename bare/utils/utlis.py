import subprocess
import sys
from distutils.spawn import find_executable
import multiprocessing
from functools import partial

def create_overview(img_file_name, scale=8):
    '''
    Function to generate overviews using gdaladdo. gdal must be installed to environment.
    Default overview scale is 8. Can specify multiple overview scales at once, e.g. scale = 8 16 32 64.
    '''
    
    if find_executable('gdaladdo') is None:
        print('gdaladdo command not found. Please make sure gdal is installed to your environment. Cannot generate overview.')
        sys.exit(1)
    
    # build call
    call = "gdaladdo -ro -r average --config COMPRESS_OVERVIEW LZW --config BIGTIFF_OVERVIEW YES".split()
    call.append(img_file_name)
    call.append(str(scale))
    
    # run command and print to standard out
    return subprocess.check_output(call)

def parallel_create_overview(image_list, scale=8, threads=8):
    # TODO
    # use threading.active_count() or os.cpu_count() to determine optimal thread count based on machine,
    # if possible.
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