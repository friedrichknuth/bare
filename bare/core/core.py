import os
import glob
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import Point
import contextily as ctx
import matplotlib.pyplot as plt

import bare.common
import bare.geospatial

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")



#TODO
'''
- write converted csv files to a seperate folder
- hide functions with preceeding _ from inheritance
- add docstrings to functions
- need to handle numpass 0 when no clean.match file is created to parse matched 
  images from match file file name.
- find better way to parse image file names from match file. 
  currently will break if - character in image file name.
'''


def extract_tsai_coordinates(cam_dir, extension='.tsai'):
    geometry = []
    cam_files = sorted(glob.glob(os.path.join(cam_dir,'*'+extension)))
    for filename in cam_files:
        if filename.endswith(extension):
            with open(filename) as f:
                sub = 'C = '
                lines = [line.rstrip('\n') for line in f]
                coords = [s for s in lines if sub in s]
                coords = coords[0].split()[-3:]
                lon = float(coords[0])
                lat = float(coords[1])
                alt = float(coords[2])
                geometry.append(Point(lon,lat,alt))
    tsai_points = gpd.GeoDataFrame(crs={'init' :'epsg:4978'}, geometry=geometry)
    tsai_points['file'] = [os.path.basename(x).split('.')[0] for x in cam_files]
    return tsai_points
       
def iter_ip_to_csv(ba_dir):
#     print('converting interest point files (vwip) to csv...')
    vwips = sorted(glob.glob(os.path.join(ba_dir,"*.vwip")))
    ip_csv_list = []
    for filename in vwips:
        fn = write_ip_to_csv(filename)
        ip_csv_list.append(fn)
    return ip_csv_list
    

def iter_mp_to_csv(ba_dir):
#     print('converting match point files to csv...')
    matches = sorted(glob.glob(os.path.join(ba_dir, "*clean.match")))
    if matches:
        print('    processing clean.match files only.')
    else:
        print('    no clean.match files found.')
        matches = sorted(glob.glob(os.path.join(ba_dir, "*.match")))
        if matches:
            print('    processing .match files.')
        else:
            print('    no .match files found.')
            sys.exit(1)
    match_csv_list = []
    for filename in matches:
        fn = write_mp_to_csv(filename)
        match_csv_list.append(fn)
    return match_csv_list 

def parse_image_names_from_match_file_name(match_file, img_dir, img_extension):
    '''
    Function to parse out image file names from match files.
    Image file names cannot have '-' in them, else this will break.
    '''

    # Get image extension in case extension prefix used (e.g. 8.tif)
    img_ext = '.' + img_extension.split('.')[-1]

    # Extract image pair file names from ASP match file name
    # For example e.g. split ../run-v2_sub8__v3_sub8-clean.csv 
    # into v2_sub8 and v3_sub8.
    # TODO Need cleaner way to extract image file paths that belong
    # to a .match file. Image names cannot have a '-' in them else this will break.
    # Need to handle numpass 0 when no clean.match file is created.
    match_img1_name = os.path.basename(match_file).split('.')[0].split('-')[-2].split('__')[0]
    img1_file_name = os.path.join(img_dir, match_img1_name+img_ext)
    match_img2_name = os.path.basename(match_file).split('.')[0].split('-')[-2].split('__')[1]
    img2_file_name = os.path.join(img_dir, match_img2_name+img_ext)
    
    return img1_file_name, img2_file_name

def parse_image_name_from_ip_file_name(ip_csv_fn, img_dir, img_extension):
    '''
    Function to parse out image file names from interest point files.
    '''
    
    # Get image extension in case extension prefix used (e.g. 8.tif)
    img_ext = '.' + img_extension.split('.')[-1]
    
    # Get image base name from ip csv file. Should match image file in image directory.
    # TODO add check and give user useful error message if no image file found.
    img_base_name = os.path.splitext(os.path.split(ip_csv_fn)[-1])[0].split('-')[-1]
    img_file_name = os.path.join(img_dir, img_base_name+img_ext)

    return img_file_name, ip_csv_fn

    
    
def ba_pointmap_to_gdf(df):
    df = df.rename(columns={'# lon':'lon',
                            ' lat':'lat',
                            ' height_above_datum':'height_above_datum',
                            ' mean_residual':'mean_residual'})
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])] 
    gdf = gpd.GeoDataFrame(df,geometry=geometry,crs={'init':'epsg:4326'})
    gdf = gdf.to_crs({'init':'epsg:3857'})
    gdf = gdf.sort_values('mean_residual',ascending=True)
    return gdf
    
def read_ip_record(mf):
    x, y = np.frombuffer(mf.read(8), dtype=np.float32)
    xi, yi = np.frombuffer(mf.read(8), dtype=np.int32)
    orientation, scale, interest = np.frombuffer(mf.read(12), dtype=np.float32)
    polarity, = np.frombuffer(mf.read(1), dtype=np.bool)
    octave, scale_lvl = np.frombuffer(mf.read(8), dtype=np.uint32)
    ndesc, = np.frombuffer(mf.read(8), dtype=np.uint64)
    desc = np.frombuffer(mf.read(int(ndesc * 4)), dtype=np.float32)
    iprec = [x, y, xi, yi, orientation, 
             scale, interest, polarity, 
             octave, scale_lvl, ndesc]
    iprec.extend(desc)
    return iprec


def write_ip_to_csv(filename):
    filename_out = os.path.splitext(filename)[0] + '.csv'
    # print('converting',filename,'to',filename_out)
    with open(filename, 'rb') as mf, open(filename_out, 'w') as out:
        size1 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
        out.write('x1 y1\n')
        im1_ip = [read_ip_record(mf) for i in range(size1)]
        for i in range(len(im1_ip)):
            out.write('{} {}\n'.format(im1_ip[i][0], im1_ip[i][1]))
    return filename_out
        
        
def write_mp_to_csv(filename):
    filename_out = os.path.splitext(filename)[0] + '.csv'
    # print('writing',filename_out)
    with open(filename, 'rb') as mf, open(filename_out, 'w') as out:
        size1 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
        size2 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
        out.write('x1 y1 x2 y2\n')
        im1_ip = [read_ip_record(mf) for i in range(size1)]
        im2_ip = [read_ip_record(mf) for i in range(size2)]
        for i in range(len(im1_ip)):
            out.write('{} {} {} {}\n'.format(im1_ip[i][0], 
                                             im1_ip[i][1], 
                                             im2_ip[i][0], 
                                             im2_ip[i][1]))
    return filename_out
