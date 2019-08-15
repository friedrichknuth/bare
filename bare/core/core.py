import os
import glob
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import Point, Polygon
import contextily as ctx
import matplotlib.pyplot as plt

import bare.io
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
- extract 3D points and plot
    # geometry = [Point(xyz) for xyz in zip(df['lon'], df['lat'], df['elevation'])]
    # gdf = gpd.GeoDataFrame(df,geometry=geometry,crs={'init':'epsg:4326'})
'''

def gcp_corners_to_gdf_polygon(gcp_file):
    # TODO
    # Pass crs from reference DEM to bare.geospatial.df_points_to_polygon_gdf(df)
    """
    Function to extract image corner coordinates from gcp file genereated with ASP cam_gen
    and return as polygon within geopandas dataframe.
    """
    # parse ASP gcp file
    
    # try:
    df = pd.read_csv(gcp_file, header=None, delim_whitespace=True)
    if len(df) != 8:
        print('''
        Unable to intersect all rays with reference DEM. Please ensure reference DEM
        is continuous (no holes) and of sufficent extent. Consider mapprojecting the images
        to determine required extent. Deleting gcp file.
        ''')
        os.remove(gcp_file)
        return
    else:
        df = df.drop([0,4,5,6,10,11],axis=1) # drop sigma columns
        df.columns = ['lat','lon','elevation','file_name','img_x','img_y']    
        df = df[:4] # first 4 are corners. next for are center of quadrants.
    # except:
    #     print(sys.exc_info()[0])
    #     pass

    polygon_gdf = bare.geospatial.df_points_to_polygon_gdf(df)
    polygon_gdf['file_name'] = os.path.split(df['file_name'][0])[-1]
    return polygon_gdf

def ba_pointmap_to_gdf(df, ascending=True):
    df = df.rename(columns={'# lon':'lon',
                            ' lat':'lat',
                            ' height_above_datum':'height_above_datum',
                            ' mean_residual':'mean_residual',
                            ' num_observations':'num_observations'})
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])] 
    gdf = gpd.GeoDataFrame(df,geometry=geometry,crs={'init':'epsg:4326'})
    gdf = gdf.to_crs({'init':'epsg:3857'})
    gdf = gdf.sort_values('mean_residual',ascending=ascending)
    return gdf
    
def extract_tsai_coordinates(camera_file):
    with open(camera_file) as f:
        sub = 'C = '
        lines = [line.rstrip('\n') for line in f]
        coords = [s for s in lines if sub in s]
        coords = coords[0].split()[-3:]
        lon = float(coords[0])
        lat = float(coords[1])
        alt = float(coords[2])
        geometry = Point(lon,lat,alt)
    return geometry
    
    
def wv_xml_to_gdf(wv_xml):
    wv_json = bare.io.xml_to_json(wv_xml)
    df = pd.DataFrame(wv_json['isd']['EPH']['EPHEMLISTList']['EPHEMLIST'])
    
    df = df[0].str.split(expand = True)
    df = df.drop([0,4,5,6,7,8,9,10,11,12],axis=1)
    df.columns = ['lon','lat','altitude']
    df = df.apply(pd.to_numeric)
    
    gdf = bare.geospatial.df_xyz_coords_to_gdf(df, z='altitude', crs='4978')
    
    return gdf
    
def tsai_to_gdf(camera_file):
    geometry = extract_tsai_coordinates(camera_file)
    gdf = gpd.GeoDataFrame(gpd.GeoSeries(geometry), columns=['geometry'], crs={'init':'epsg:4978'})
    return gdf
    
def iter_extract_tsai_coordinates(cam_dir, extension='.tsai'):
    geometries = []
    cam_files = sorted(glob.glob(os.path.join(cam_dir,'*'+extension)))
    for filename in cam_files:
        if filename.endswith(extension):
            geometry = extract_tsai_coordinates(filename)
            geometries.append(geometry)
    tsai_points = gpd.GeoDataFrame(crs={'init' :'epsg:4978'}, geometry=geometries)
    tsai_points['file'] = [os.path.basename(x).split('.')[0] for x in cam_files]
    return tsai_points
       
def iter_ip_to_csv(ba_dir):
    vwips = sorted(glob.glob(os.path.join(ba_dir,"*.vwip")))
    ip_csv_list = []
    for filename in vwips:
        fn = write_ip_to_csv(filename)
        ip_csv_list.append(fn)
    return ip_csv_list
    

def iter_mp_to_csv(ba_dir):
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
    # TODO 
    # - Need cleaner way to extract image file paths that belong to a .match file. Image names cannot have a '-' in them else this will break. Could write in a check and exception with custom error for this.
    # - Need to handle numpass 0 when no clean.match file is created.
    
    """
    Function to parse out image file names from match files.
    Image file names cannot have '-' in them, else this will break.
    """

    # get image extension in case extension prefix used (e.g. 8.tif)
    img_ext = '.' + img_extension.split('.')[-1]

    # extract image pair file names from ASP match file name
    # e.g. split ../run-v2_sub8__v3_sub8-clean.csv into v2_sub8 and v3_sub8
    match_img1_name = os.path.split(match_file)[-1].split('-')[-2].split('__')[0]
    img1_file_name = os.path.join(img_dir, match_img1_name+img_ext)
    match_img2_name = os.path.split(match_file)[-1].split('-')[-2].split('__')[1]
    img2_file_name = os.path.join(img_dir, match_img2_name+img_ext)
    
    return img1_file_name, img2_file_name

def parse_image_name_from_ip_file_name(ip_csv_fn, img_dir, img_extension):
    # TODO 
    # - Add check and give user useful error message if no image file found.
    """
    Function to parse out image file names from interest point files.
    """
    
    # get image extension in case extension prefix used (e.g. 8.tif)
    img_ext = '.' + img_extension.split('.')[-1]
    
    # get image base name from ip csv file.
    img_base_name = os.path.splitext(os.path.split(ip_csv_fn)[-1])[0].split('-')[-1]
    img_file_name = os.path.join(img_dir, img_base_name+img_ext)

    return img_file_name, ip_csv_fn
    
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
    with open(filename, 'rb') as mf, open(filename_out, 'w') as out:
        size1 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
        out.write('x1 y1\n')
        im1_ip = [read_ip_record(mf) for i in range(size1)]
        for i in range(len(im1_ip)):
            out.write('{} {}\n'.format(im1_ip[i][0], im1_ip[i][1]))
    return filename_out
        
        
def write_mp_to_csv(filename):
    filename_out = os.path.splitext(filename)[0] + '.csv'
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
