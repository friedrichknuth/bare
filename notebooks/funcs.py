import geopandas as gpd
import glob
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sys import argv
from osgeo import gdal
import matplotlib.pyplot as plt
import os


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def extract_gpd_geometry(gdf):
    x = []
    y = []
    z = []
    for i in range(len(gdf)): 
        x.append(gdf['geometry'].iloc[i].coords[:][0][0])
        y.append(gdf['geometry'].iloc[i].coords[:][0][1])
        z.append(gdf['geometry'].iloc[i].coords[:][0][2])
    
    gdf['x'] = x
    gdf['y'] = y
    gdf['z'] = z
    
    
def extract_tsai_coordinates(tsai_dir, extension):
    geometry = []
    for filename in sorted(glob.glob(tsai_dir+'*'+extension)):
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
    return tsai_points


def iter_ip_to_csv(ba_dir):
    print('converting interest point files (vwip) to csv...')
    vwips = sorted(glob.glob(ba_dir+"*.vwip"))
    ip_csv_list = []
    for filename in vwips:
        fn = write_ip_to_csv(filename)
        ip_csv_list.append(fn)
    return ip_csv_list
        

def iter_mp_to_csv(ba_dir):
    print('converting match point files to csv...')
    matches = sorted(glob.glob(ba_dir+"*clean.match"))
    if matches:
        print('processing clean.match files only.')
    else:
        print('no clean.match files found.')
        matches = sorted(glob.glob(ba_dir+"*.match"))
        if matches:
            print('processing .match files.')
        else:
            print('no .match files found.')
            sys.exit(1)
    match_csv_list = []
    for filename in matches:
        fn = write_mp_to_csv(filename)
        match_csv_list.append(fn)
    return match_csv_list

def plot_ip_over_images(ba_dir, img_dir, img_extension='8.tif', out_dir='plots/'):
    
    create_dir(out_dir)
    ip_csv_list = iter_ip_to_csv(ba_dir)
    img_list = sorted(glob.glob(img_dir+'*'+img_extension))
    
    for i,v in enumerate(ip_csv_list):
        img_base_name = img_list[i].split('/')[-1].split('.tif')[0]
        
        img = gdal.Open(img_list[i])
        img = np.array(img.ReadAsArray())
        
        df = pd.read_csv(v, delimiter=r"\s+")
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.scatter(df['x1'],df['y1'],color='r',marker='o',facecolor='none',s=10)
        ax.imshow(img,cmap='gray')
        ax.set_title('interest points\n'+img_base_name+'.tif')
        
        out = out_dir+img_base_name+'_ip_plot.png'
    
        fig.savefig(out)

def read_ip_record(mf):
    x, y = np.frombuffer(mf.read(8), dtype=np.float32)
    xi, yi = np.frombuffer(mf.read(8), dtype=np.int32)
    orientation, scale, interest = np.frombuffer(mf.read(12), dtype=np.float32)
    polarity, = np.frombuffer(mf.read(1), dtype=np.bool)
    octave, scale_lvl = np.frombuffer(mf.read(8), dtype=np.uint32)
    ndesc, = np.frombuffer(mf.read(8), dtype=np.uint64)
    desc = np.frombuffer(mf.read(int(ndesc * 4)), dtype=np.float32)
    iprec = [x, y, xi, yi, orientation, scale, interest, polarity, octave, scale_lvl, ndesc]
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
            out.write('{} {} {} {}\n'.format(im1_ip[i][0], im1_ip[i][1], im2_ip[i][0], im2_ip[i][1]))
    return filename_out
