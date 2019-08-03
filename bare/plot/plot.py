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
import bare.core

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")


def plot_ip_over_images(ba_dir, 
                        img_dir, 
                        img_extension='.tif', 
                        out_dir='qc_plots/interest_points'):

    '''

    Function to visualize interest points in an image.

    All images ending in img_extension pattern will be used. 
    For example, img_extension can be '.tif' or '8.tif'

    '''

    print('plotting interest points over images...')

    # create output directory
    out_dir_abs = bare.common.create_dir(out_dir)

    # convert interest points in binary vwip files to csv.
    ip_csv_list = bare.core.iter_ip_to_csv(ba_dir)

    # get images list
    img_list = sorted(glob.glob(os.path.join(img_dir, '*'+img_extension)))

    # check if img_list and ip_csv_list same length
    if not len(ip_csv_list) == len(img_list):
        print(
            '''
            Length of interest point file list and image file list
            does not match. Check inputs. 
        
            TODO note: Need to implement exception to proceed with 
            corresponding files and list omissions.
            '''
        )
        sys.exit(1)

    # plot interest points over images
    for i,v in enumerate(ip_csv_list):
        img_base_name = os.path.basename(img_list[i]).split('.')[0]
    
        img = gdal.Open(img_list[i])
        img = np.array(img.ReadAsArray())
    
        df = pd.read_csv(v, delimiter=r"\s+")
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.scatter(df['x1'],df['y1'],
                   color='r',
                   marker='o',
                   facecolor='none',
                   s=10)
        ax.imshow(img,cmap='gray')
        ax.set_title('interest points\n'+
                     img_base_name+img_extension.split('.')[-1])
    
        out = os.path.join(out_dir_abs,
                           img_base_name+'_interest_points.png')
        fig.savefig(out, bbox_inches = "tight")
        plt.close()

def plot_mp_over_images(ba_dir, 
                        img_dir, 
                        img_extension='.tif', 
                        out_dir='qc_plots/match_points'): 

    '''

    Function to visualize match points found between two images.

    All images ending in pattern and contained in img_dir  
    will be extracted. For example, img_extension can be 
    '.tif' or '8.tif'

    Image names cannot have a '-' character in them, else
    the images for a given match file won't be extracted properly.

    TODO note: need cleaner way to extract image file paths that belong
    to a .match file.

    '''

    print('plotting match points over images...')

    # create output directory
    out_dir_abs = bare.common.create_dir(out_dir)

    # convert .match files to csv
    match_csv_list = bare.core.iter_mp_to_csv(ba_dir)

    # get images list
    img_list = sorted(glob.glob(os.path.join(img_dir, '*'+img_extension)))

    for match_csv_fn in match_csv_list:
        #TODO: This needs work
        img1_fn, img2_fn, match_img1_fn, match_img2_fn = \
                bare.core.parse_image_names_from_match_file_name(match_csv_fn,img_list)
        mp_plot(img1_fn, img2_fn, match_csv_fn, out_dir_abs)

def mp_plot(img1_fn, img2_fn, match_csv_fn, out_dir_abs=None, scale=1.0):
        """
        Helper function to generate plots fro two images and match csv
        """

        #Load match file into DataFrame
        df = pd.read_csv(match_csv_fn, delimiter=r"\s+")
        
        #Extract short filenames 
        match_img1_name = os.path.splitext(os.path.split(img1_fn)[-1])[0]
        match_img2_name = os.path.splitext(os.path.split(img1_fn)[-1])[0]
    
        fig, ax = plt.subplots(1,2,figsize=(10,6))
        fig.suptitle('Match points:\n%s' % os.path.split(match_csv_fn)[-1])

        buf_xsize = None
        buf_ysize = None

        img1_ds = gdal.Open(img1_fn)

        #Handle scaling of input images using ReadAsArray buffer sizes
        if scale > 1:
            buf_xsize = int(round(img1_ds.RasterXSize/scale))
            buf_ysize = int(round(img1_ds.RasterYSize/scale))
            df['x1'] /= scale
            df['y1'] /= scale

        img1 = img1_ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
        clim = np.percentile(img1, (2,98))
        ax[0].scatter(df['x1'],df['y1'],color='r',marker='o',facecolor='none',s=10)
        ax[0].imshow(img1, clim=clim, cmap='gray')
        #ax[0].set_title(img1_fn)
        ax[0].set_aspect('equal')
        img1 = None
        img1_ds = None

        img2_ds = gdal.Open(img2_fn)
        if scale > 1:
            buf_xsize = int(round(img2_ds.RasterXSize/scale))
            buf_ysize = int(round(img2_ds.RasterYSize/scale))
            df['x2'] /= scale
            df['y2'] /= scale

        img2 = img2_ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
        clim = np.percentile(img2, (2,98))
        ax[1].scatter(df['x2'],df['y2'],color='r',marker='o',facecolor='none',s=10)
        ax[1].imshow(img2, clim=clim, cmap='gray')
        #ax[1].set_title(img2_fn)
        ax[1].set_aspect('equal')
        img2 = None
        img2_ds = None
  
        plt.tight_layout()

        if out_dir_abs is not None:
            out = os.path.join(out_dir_abs, match_img1_name + '__' + match_img2_name+'_match_points.png')
            fig.savefig(out, bbox_inches = "tight")
            plt.close()
        else:
            plt.show()
      
def plot_dxdy(ba_dir, out_dir='qc_plots/dxdy'):

    print('plotting dxdy...')

    # create output directory
    out_dir_abs = bare.common.create_dir(out_dir)

    match_csv_list = bare.core.iter_mp_to_csv(ba_dir)

    for i,v in enumerate(match_csv_list):
    
        df = pd.read_csv(v, delimiter=r"\s+")
    
        df['dx'] = df['x2'] - df['x1']
        df['dy'] = df['y2'] - df['y1']
    
        img_match_names = os.path.basename(v).split('.')[0].split('-')[-2]

        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.scatter(df['dx'],df['dy'],color='b',marker='o',facecolor='none',s=10)
        ax.set_aspect('equal')
        ax.set_title('dx/dy\n'+img_match_names)

        out = os.path.join(out_dir_abs,img_match_names+'_dxdy_plot.png')
        fig.savefig(out, bbox_inches = "tight")
        plt.close()


def plot_residuals(ba_dir, 
                   out_dir='qc_plots/residuals', 
                   glacier_shape_fn=None):
    # TODO
    # create interactive plot (html with bokeh maybe) to pan and zoom around
    # when residuals end up in weird places
    '''

    Function to visualize residuals before and after camera alignment during bundle adjustment.

    '''

    print('plotting residuals before and after bundle adjustment...')

    out_dir_abs = bare.common.create_dir(out_dir)

    initial_point_map_csv_fn = glob.glob(os.path.join(ba_dir,
                                                      '*initial_*_pointmap*'))[0]
    final_point_map_csv_fn = glob.glob(os.path.join(ba_dir,
                                                    '*final_*_pointmap*'))[0]
    initial_df = pd.read_csv(initial_point_map_csv_fn,skiprows=[1, 1])
    final_df = pd.read_csv(final_point_map_csv_fn,skiprows=[1, 1])
    initial_gdf = bare.core.ba_pointmap_to_gdf(initial_df)
    final_gdf = bare.core.ba_pointmap_to_gdf(final_df)

    fig, ax = plt.subplots(1,2,figsize=(20,10))
    clim = np.percentile(initial_gdf['mean_residual'].values,(2,98))
    #     clim_final = np.percentile(final_gdf['mean_residual'].values,(2,98))

    initial_gdf.plot(column='mean_residual',
                     ax=ax[0], 
                     cmap='inferno',
                     vmin=clim[0],
                     vmax=clim[1], 
                     legend=True,
                     s=1)

    final_gdf.plot(column='mean_residual',
                   ax=ax[1], 
                   cmap='inferno',
                   vmin=clim[0],
                   vmax=clim[1], 
                   legend=True,
                   s=1)

    if glacier_shape_fn:
        glacier_shape = gpd.read_file(glacier_shape_fn)
        glacier_shape = glacier_shape.to_crs({'init' :'epsg:3857'})
        glacier_shape.plot(ax=ax[0],alpha=0.5)
        glacier_shape.plot(ax=ax[1],alpha=0.5)

    ctx.add_basemap(ax[0])
    ctx.add_basemap(ax[1])

    out = os.path.join(out_dir_abs,'residuals_before_and_after.png')
    fig.savefig(out, bbox_inches = "tight")
    plt.close()

def plot_tsai_camera_positions_before_and_after(ba_dir,
                                                input_cam_dir, 
                                                extension='.tsai',
                                                glacier_shape_fn=None,
                                                out_dir='qc_plots/camera_positions'):

    # TODO
    # Add if then flow to accomodate final.tsai camera models in
    # bundle adjust directory.
    # Add exception to increase extent until stamen tile can be
    # retrieved, even when no glacier shape file provided.

    '''

    Function to plot camera positions in x, y, and z before and after
    bundle adjustment.

    Note: if no glacier shape provided, extent may be insufficient to retrieve
    a stamen tile using contextily.

    '''
    print('plotting tsai camera positions before and after bundle adjustment...')

    out_dir_abs = bare.common.create_dir(out_dir)
                                            
    positions_before_ba = bare.core.extract_tsai_coordinates(input_cam_dir,
                                        extension=extension)
    positions_before_ba = positions_before_ba.to_crs({'init' :'epsg:3857'})

    positions_after_ba = bare.core.extract_tsai_coordinates(ba_dir,
                                        extension=extension)
    positions_after_ba = positions_after_ba.to_crs({'init' :'epsg:3857'}) 

    bare.geospatial.extract_gpd_geometry(positions_before_ba)
    bare.geospatial.extract_gpd_geometry(positions_after_ba)



    # Plot XY
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    positions_before_ba.plot(column='z',
                             ax=ax[0],
                             cmap='viridis',
                             legend=True, 
                             edgecolor='k')
    positions_after_ba.plot(column='z',
                            ax=ax[1], 
                            cmap='viridis',
                            legend=True, 
                            edgecolor='k')
    if glacier_shape_fn:
        glacier_shape = gpd.read_file(glacier_shape_fn)
        glacier_shape = glacier_shape.to_crs({'init' :'epsg:3857'})
        glacier_shape.plot(ax=ax[0],alpha=0.5)
        glacier_shape.plot(ax=ax[1],alpha=0.5)
    
    ctx.add_basemap(ax[0])
    ctx.add_basemap(ax[1])

    ax[0].set_title('xy camera positions before bundle adjust')
    ax[1].set_title('xy camera positions after bundle adjust')

    out = os.path.join(out_dir_abs,
                       'xy_camera_positions_before_and_after_bundle_adjust.png')
    fig.savefig(out, bbox_inches = "tight")
    plt.close()

    # Plot Z
    fig, ax = plt.subplots(1,2,figsize=(20,10),
                           sharey=True)

    ax[0].scatter(positions_before_ba['file'], 
                  positions_before_ba['z'], 
                  color='b')
    ax[0].tick_params(labelrotation=90,axis='x')
    ax[0].set_title('z position before bundle_adjust')
    ax[0].set_xlabel('\nimage')
    ax[0].set_ylabel('height above datum (m)')

    ax[1].scatter(positions_after_ba['file'], 
                  positions_after_ba['z'], 
                  color='b')
    ax[1].tick_params(labelrotation=90,axis='x')
    ax[1].set_title('z position after bundle_adjust')
    ax[1].set_xlabel('\nimage')
    ax[1].set_ylabel('height above datum (m)')

    out = os.path.join(out_dir_abs,
                       'z_camera_positions_before_and_after_bundle_adjust.png')
    fig.savefig(out, bbox_inches = "tight")
    plt.close()

def plot_all_qc_products(ba_dir,
               img_dir,
               input_cam_dir,
               img_extension='8.tif',
               glacier_shape_fn=None):

    plot_tsai_camera_positions_before_and_after(ba_dir,
                                                input_cam_dir,
                                                glacier_shape_fn=glacier_shape_fn)

    plot_ip_over_images(ba_dir,
                        img_dir, 
                        img_extension=img_extension)
                         
    plot_mp_over_images(ba_dir, 
                        img_dir, 
                        img_extension=img_extension)

    plot_dxdy(ba_dir)

    plot_residuals(ba_dir)
    

                             
                             
