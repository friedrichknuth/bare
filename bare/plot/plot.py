import os
import glob
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import Point
from pathlib import Path


import contextily as ctx
import matplotlib.pyplot as plt

import bare.io
import bare.geospatial
import bare.core
import bare.utils

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")

# TODO
# - Organize functions alphabetically.
# - Use class structure to make ctx a global property 
#   instead of repeating condition throughout

def plot_cam(footprint_polygon, camera_positions, basemap='ctx', camera_type='.xml'):
    
    # convert crs for ctx plotting
    if basemap == 'ctx':
        footprint_polygon = footprint_polygon.to_crs(epsg=3857)
        camera_positions = camera_positions.to_crs(epsg=3857)
     
    # extract footprint polygon corner coordinates  
    corners = bare.geospatial.polygon_gdf_to_coordiantes(footprint_polygon)[0][0:-1]
    df = pd.DataFrame(corners)
    df.columns = ['lon','lat','altitude']
    corners = bare.geospatial.df_xyz_coords_to_gdf(df,
                                                   lon='lon',
                                                   lat='lat',
                                                   z='altitude',
                                                   crs='3857')
    
    # if camera_type == '.xml':
    start = camera_positions['geometry'].iloc[0]
    UL = corners['geometry'].iloc[0]
    UR = corners['geometry'].iloc[1]

    end = camera_positions['geometry'].iloc[-1]
    LR = corners['geometry'].iloc[2]
    LL = corners['geometry'].iloc[3]
    
    line0 = bare.geospatial.create_line(start,UL)
    line1 = bare.geospatial.create_line(start,UR)
    line2 = bare.geospatial.create_line(end,LR)
    line3 = bare.geospatial.create_line(end,LL)
    
    return line0, line1, line2, line3


def prepare_footprint(img_file_name, 
                      camera_file, 
                      reference_dem, 
                      output_directory='qc/tmp',
                      cleanup=True,
                      verbose=False):
    
    gcp_file = bare.utils.generate_corner_coordinates(img_file_name, 
                                                      camera_file, 
                                                      reference_dem,
                                                      output_directory=output_directory,
                                                      verbose=verbose)                         
                                 
    footprint_polygon = bare.core.gcp_corners_to_gdf_polygon(gcp_file)
    
    if cleanup == True:
        for p in Path(output_directory).glob("*.tsai"):
            p.unlink()
        for p in Path(output_directory).glob("*.gcp"):
            p.unlink()
    
    if type(footprint_polygon) == gpd.geodataframe.GeoDataFrame:
        return footprint_polygon      
    else:
        print('No footprint generated for ' + img_file_name)
        return

def plot_footprint(img_file_name, camera_file, 
                   reference_dem, output_directory=None,
                   basemap='ctx', cam_on=True,
                   verbose=False):
    # TODO
    # - Add tsai camera plotting.
    
    """
    Function to plot camera footprints.
    """
                                
    out_dir_abs = bare.io.create_dir(output_directory)
    img_base_name = os.path.splitext(os.path.split(img_file_name)[-1])[0]
    cam_extension = os.path.splitext(camera_file)[-1] 
    
    footprint_polygon = prepare_footprint(img_file_name,
                                          camera_file,
                                          reference_dem,
                                          verbose=verbose)
    
    if type(footprint_polygon) == gpd.geodataframe.GeoDataFrame:
        print('Plotting camera footprint.')
        if basemap == 'ctx':
            footprint_polygon = footprint_polygon.to_crs(epsg=3857)
        
        footprint_polygon = bare.geospatial.extract_polygon_centers(footprint_polygon)

        fig, ax = plt.subplots(1,figsize=(10,10))
        footprint_polygon.plot(ax=ax,
                         facecolor="none",
                         edgecolor='b')
    
        if cam_on == True:
            if cam_extension == '.xml':
                ax.set_title('camera footprint and scanner positions')
                camera_positions = bare.core.wv_xml_to_gdf(camera_file)
                if basemap == 'ctx':
                    camera_positions = camera_positions.to_crs(epsg=3857)
                # add coordinates as seperate columns to gdf
                bare.geospatial.extract_gpd_geometry(camera_positions)
                # annotate start and end of aquisition
                plt.annotate(s='start',
                             xy=(camera_positions.iloc[0].x, camera_positions.iloc[0].y),
                             horizontalalignment='center')
                             
                plt.annotate(s='end',
                             xy=(camera_positions.iloc[-1].x, camera_positions.iloc[-1].y),
                             horizontalalignment='center')
                                  
            elif cam_extension == '.tsai':
                ax.set_title('camera footprint and position')
                camera_positions = bare.core.tsai_to_gdf(camera_file)
                if basemap == 'ctx':
                    camera_positions = camera_positions.to_crs(epsg=3857)
                    
                # # Not sure if this is useful to be labeled for tsai.
                # # add coordinates as seperate columns to gdf
                # bare.geospatial.extract_gpd_geometry(camera_positions)
                # # annotate camera position
                # plt.annotate(s='camera position',
                #              xy=(camera_positions.iloc[-1].x, camera_positions.iloc[-1].y),
                #              horizontalalignment='center')
                
            if basemap == 'ctx':
                camera_positions = camera_positions.to_crs(epsg=3857)
            camera_positions.plot(ax=ax,marker='.',color='b')
            
            line0, line1, line2, line3 = plot_cam(footprint_polygon, 
                                                  camera_positions, 
                                                  basemap=basemap, 
                                                  camera_type='.xml')
            line0.plot(ax=ax,color='b')
            line1.plot(ax=ax,color='b')
            line2.plot(ax=ax,color='b')
            line3.plot(ax=ax,color='b')
        
        else:
            ax.set_title('camera footprint')
            
        if basemap == 'ctx':
            ctx.add_basemap(ax)

        for idx, row in footprint_polygon.iterrows():
            plt.annotate(s=row['file_name'],
                         xy=row['polygon_center'],
                         horizontalalignment='center')
             
        

        if out_dir_abs is not None:
            out = os.path.join(out_dir_abs, img_base_name+'_footprint.png')
            fig.savefig(out, bbox_inches = "tight")
            plt.close()
        else:
            plt.show()
    else:
        pass
    
def ip_plot(img_file_name, ip_csv_fn, out_dir_abs=None, scale=1.0):
    # TODO
    # - Take .vwip file as input and generate csv if not exists.
    
    # create output directory
    out_dir_abs = bare.io.create_dir(out_dir_abs)
    
    img_base_name = os.path.splitext(os.path.split(img_file_name)[-1])[0]
    
    # load interest points
    df = pd.read_csv(ip_csv_fn, delimiter=r"\s+")

    # read in image data and scale accordingly
    img_ds = gdal.Open(img_file_name)
    buf_xsize = None
    buf_ysize = None
    if scale > 1:
        buf_xsize = int(round(img_ds.RasterXSize/scale))
        buf_ysize = int(round(img_ds.RasterYSize/scale))
        df['x1'] /= scale
        df['y1'] /= scale
    img = img_ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)

    # plot the data
    fig, ax = plt.subplots(1,figsize=(10,10))
    clim = np.percentile(img, (2,98))
    
    ax.scatter(df['x1'],df['y1'],
               color='r',
               marker='o',
               facecolor='none',
               s=10)
    ax.imshow(img, clim=clim, cmap='gray')
    ax.set_aspect('equal')
    ax.set_title('interest points\n'+ img_base_name)
    plt.tight_layout()
    
    # visualize or write to file if out_dir_abs provided
    if out_dir_abs is not None:
        out = os.path.join(out_dir_abs, img_base_name+'_interest_points.png')
        fig.savefig(out, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()
        
    # drop image data from memory
    img = None
    img_ds = None


def mp_plot(img1_fn, img2_fn, match_csv_fn, out_dir_abs=None, scale=1.0):
    # TODO
    # - Take .match file as input and generate csv if not exists.
    
    '''
    Function to generate plots fro two images and their corresponding match csv.
    '''
    # create output directory
    out_dir_abs = bare.io.create_dir(out_dir_abs)
    
    # load match file into DataFrame
    df = pd.read_csv(match_csv_fn, delimiter=r"\s+")
    
    # extract short filenames 
    match_img1_name = os.path.splitext(os.path.split(img1_fn)[-1])[0]
    match_img2_name = os.path.splitext(os.path.split(img2_fn)[-1])[0]

    fig, ax = plt.subplots(1,2,figsize=(18,10))
    fig.suptitle('match points\n' + os.path.split(match_csv_fn)[-1])

    buf_xsize = None
    buf_ysize = None

    img1_ds = gdal.Open(img1_fn)

    # handle scaling of input images using ReadAsArray buffer sizes
    if scale > 1:
        buf_xsize = int(round(img1_ds.RasterXSize/scale))
        buf_ysize = int(round(img1_ds.RasterYSize/scale))
        df['x1'] /= scale
        df['y1'] /= scale

    img1 = img1_ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
    clim = np.percentile(img1, (2,98))
    ax[0].scatter(df['x1'], df['y1'], color='r', marker='o', facecolor='none', s=10)
    ax[0].imshow(img1, clim=clim, cmap='gray')
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


def plot_dxdy(ba_dir, output_directory='qc_plots/dxdy'):
    # TODO
    # - Break this into a dedicated function and move iteration to batch
    print('plotting dxdy...')

    # create output directory
    out_dir_abs = bare.io.create_dir(output_directory)

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


def plot_residuals(ba_dir, output_directory=None, ascending=True, basemap='ctx', glacier_shape_fn=None):
    # TODO
    # - Create interactive plot (html with bokeh maybe) to pan and zoom around
    # when residuals end up in weird places.
    # - Allow for plotting over orthoimage mosaic.
    # - Add condition to only sharex sharey if extent similar enough,
    # else filter and report gross outliers.
    '''

    Function to visualize residuals before and after camera alignment during bundle adjustment.

    '''
    
    print('Plotting residuals before and after bundle adjustment...')
    
    # create output directory
    out_dir_abs = bare.io.create_dir(output_directory)
    
    initial_point_map_csv_fn = glob.glob(os.path.join(ba_dir,'*initial_*_pointmap*csv'))[0]
    final_point_map_csv_fn = glob.glob(os.path.join(ba_dir,'*final_*_pointmap*csv'))[0]

    initial_df = pd.read_csv(initial_point_map_csv_fn,skiprows=[1, 1])
    final_df = pd.read_csv(final_point_map_csv_fn,skiprows=[1, 1])

    # convert to GeoDataFrame
    initial_gdf = bare.core.ba_pointmap_to_gdf(initial_df, ascending=ascending)
    final_gdf = bare.core.ba_pointmap_to_gdf(final_df, ascending=ascending)

    # TODO
    # - Filter outliers on 'after' plot to make sharex, sharey more useful. Right now the
    #   extent may reach accross the globe. See examples.
    # fig, ax = plt.subplots(1,2,figsize=(10,5), sharex=True, sharey=True)
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    clim = np.percentile(initial_gdf['mean_residual'].values,(2,98))

    # add plots to show number of images per match point
    initial_gdf.plot(column='mean_residual',
                     ax=ax[0], 
                     cmap='inferno',
                     vmin=clim[0],
                     vmax=clim[1], 
                     legend=True,
                     s=initial_gdf['num_observations']/initial_gdf['num_observations'].max())

    final_gdf.plot(column='mean_residual',
                   ax=ax[1], 
                   cmap='inferno',
                   vmin=clim[0],
                   vmax=clim[1], 
                   legend=True,
                   s=final_gdf['num_observations']/final_gdf['num_observations'].max())

    ax[0].set_title('Before bundle adjustment (n=%i)' % initial_df.shape[0])
    ax[1].set_title('After bundle adjustment (n=%i)' % final_df.shape[0])
    ax[0].set_facecolor('0.5')
    ax[1].set_facecolor('0.5')

    if glacier_shape_fn:
        glacier_shape = gpd.read_file(glacier_shape_fn)
        glacier_shape = glacier_shape.to_crs({'init' :'epsg:3857'})
        glacier_shape.plot(ax=ax[0],alpha=0.5)
        glacier_shape.plot(ax=ax[1],alpha=0.5)

    if basemap == 'ctx':
        ctx.add_basemap(ax[0])
        ctx.add_basemap(ax[1])

    plt.suptitle("Match point mean residuals (m)")
    
    if out_dir_abs is not None:
        out = os.path.join(out_dir_abs,'ba_match_residuals_before_and_after.jpg')
        fig.savefig(out, quality=85, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_tsai_camera_positions_before_and_after(ba_dir,
                                                input_cam_dir, 
                                                extension='.tsai',
                                                output_directory=None):

    # TODO
    # - Reduce duplicate plotting code with plot_z_camera_positions() and 
    #   plot_xy_camera_positions() by passing axis object to this function. 
    # - Add option to pass custom extent

    """
    Function to plot camera positions in x, y, and z before and after
    bundle adjustment.

    """

    output_directory = bare.io.create_dir(output_directory)
    
    positions_before_ba = camera_models_to_ctx_df(input_cam_dir,
                                                  extension=extension)
                                            
    positions_after_ba = camera_models_to_ctx_df(ba_dir,
                                                 extension=extension)

    # plot XY
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    positions_before_ba.plot(column='z',
                             ax=ax[0],
                             cmap='winter',
                             legend=True, 
                             edgecolor='k')
    positions_after_ba.plot(column='z',
                            ax=ax[1], 
                            cmap='winter',
                            legend=True, 
                            edgecolor='k')

    
    add_ctx_basemap(ax[0],15)
    add_ctx_basemap(ax[1],15)

    ax[0].set_title('xy camera positions before bundle adjust')
    ax[1].set_title('xy camera positions after bundle adjust')

    if output_directory != None:
        out = os.path.join(output_directory,'xy_camera_positions_before_and_after_bundle_adjust.png')
        fig.savefig(out, quality=85, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show() 

    # plot Z
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
    
    if output_directory != None:
        out = os.path.join(output_directory,'z_camera_positions_before_and_after_bundle_adjust.png')
        fig.savefig(out, quality=85, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show() 

    
def add_ctx_basemap(ax, zoom, url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))
    
def plot_xy_camera_positions(camera_positions,
                             extension='.tsai',
                             output_directory=None,
                             title= 'xy camera positions'):
    
    output_directory = bare.io.create_dir(output_directory)
    
    fig, ax = plt.subplots(figsize=(10,10))
    camera_positions.plot(column='z',
                             ax=ax,
                             cmap='winter',
                             legend=True, 
                             edgecolor='k')
    add_ctx_basemap(ax,15)
    
    ax.set_title(title)
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.tick_params(labelrotation=70,axis='x')
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')

    if output_directory != None:
        out = os.path.join(output_directory,'camera_xy_positions.jpg')
        fig.savefig(out, quality=85, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
                             
def plot_z_camera_positions(camera_positions,
                            extension='.tsai',
                            output_directory=None,
                            title='z camera positions'):
    
    output_directory = bare.io.create_dir(output_directory)
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(camera_positions['file'],
               camera_positions['z'],
               color='b')
    
    ax.set_title(title)
    ax.tick_params(labelrotation=90,axis='x')
    ax.set_xlabel('\nimage')
    ax.set_ylabel('height above datum (m)')

    if output_directory != None:
        out = os.path.join(output_directory,'camera_xy_positions.jpg')
        fig.savefig(out, quality=85, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def camera_models_to_ctx_df(camera_model_directory,
                            extension='.tsai'):
    
    camera_positions = bare.core.iter_extract_tsai_coordinates(camera_model_directory,
                                                                  extension=extension)
    camera_positions = camera_positions.to_crs({'init' :'epsg:3857'})
    bare.geospatial.extract_gpd_geometry(camera_positions)
    return camera_positions