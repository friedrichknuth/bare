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

from asp_ba_eval import BasicFunctions
from asp_ba_eval import Geospatial

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")



#TODO
'''
- break class into plotting and wrangling classes
- write converted csv files to a seperate folder
- hide functions with preceeding _ from inheritance
- add docstrings to functions
- need to handle numpass 0 when no clean.match file is created to parse matched 
  images from match file file name.
'''



class BundleAdjustRunEvaluation:
    
    def iter_ip_to_csv(ba_dir):
    #     print('converting interest point files (vwip) to csv...')
        vwips = sorted(glob.glob(ba_dir+"*.vwip"))
        ip_csv_list = []
        for filename in vwips:
            fn = BundleAdjustRunEvaluation.write_ip_to_csv(filename)
            ip_csv_list.append(fn)
        return ip_csv_list
        

    def iter_mp_to_csv(ba_dir):
    #     print('converting match point files to csv...')
        matches = sorted(glob.glob(ba_dir+"*clean.match"))
        if matches:
            print('    processing clean.match files only.')
        else:
            print('    no clean.match files found.')
            matches = sorted(glob.glob(ba_dir+"*.match"))
            if matches:
                print('    processing .match files.')
            else:
                print('    no .match files found.')
                sys.exit(1)
        match_csv_list = []
        for filename in matches:
            fn = BundleAdjustRunEvaluation.write_mp_to_csv(filename)
            match_csv_list.append(fn)
        return match_csv_list
        

    def parse_image_names_from_match_file_name(match_file, img_list):
        '''
    
        Function to parse out image file names from match files.
        Image file names cannot have '-' in them, else this will break.

        '''
    
        # Get image extension and image base path
        img_ext = '.' + os.path.basename(img_list[0]).split('.')[-1]
        img_base_path = os.path.dirname(img_list[0])
    
        # Extract image pair file names from ASP match file name
        # For example e.g. split ../run-v2_sub8__v3_sub8-clean.csv 
        # into v2_sub8 and v3_sub8.
        # Image names cannot have a '-' in them else this will break.
        # Needs better solution.
        # Need to handle numpass 0 when no clean.match file is created.
    
        match_img1_name = os.path.basename(match_file).split('.')[0].split('-')[-2].split('__')[0]
        img1_file_name = os.path.join(img_base_path, match_img1_name+img_ext)

        match_img2_name = os.path.basename(match_file).split('.')[0].split('-')[-2].split('__')[1]
        img2_file_name = os.path.join(img_base_path, match_img2_name+img_ext)
    
        return img1_file_name, img2_file_name , match_img1_name, match_img2_name

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
            im1_ip = [BundleAdjustRunEvaluation.read_ip_record(mf) for i in range(size1)]
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
            im1_ip = [BundleAdjustRunEvaluation.read_ip_record(mf) for i in range(size1)]
            im2_ip = [BundleAdjustRunEvaluation.read_ip_record(mf) for i in range(size2)]
            for i in range(len(im1_ip)):
                out.write('{} {} {} {}\n'.format(im1_ip[i][0], 
                                                 im1_ip[i][1], 
                                                 im2_ip[i][0], 
                                                 im2_ip[i][1]))
        return filename_out

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
        out_dir_abs = BasicFunctions.create_dir(out_dir)
    
        # convert interest points in binary vwip files to csv.
        ip_csv_list = BundleAdjustRunEvaluation.iter_ip_to_csv(ba_dir)
    
        # get images list
        img_list = sorted(glob.glob(img_dir+'*'+img_extension))
    
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
        out_dir_abs = BasicFunctions.create_dir(out_dir)
    
        # convert .match files to csv
        match_csv_list = BundleAdjustRunEvaluation.iter_mp_to_csv(ba_dir)
    
        # get images list
        img_list = sorted(glob.glob(img_dir+'*'+img_extension))

        for i,v in enumerate(match_csv_list):
            tmp = BundleAdjustRunEvaluation.parse_image_names_from_match_file_name(v,img_list)
            img1_file_name= tmp[0]
            img2_file_name = tmp[1]
            match_img1_name= tmp[2]
            match_img2_name = tmp[3]

            df = pd.read_csv(v, delimiter=r"\s+")
        
            img1 = gdal.Open(img1_file_name)
            img1 = np.array(img1.ReadAsArray())        

            img2 = gdal.Open(img2_file_name)
            img2 = np.array(img2.ReadAsArray()) 
        
        
            fig, ax = plt.subplots(1,2,figsize=(20,10))
            fig.suptitle('match points\n'+ 
                         match_img1_name +' and '+
                         match_img2_name)
        
            ax[0].scatter(df['x1'],df['y1'],color='r',marker='o',facecolor='none',s=10)
            ax[0].imshow(img1,cmap='gray')
            ax[0].set_title(match_img1_name)
        
            ax[1].scatter(df['x2'],df['y2'],color='r',marker='o',facecolor='none',s=10)
            ax[1].imshow(img2,cmap='gray')
            ax[1].set_title(match_img2_name)
        
            out = os.path.join(out_dir_abs,
                               match_img1_name + '__' + match_img2_name+'_match_points.png')
            fig.savefig(out, bbox_inches = "tight")
            plt.close()
          
    def plot_dxdy(ba_dir, out_dir='qc_plots/dxdy'):
    
        print('plotting dxdy...')
    
        # create output directory
        out_dir_abs = BasicFunctions.create_dir(out_dir)
    
        match_csv_list = BundleAdjustRunEvaluation.iter_mp_to_csv(ba_dir)
    
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
    
        out_dir_abs = BasicFunctions.create_dir(out_dir)
    
        initial_point_map_csv_fn = glob.glob(os.path.join(ba_dir,
                                                          '*initial_*_pointmap*'))[0]
        final_point_map_csv_fn = glob.glob(os.path.join(ba_dir,
                                                        '*final_*_pointmap*'))[0]
        initial_df = pd.read_csv(initial_point_map_csv_fn,skiprows=[1, 1])
        final_df = pd.read_csv(final_point_map_csv_fn,skiprows=[1, 1])
        initial_gdf = BundleAdjustRunEvaluation.ba_pointmap_to_gdf(initial_df)
        final_gdf = BundleAdjustRunEvaluation.ba_pointmap_to_gdf(final_df)
    
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

        out_dir_abs = BasicFunctions.create_dir(out_dir)
                                                
        positions_before_ba = BundleAdjustRunEvaluation.extract_tsai_coordinates(input_cam_dir,
                                            extension=extension)
        positions_before_ba = positions_before_ba.to_crs({'init' :'epsg:3857'})
    
        positions_after_ba = BundleAdjustRunEvaluation.extract_tsai_coordinates(ba_dir,
                                            extension=extension)
        positions_after_ba = positions_after_ba.to_crs({'init' :'epsg:3857'}) 
    
        Geospatial.extract_gpd_geometry(positions_before_ba)
        Geospatial.extract_gpd_geometry(positions_after_ba)
    
    
    
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
    
        ax[0].set_title('camera positions before bundle adjust')
        ax[1].set_title('camera positions after bundle adjust')
    
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
        ax[0].set_title('before bundle_adjust')
        ax[0].set_xlabel('\nimage')
        ax[0].set_ylabel('height above datum (m)')

        ax[1].scatter(positions_after_ba['file'], 
                      positions_after_ba['z'], 
                      color='b')
        ax[1].tick_params(labelrotation=90,axis='x')
        ax[1].set_title('after bundle_adjust')
        ax[1].set_xlabel('\nimage')
        ax[1].set_ylabel('height above datum (m)')
    
        out = os.path.join(out_dir_abs,
                           'z_camera_positions_before_and_after_bundle_adjust.png')
        fig.savefig(out, bbox_inches = "tight")
        plt.close()
    
    def run_all_qc(ba_dir,
                   img_dir,
                   input_cam_dir,
                   img_extension='8.tif',
                   glacier_shape_fn=None):
    
        BundleAdjustRunEvaluation.plot_ip_over_images(ba_dir,
                            img_dir, 
                            img_extension=img_extension)
                             
        BundleAdjustRunEvaluation.plot_mp_over_images(ba_dir, 
                            img_dir, 
                            img_extension=img_extension)
    
        BundleAdjustRunEvaluation.plot_dxdy(ba_dir)
    
        BundleAdjustRunEvaluation.plot_residuals(ba_dir)
    
        BundleAdjustRunEvaluation.plot_tsai_camera_positions_before_and_after(ba_dir,
                                                    input_cam_dir,
                                                    glacier_shape_fn=glacier_shape_fn)
                             
                             
