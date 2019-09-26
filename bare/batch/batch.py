import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from pathlib import Path
import shutil

import bare.io
import bare.core
import bare.plot


def plot_footprints(cam_dir, 
                    img_dir, 
                    reference_dem,
                    output_directory='qc/footprints',
                    show=False,
                    verbose=False,
                    basemap='ctx',
                    img_file_extension='.tif',
                    cam_file_extension='.tsai',
                    cleanup=True):
                    
    """
    Function to plot image footprints from images and camera files
    """
    # TODO
    # - This should use bare.plot.plot_footprint. Needs to be reconciled.
                                  
    out_dir_abs = bare.io.create_dir(output_directory)
    
    cam_list = sorted(glob.glob(os.path.join(cam_dir, '*' + cam_file_extension)))
    img_list = sorted(glob.glob(os.path.join(img_dir, '*' + img_file_extension)))
        
    df = pd.DataFrame()
    
    for cam_file in cam_list:
        
        cam_file_path, cam_file_base_name, cam_file_extension = bare.io.split_file(cam_file)
        
        for img_file in img_list:
            img_file_path, img_file_base_name, img_file_extension = bare.io.split_file(img_file)
            
            if img_file_base_name in cam_file_base_name:
                img_file_name = img_file
        
        
        if verbose==True:
            print('\nGenerating footprint for ' + img_base_name + img_file_extension +'.')
        
        footprint = bare.plot.prepare_footprint(img_file_name, 
                                                cam_file, 
                                                reference_dem,
                                                cleanup=cleanup)
                 
        if footprint is not None:
            crs = footprint.crs
            df = df.append(footprint)
        else:
            continue
            
    if not df.empty:
        # plot returned footprints
        footprints = gpd.GeoDataFrame(df, columns=['file_name','geometry'], crs=crs)
        if basemap == 'ctx':
            footprints = footprints.to_crs(epsg=3857)
        
        footprints = bare.geospatial.extract_polygon_centers(footprints)
        
        
        fig, ax = plt.subplots(1,figsize=(10,10))

        footprints.plot(ax=ax,
                color='b',
                edgecolor='b',
                alpha=0.1)

        for idx, row in footprints.iterrows():
            plt.annotate(s=row['file_name'],
                         xy=row['polygon_center'],
                         horizontalalignment='center')

        # # alternative plotting approaches
        # footprints.plot(ax=ax,
        #         facecolor='none',
        #         edgecolor='b')
        #
        # footprints.plot(ax=ax,
        #         column='file_name',
        #         legend=True,
        #         facecolor='none',
        #         edgecolor='b',
        #         legend_kwds={'bbox_to_anchor': (1.41, 1)})


        bare.plot.add_ctx_basemap(ax)
        ax.set_title('camera footprints')

        # visualize or write to file if out_dir_abs provided
        if show == False:
            out = os.path.join(out_dir_abs, 'footprints.png')
            fig.savefig(out, bbox_inches = "tight")
            plt.close()

        else:
            plt.show()
            if cleanup == True:
                shutil.rmtree(out_dir_abs)
            

            

def plot_ip_over_images(ba_dir, 
                        img_dir, 
                        img_extension='.tif',
                        scale=1.0, 
                        output_directory='qc/interest_points'):

    '''
    Function to visualize interest points in an image.
    All images ending in img_extension pattern will be used. 
    For example, img_extension can be '.tif' or '8.tif'
    '''

    print('plotting interest points over images...')

    # create output directory
    out_dir_abs = bare.io.create_dir(output_directory)

    # convert interest points in binary vwip files to csv.
    ip_csv_list = bare.core.iter_ip_to_csv(ba_dir)
    
    for ip_csv_fn in ip_csv_list:
        
        img_file_name, ip_csv_fn = bare.core.parse_image_name_from_ip_file_name(ip_csv_fn, img_dir, img_extension)
        
        bare.plot.ip_plot(img_file_name, ip_csv_fn, out_dir_abs, scale=scale)
    

def plot_mp_over_images(ba_dir, 
                        img_dir, 
                        img_extension='.tif', 
                        scale=1.0,
                        output_directory='qc/match_points'): 

    '''
    Function to visualize match points found between two images.
                        
    All images ending in pattern and contained in img_dir  
    will be extracted. For example, img_extension can be 
    '.tif' or '8.tif'
    '''
                        
    print('plotting match points over images...')

    # create output directory
    out_dir_abs = bare.io.create_dir(output_directory)

    # convert .match files to csv
    match_csv_list = bare.core.iter_mp_to_csv(ba_dir)

    # get images list
    for match_csv_fn in match_csv_list:
        
        # extract image pairs
        img1_file_name, img2_file_name = \
                bare.core.parse_image_names_from_match_file_name(match_csv_fn, img_dir, img_extension)
                
        bare.plot.mp_plot(img1_file_name, img2_file_name, match_csv_fn, out_dir_abs, scale=scale)


def plot_all_qc_products(ba_dir,
                         img_dir,
                         input_cam_dir,
                         img_extension='8.tif'):

    bare.plot.plot_tsai_camera_positions_before_and_after(ba_dir,
                                                          input_cam_dir,
                                                          output_directory='qc/camera_positions')

    plot_ip_over_images(ba_dir,
                        img_dir, 
                        img_extension=img_extension)
                         
    plot_mp_over_images(ba_dir, 
                        img_dir, 
                        img_extension=img_extension)

    bare.plot.plot_dxdy(ba_dir,output_directory='qc/dxdy')

    bare.plot.plot_residuals(ba_dir,output_directory='qc/residuals')
    

                             
                             
