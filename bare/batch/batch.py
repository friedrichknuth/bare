import bare.common
import bare.core
import bare.plot


def plot_ip_over_images(ba_dir, 
                        img_dir, 
                        img_extension='.tif',
                        scale=1.0, 
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
    
    for ip_csv_fn in ip_csv_list:
        
        img_file_name, ip_csv_fn = bare.core.parse_image_name_from_ip_file_name(ip_csv_fn, img_dir, img_extension)
        
        bare.plot.ip_plot(img_file_name, ip_csv_fn, out_dir_abs, scale=scale)
    

def plot_mp_over_images(ba_dir, 
                        img_dir, 
                        img_extension='.tif', 
                        scale=1.0,
                        out_dir='qc_plots/match_points'): 

    '''
    Function to visualize match points found between two images.
                        
    All images ending in pattern and contained in img_dir  
    will be extracted. For example, img_extension can be 
    '.tif' or '8.tif'
    '''
                        
    print('plotting match points over images...')

    # create output directory
    out_dir_abs = bare.common.create_dir(out_dir)

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
               img_extension='8.tif',
               glacier_shape_fn=None):

    bare.plot.plot_tsai_camera_positions_before_and_after(ba_dir,
                                                input_cam_dir,
                                                glacier_shape_fn=glacier_shape_fn,
                                                out_dir='qc_plots/camera_positions')

    plot_ip_over_images(ba_dir,
                        img_dir, 
                        img_extension=img_extension)
                         
    plot_mp_over_images(ba_dir, 
                        img_dir, 
                        img_extension=img_extension)

    bare.plot.plot_dxdy(ba_dir,out_dir='qc_plots/dxdy')

    bare.plot.plot_residuals(ba_dir,out_dir='qc_plots/residuals')
    

                             
                             
