import rasterio
from shapely.geometry import Point, Polygon, LineString, mapping
import geopandas as gpd


def extract_gpd_geometry(point_gdf):
    """
    Function to extract x, y, z coordinates and add as columns to input geopandas data frame.
    """
    x = []
    y = []
    z = []
    for i in range(len(point_gdf)): 
        x.append(point_gdf['geometry'].iloc[i].coords[:][0][0])
        y.append(point_gdf['geometry'].iloc[i].coords[:][0][1])
        z.append(point_gdf['geometry'].iloc[i].coords[:][0][2])

    point_gdf['x'] = x
    point_gdf['y'] = y
    point_gdf['z'] = z

def polygon_gdf_to_coordiantes(polygon_gdf):
    geometries = [i for i in polygon_gdf.geometry]
    all_coords = mapping(geometries[0])["coordinates"]
    return all_coords
    
def geotif2polygon(geotif_name):
    """
    Function to return polygon geodataframe matching the extent and coordinate system of the input geotif.
    """

    # read in dem using rasterio
    geotif = rasterio.open(geotif_name)

    # extact total bounds of dem
    bbox = geotif.bounds

    # convert to corner points
    p1 = Point(bbox[0], bbox[3])
    p2 = Point(bbox[2], bbox[3])
    p3 = Point(bbox[2], bbox[1])
    p4 = Point(bbox[0], bbox[1])

    # extract corner coordinates
    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    # convert to polygon
    bb_polygon = Polygon([np1, np2, np3, np4])

    # create geodataframe
    geotif_polygon_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=['geometry'])

    geotif_polygon_gdf.crs = geotif.crs

    return geotif_polygon_gdf


def create_line(point0,point1):
    line = LineString([point0, point1])
    line = gpd.GeoDataFrame(gpd.GeoSeries(line),columns=['geometry'],crs='3857')
    return line
    
def df_points_to_polygon_gdf(df, 
                             lon='lon',
                             lat='lat',
                             z='elevation',
                             crs='4326'):
    vertices = []

    for i in range(len(df)):
        vertex = (df[lon][i], df[lat][i], df[z][i])
        vertices.append(vertex)
        
    polygon = Polygon(vertices)
    polygon_gdf = gpd.GeoDataFrame(gpd.GeoSeries(polygon), 
                                          columns=['geometry'],
                                          crs={'init':'epsg:'+crs}) 

    return polygon_gdf

def extract_polygon_centers(gdf):
    gdf['polygon_center'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
    gdf['polygon_center'] = [coords[0] for coords in gdf['polygon_center']]
    return gdf
    
def df_xyz_coords_to_gdf(df, 
                         lon='lon',
                         lat='lat',
                         z='elevation',
                         crs='4326'):

    geometry = [Point(xyz) for xyz in zip(df[lon], df[lat], df[z])]      
    gdf = gpd.GeoDataFrame(gpd.GeoSeries(geometry), columns=['geometry'], crs={'init':'epsg:'+crs})
    
    return gdf

def wgs_lon_lat_to_epsg_code(lon, lat):
    """
    Function to retreive local UTM EPSG code from WGS84 geographic coordinates.
    """
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code
    
    
def rescale_geotif_to_file(geotif_file_name, scale_factor):
    
    # TODO
    # - Get no data value instead of hard code -9999
    
    # create output file name
    file_path, file_name, file_extension = hsfm.io.split_file(geotif_file_name)
    output_file = os.path.join(file_path, file_name +'_sub'+str(scale_factor)+'.tif')
    
    # downsample array
    img = downsample_geotif_to_array(geotif_file_name, scale_factor)

    # get new shape
    [cols,rows] = img.shape

    # preserve information about original file
    transform = img_ds.GetGeoTransform()
    projection = img_ds.GetProjection()
    nodatavalue = -9999
    # nodatavalue = img_ds.GetNoDataValue() # get attributes property broken in gdal 2.4

    # set gdal GTiff driver
    outdriver = gdal.GetDriverByName("GTiff")

    # create output file, write data and add metadata
    outdata = outdriver.Create(output_file, rows, cols, 1, gdal.GDT_Byte)
    outdata.GetRasterBand(1).WriteArray(img)
    outdata.GetRasterBand(1).SetNoDataValue(nodatavalue)
    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)
    
    
def downsample_geotif_to_array(img_file_name, scale):
    """
    Function to downsample image and return as numpy array.
    """
    img_ds = gdal.Open(img_file_name)
    
    buf_xsize = int(round(img_ds.RasterXSize/scale))
    buf_ysize = int(round(img_ds.RasterYSize/scale))
    
    img = img_ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
    
    return img