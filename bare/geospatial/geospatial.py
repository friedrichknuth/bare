import rasterio
from shapely.geometry import Point, Polygon
import geopandas as gpd


def dem2polygon(dem_file_name):
    '''
    Function to return polygon geodataframe matching the extent and coordinate system of the input DEM.
    '''

    # read in dem using rasterio
    dem = rasterio.open(dem_file_name)

    # extact total bounds of dem
    bbox = dem.bounds

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
    dem_polygon_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=['geometry'])

    dem_polygon_gdf.crs = dem.crs

    return dem_polygon_gdf


def extract_gpd_geometry(gdf):
    '''
    Function to extract x, y, z coordinates from input geopandas data frame.
    '''
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


