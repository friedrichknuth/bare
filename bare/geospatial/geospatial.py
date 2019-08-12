import rasterio
from shapely.geometry import Point, Polygon
import geopandas as gpd


def extract_gpd_geometry(gdf):
    """
    Function to extract x, y, z coordinates and add as columns to input geopandas data frame.
    """
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



def df_points_to_polygon_gdf(df, 
                             lon='lon',
                             lat='lat',
                             elevation='elevation',
                             crs='4326'):
    vertices = []

    for i in range(len(df)):
        vertex = (df[lon][i], df[lat][i], df[elevation][i])
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