import geopandas as gpd

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
