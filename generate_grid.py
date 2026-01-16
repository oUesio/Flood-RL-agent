import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterstats import zonal_stats

# Creates overall grid shapefile for watercourse, flood risk, elevation, impervious surface area, historic flood, road, hospital locations
# recrop area using boundary shapefile for handling updated shapefile/tiff files (GM_shapefile/CAUTH_MAY_2025_EN_BSC.shp)
# ignore cell if on or past boundary
# cells of size 1km x 1km
# centroids used for distance calculations
''' 
data/watercourse/Watercourse.shp
- closest watercourse to centre of cell (m)
- density of watercourses in cell
data/flood_risk/rofsw_4bandPolygon/merged_rofsw_4bandPolygon.shp
- confidence-weighted average risk
data/elevation.tif
- average elevation in cell (m)
data/impervious_surface.tif
- fraction of impervious surface in cell
data/historic_flood_map/Historic_Flood_MapPolygon.shp
- if cell has been flooded in the past
data/road/RoadLink.shp
- closest major road to centre of cell (km)
- density of roads in cell
data/hospital_locations/hospital_locations.shp
- distance to nearest hospital (km)
'''

CELL_SIZE = 1000  # metres
CELL_AREA_M2 = CELL_SIZE ** 2
CELL_AREA_KM2 = CELL_AREA_M2 / 1e6

WATERCOURSE = "data/watercourse/Watercourse.shp"
FLOOD_RISK = "data/flood_risk/rofsw_4bandPolygon/merged_rofsw_4bandPolygon.shp"
ELEVATION = "data/elevation.tif"
IMPERVIOUS = "data/impervious.tif"
HISTORIC_FLOOD = "data/historic_flood_map/Historic_Flood_MapPolygon.shp"
ROAD = "data/road/RoadLink.shp"
HOSPITAL = "data/hospital_locations/hospital_locations.shp"
BOUNDARY = "GM_shapefile/CAUTH_MAY_2025_EN_BSC.shp"

OUTPUT = "data/grid/grid.shp"

RISK_SCORES = {"Very low": 1, "Low": 2, "Medium": 3, "High": 4}

'''RISK_SCORES = {
    "Very Low": 0.0005, # (0 + 0.1)/2 
    "Low": 0.00055, # (0.1 + 1)/2
    "Medium": 0.0215, # (1 + 3.3)/2
    "High": 0.033 # (3.3 + 3.3)/2
}'''

def build_grid(boundary):
    minx, miny, maxx, maxy = boundary.total_bounds

    xs = np.arange(minx, maxx, CELL_SIZE)
    ys = np.arange(miny, maxy, CELL_SIZE)

    grid = gpd.GeoDataFrame(
        geometry=[box(x, y, x + CELL_SIZE, y + CELL_SIZE) for x in xs for y in ys],
        crs="EPSG:27700"
    )

    grid = grid[grid.geometry.within(boundary.geometry.union_all())]
    grid["cell_id"] = grid.index
    return grid

def line_density(grid, lines, colname, cell_size):
    grid[colname] = 0.0
    cell_area_km2 = (cell_size * cell_size) / 1e6  # km²
    for i, cell in grid.geometry.items():
        inter = lines.geometry.intersection(cell)
        length_m = sum(geom.length for geom in inter if not geom.is_empty)
        grid.at[i, colname] = (length_m / 1000) / cell_area_km2
    return grid


def nearest_distance(grid, targets, colname, km=True):
    centroids = grid.copy()
    centroids["geometry"] = centroids.geometry.centroid

    nearest = gpd.sjoin_nearest(
        centroids,
        targets[["geometry"]],
        how="left",
        distance_col=colname
    )

    dist = nearest.groupby(nearest.index)[colname].first()
    if km:
        dist = dist / 1000

    grid[colname] = dist
    return grid


def flood_risk_score(grid, risk):
    inter = gpd.overlay(
        grid[["cell_id", "geometry"]],
        risk,
        how="intersection"
    )

    if inter.empty:
        grid["risk_score"] = 0.0
        return grid

    inter["area"] = inter.geometry.area
    inter["risk_value"] = inter["risk_band"].map(RISK_SCORES).fillna(0)
    inter["conf_weight"] = inter["confidence"] / 10

    inter["num"] = inter["area"] * inter["risk_value"] * inter["conf_weight"]
    inter["den"] = inter["area"] * inter["conf_weight"]

    agg = inter.groupby("cell_id")[["num", "den"]].sum()
    grid["risk_score"] = (agg["num"] / agg["den"]).reindex(grid.cell_id).fillna(0)
    return grid


def zonal_mean(grid, raster_path, colname):
    stats = zonal_stats(
        grid.geometry,
        raster_path,
        stats="mean",
        nodata=0
    )
    grid[colname] = [s["mean"] for s in stats]
    return grid


def zonal_fraction_nonzero(grid, raster_path, colname):
    stats = zonal_stats(
        grid.geometry,
        raster_path,
        stats=["count", "nodata"],
        add_stats={"nonzero": lambda x: np.count_nonzero(x)}
    )
    grid[colname] = [
        s["nonzero"] / s["count"] if s["count"] else 0
        for s in stats
    ]
    return grid


def historic_flood_flag(grid, historic):
    inter = gpd.overlay(
        grid[["cell_id", "geometry"]],
        historic,
        how="intersection"
    )
    inter["area"] = inter.geometry.area
    flooded = inter.groupby("cell_id")["area"].sum()

    grid["historic"] = (
        flooded / CELL_AREA_M2 > 0.5
    ).reindex(grid.cell_id).fillna(False).astype(int)

    return grid


# LOAD DATA
print("Loading data...")

water = gpd.read_file(WATERCOURSE).set_crs(epsg=27700, allow_override=True)
risk = gpd.read_file(FLOOD_RISK).set_crs(epsg=27700, allow_override=True)
historic = gpd.read_file(HISTORIC_FLOOD).set_crs(epsg=27700, allow_override=True)
road = gpd.read_file(ROAD).set_crs(epsg=27700, allow_override=True)
hospital = gpd.read_file(HOSPITAL).set_crs(epsg=27700, allow_override=True)
boundary = gpd.read_file(BOUNDARY).set_crs(epsg=27700, allow_override=True)

# GRID
print("Building grid...")
grid = build_grid(boundary)
grid.to_file("data/grid/grid_step_01.shp")

# LINE DENSITIES
print("Calculating watercourse density...")
grid = line_density(grid, water, "water_dens", CELL_SIZE)

print("Calculating road density...")
grid = line_density(grid, road, "road_dens", CELL_SIZE)

# NEAREST DISTANCES
print("Calculating nearest watercourse distance...")
grid = nearest_distance(grid, water, "water_dist", km=False)

print("Calculating nearest hospital distance...")
grid = nearest_distance(grid, hospital, "hospital")

print("Calculating nearest major road distance...")
major_roads = road[road["function"].isin(["A Road", "Motorway"])]
grid = nearest_distance(grid, major_roads, "road_dist")

# FLOOD RISK
print("Calculating flood risk score...")
grid = flood_risk_score(grid, risk)

# RASTER FEATURES
print("Calculating elevation...")
grid = zonal_mean(grid, ELEVATION, "elevation")

print("Calculating impervious fraction...")
grid = zonal_fraction_nonzero(grid, IMPERVIOUS, "impervious")

# HISTORIC FLOOD
print("Calculating historic flood flag...")
grid = historic_flood_flag(grid, historic)

# SAVE OUTPUT
cols = [
    "water_dens", "water_dist", "risk_score", "elevation",
    "impervious", "historic", "road_dens", "road_dist", "hospital", "geometry"
]

grid[cols].to_file(OUTPUT)