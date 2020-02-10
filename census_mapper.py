import matplotlib.pyplot as plt

from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO, StringIO
import shapefile
import geopandas as gpd
from shapely.geometry import shape
from osgeo import osr
import pandas as pd
import requests
from shapely.geometry import Point
from numpy.random import RandomState, uniform
import numpy as np
from census import Census
from us import states
import os
from shapely import wkt


def gen_random_points_poly(poly, num_points, seed=None):
    """
    Returns a list of N randomly generated points within a polygon.

    """
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    i = 0
    while len(points) < num_points:
        s = RandomState(seed + i) if seed else RandomState(seed)
        random_point = Point([s.uniform(min_x, max_x), s.uniform(min_y, max_y)])
        if random_point.within(poly):
            points.append(random_point)
        i += 1
    return points


def gen_points_in_gdf_polys(geometry, values, points_per_value=None, seed=None):
    """
    Take a GeoSeries of Polygons along with a Series of values and returns randomly generated points within
    these polygons. Optionally takes a "points_per_value" integer which indicates the number of points that
    should be generated for each 1 value.
    """
    if points_per_value:
        new_values = (values / points_per_value).astype(int)
    else:
        new_values = values
    new_values = new_values[new_values > 0]

    data = new_values.to_frame(name="vals")
    g = gpd.GeoDataFrame(data=data, geometry=geometry)

    a = g.apply(
        lambda row: tuple(gen_random_points_poly(row["geometry"], row["vals"], seed)), 1
    )
    b = gpd.GeoSeries(a.apply(pd.Series).stack(), crs=geometry.crs)
    b.name = "geometry"
    return b


def zip_shp_to_gdf(zip_file_name):
    """
    Returns a GeoDataFrame from a URL for a zipped Shapefile
    """
    zipfile = ZipFile(BytesIO(urlopen(zip_file_name).read()))

    filenames = [
        y
        for y in sorted(zipfile.namelist())
        for ending in ["dbf", "prj", "shp", "shx"]
        if y.endswith(ending)
    ]

    dbf, prj, shp, shx = [BytesIO(zipfile.read(filename)) for filename in filenames]
    r = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)

    attributes, geometry = [], []
    field_names = [field[0] for field in r.fields[1:]]
    for row in r.shapeRecords():
        geometry.append(shape(row.shape.__geo_interface__))
        attributes.append(dict(list(zip(field_names, row.record))))

    proj4_string = osr.SpatialReference(
        str(prj.read(), encoding="utf-8")
    ).ExportToProj4()
    gdf = gpd.GeoDataFrame(data=attributes, geometry=geometry, crs=proj4_string)
    return gdf


def get_census_data(year, dataset, geography, area, variables, variable_labels=None):
    c = Census(os.environ["CENSUS_API_KEY"], year=year)
    data = c.acs5.get(
        variables,
        geo={"for": "tract:*", "in": "state:{} county:001".format(states.ID.fips)},
    )

    # read json into pandas dataframe, specifying first row as column names
    df = pd.DataFrame(columns=data[0], data=data[1:])

    # identify geography fields - concatenate them into a fips code to be set as index and then delete them
    geo_fields = [x for x in df.columns if x not in ["NAME"] + variables]
    df.index = df[geo_fields].apply(lambda row: "".join(map(str, row)), 1)
    df.index.name = "FIPS"
    df = df.drop(geo_fields, 1)

    if variable_labels:
        df = df.rename(columns=dict(list(zip(variables, variable_labels))))

    # convert data numeric
    df = df.applymap(lambda x: pd.to_numeric(x, errors="ignore"))
    return df


def gen_count_dot_density_map(
    county,
    pts_per_person=300,
    epsg=2163,
    seed=10,
    dot_transparency=0.5,
    figsize=(18, 10),
    ax=None,
    legend=True,
):
    """
    Wraps previous functions and generates population dot density maps for a specified county by race

    """
    # read in fips to county name relationship file
    fips = pd.read_csv(
        "https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt",
        header=None,
        dtype={1: np.object, 2: np.object},
    )
    fips["name"] = fips[3] + ", " + fips[0]
    fips["fips"] = fips[1] + fips[2]

    # get name from fips if fips specified
    if county.isdigit():
        lookup = fips.set_index("fips")["name"]
        county_fips = county
        name = lookup[county_fips]
    # get fips from name if name specified
    else:
        lookup = fips.set_index("name")["fips"]
        name = county
        county_fips = lookup[name]

    # get geodataframe of block group shapefile
    bgfile_name = "http://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_{}_tract_500k.zip".format(
        county_fips[:2]
    )
    bg_geo = zip_shp_to_gdf(bgfile_name)

    # subset to those that are in the county and project it to the CRS
    bg_geo = (
        bg_geo[bg_geo["GEOID"].str[:5] == county_fips]
        .to_crs(epsg=epsg)
        .set_index("GEOID")["geometry"]
    )

    # specify variable list and variable names for the census api function
    varlist = [
        "B03002_003E",
        "B03002_012E",
        "B03002_004E",
        "B03002_006E",
        "B03002_005E",
        "B03002_007E",
        "B03002_008E",
        "B03002_009E",
    ]
    names = [
        "White",
        "Hispanic",
        "Black",
        "Asian",
        "AI/AN",
        "NH/PI",
        "Other_",
        "Two Plus",
    ]

    # read in block group level census variables
    dems = get_census_data(
        2018,
        "acs5",
        "block group",
        {"county": county_fips[2:], "state": county_fips[:2]},
        varlist,
        names,
    )
    # Calculate other as sum of those not in the 4 most populated race categories
    dems["Other"] = dems[["AI/AN", "NH/PI", "Other_", "Two Plus"]].sum(1)

    # Calculate county boundaries as the union of block groups
    union = gpd.GeoSeries(bg_geo.unary_union)

    # if axes object is specified, plot to this axis, otherwise create a new one
    if ax:
        union.plot(color="white", figsize=figsize, ax=ax)
    else:
        ax = union.plot(color="white", figsize=figsize)

    # set aspect equal and add title if specified
    ax.set(aspect="equal", xticks=[], yticks=[])
    # set title as county name
    ax.set_title(name, size=15)

    # annotate the dot per person ratio
    ax.annotate(
        "1 dot = {} {}".format(
            pts_per_person, "person" if pts_per_person == 1 else "people"
        ),
        xy=(0.5, 0.97),
        xycoords="axes fraction",
        horizontalalignment="center",
        fontsize=12,
    )

    # loop each race category and generate points for each within each block group
    list_of_point_categories = []
    for field in ["White", "Black", "Asian", "Hispanic", "Other"]:
        ps = gpd.GeoDataFrame(
            gen_points_in_gdf_polys(
                geometry=bg_geo,
                values=dems[field],
                points_per_value=pts_per_person,
                seed=seed,
            )
        )
        ps["field"] = field
        list_of_point_categories.append(ps)

    all_categories = pd.concat(list_of_point_categories)
    all_points = gpd.GeoDataFrame(all_categories)
    all_points.plot(
        ax=ax,
        markersize=0.125,
        alpha=dot_transparency,
        column="field",
        categorical=True,
        legend=legend,
        cmap="Accent",
        marker=",",
    )

    return ax


fig, ax = plt.subplots(1, 1)
plt.box(False)

gen_count_dot_density_map("16001", pts_per_person=10, ax=ax)
# plt.show()
plt.savefig("one-person-one-dot.png", bbox_inches="tight", dpi=5000)

