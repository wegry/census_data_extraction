import requests
import pandas as pd

from census import Census
from us import states
import os


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


varlist = [
    "B02001_001E",
    "B03002_003E",
    "B03002_012E",
    "B02001_002E",
    "B02001_003E",
    "B02001_005E",
    "B02001_004E",
    "B02001_006E",
    "B02001_007E",
    "B02001_008E",
]

names = [
    "total",
    "white_nhs",
    "hispanic",
    "white",
    "black",
    "asian",
    "ai_an",
    "nh_pi",
    "other",
    "two_plus",
]

df = get_census_data(2018, "acs5", "tract", {}, varlist, names)

df.to_csv("./output-data.csv")

