from urllib.parse import urlencode
import pandas as pd
import numpy as np
from astroquery.cadc import Cadc
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.table import Table, Column
from shapely import Polygon 
    

def CADC_Fixed_Polygon(polygon: Polygon, start_date, end_date, collection=''):
    """
    Find the Images Within a Fixed Polygon area
    :params Polygon: The polygon defining the RAxDEC area of interest
    :params start_date: The start date for the query. Must be in iso Astropy format
    :params end_date: The end date for the query. Must be in iso Astropy format
    """

    query_outline = """SELECT {num}
    *
    FROM caom2.Plane AS Plane
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
    WHERE (INTERSECTS( INTERVAL( {mjd_start}, {mjd_end} ), Plane.time_bounds_samples ) = 1
    AND INTERSECTS(Plane.position_bounds, POLYGON('ICRS', {polygon})) = 1
    AND collection IN ({collection})
    AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
    ORDER BY time_bounds_lower {order}
    """

    query_params = {
    'num': '',  # Restricts the number of results (empty string returns all)
    'mjd_start': start_date.mjd,
    'mjd_end': end_date.mjd,
    'polygon': ', '.join([f"{ra} {dec}" for ra, dec in polygon.exterior.coords]),
    'collection': collection,
    'order': 'ASC'  # Order the results from oldest to newest
    }

    cadc = Cadc()

    results = cadc.exec_sync(query_outline.format(**query_params))      #Results of query

    #Only Care about RA, DEC, Time, image number
    columns_subset = [
        'productID','Image','collection', 'time_bounds_samples', 'position_bounds'
    ]

    table = results[columns_subset]
    
    return table

def SSOIS_Query_Apophis(error, start_date, end_date):
    """
    Conduct SSOIS query for the asteroid Apophis.
    Options between:
    - Positional uncertainty
    - No positional uncertainty
    :params error: The positional uncertainty in arcseconds. MPC assumes a box independant of time.
    :params start_date: The start date for the query ("YYYY+MM+DD") format
    :params end_date: The end date for the query ("YYYY+MM+DD") format
    """
    CADC_name = 'Apophis'
    eunits = 'arcseconds'

    assert isinstance(start_date, str), "Start date must be a string in the format 'YYYY+MM+DD'"
    assert isinstance(end_date, str), "End date must be a string in the format 'YYYY+MM+DD'"

    print('Conducting SSOIS Query...')

    baseurl = 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl'
    url = f"{baseurl}?lang=en;object={CADC_name};search=bynameCADC;epoch1={start_date};epoch2={end_date};eellipse={error};eunits={eunits};extres=no;xyres=no;format=tsv"

    #Convert to DataFrame for ease 
    data_table = pd.read_csv(url, sep='\t')

    return data_table





     




# Fixed Space Query - 
coords = SkyCoord(ra=334.0276125, dec=-17.70465556, unit=(u.deg, u.deg))
days_before = 20.0
days_after = 100.0
maxdate = Time('2003-09-03T00:00:00', format='isot', scale='utc')

start_date = maxdate - TimeDelta(days_before, format='jd')
end_date = maxdate + TimeDelta(days_after, format='jd')

print('Coordinates: {}'.format(coords))
print('Query Date range: {} to {}'.format(start_date.iso, end_date.iso))

query_outline = """SELECT {num}
*
FROM caom2.Plane AS Plane 
JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
WHERE (INTERSECTS( INTERVAL( {mjd_start}, {mjd_end} ), Plane.time_bounds_samples ) = 1 
AND INTERSECTS(Plane.position_bounds, CIRCLE('ICRS', {ra}, {dec}, 0.5) ) = 1 
    AND LOWER(Plane.energy_bandpassName) LIKE '{filter}%' 
    AND collection IN ('{collection}', 'GEMINI')
    AND calibrationLevel >= {cal_level}
    AND (Plane.quality_flag IS NULL OR Plane.quality_flag != 'junk'))
    ORDER BY time_bounds_lower {order}"""


# Build query url
query_params = {
    'num': '',  # Restricts the number of results (empty string returns all)
    'mjd_start': start_date.mjd,
    'mjd_end': end_date.mjd,
    'ra': coords.ra.degree,
    'dec': coords.dec.degree,
    'filter': 'r',
    'collection': 'CFHT',
    'cal_level': 2,
    'order': 'ASC'  # Order the results from oldest to newest
}

cadc = Cadc()

results = cadc.exec_sync(query_outline.format(**query_params))

columns_subset = [
    'productID', 'collection', 'time_bounds_samples', 'time_bounds_lower', 'time_exposure'
]

print('Total number of results: {}'.format(len(results)))
print(results[columns_subset][0:5])

## SSOIS Query: Obtain MPC data
print('Conducting SSOIS Query...')

CADC_name = 'Apophis'
start_date = '2004+01+18'
end_date = '2007+01+19'
error = str(0.5)
min_exp_time = 20 #seconds

# SSOIS expects semicolon-separated parameters, not ampersand-separated
# Build Query URL manually to match expected format
baseurl = 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl'

# Construct the URL with semicolons as separators
url = f"{baseurl}?lang=en;object={CADC_name};search=bynameCADC;epoch1={start_date};epoch2={end_date};eellipse={error};eunits=arcseconds;extres=no;xyres=no;format=tsv"

print(f"SSOIS Query URL: {url}")
#Access data query
data_table = pd.read_csv(url, sep='\t')

data_table = data_table[(data_table['Exptime'] >= min_exp_time)]

print("SSOIS Query Results:")
print(data_table)

#Transform into AstroPy table

table = Table.from_pandas(data_table)
table.rename_column('Object_RA', 'RA')
table.rename_column('Object_Dec', 'Dec')
table.rename_column('Exptime', 'Exposure Time')
table['Image'] = Column(data=data_table['Image'], 
                        name='Image',
                        dtype=np.dtype('S10'))

subtable = table['Image', 'MJD', 'RA', 'Dec', 'Exposure Time']
print(subtable[0:5])
print("Obtaining Image Metadata...")
#Query the CADC caom2 database with SOISS data
cadc = Cadc()

query = """SELECT upload_table. *, Plane.publisherID 
FROM tap_upload.upload_table AS upload_table
JOIN caom2.Plane AS Plane ON Plane.productID = upload_table.Image
JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID
"""

cadc_results = cadc.exec_sync(query, uploads={'upload_table' : subtable[['Image', 'MJD', 'RA', 'Dec']]})

cadc_table = cadc_results.to_pandas()

str_results = cadc_table[['Image', 'publisherID']].stack().str.decode('utf-8').unstack()

for col in str_results:
    cadc_table[col] = str_results[col]

print(cadc_table)

