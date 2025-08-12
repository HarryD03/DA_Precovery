#import packages
import utils.observation as obs
import utils.time_reference as time_ref
import utils.propagation as prop
import utils.iod as iod
import utils.dynamics as dynamics

from typing import Callable, List, Union, overload, Tuple
import time
import numpy as np 
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import pandas as pd

import poliastro as pl
import astropy.time as at
import astropy.constants as ac
import astropy.coordinates as acoords
import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from poliastro.frames.ecliptic import HeliocentricEclipticJ2000

from daceypy import DA, array, ADS
import daceypy.op as op

#TODO: 1) Observe if the Earth Emphermis code obtains heliocentric velocity at each posiiton  
#      2) Check if the DAIOD lambert correction step is okay when in ECI frame but orbiting Mu=sun
#      3) Test ECI 2 Heliocentric covnersion for DAIOD algoirthm if 2) is correct


############################### USER INPUTS ###############################

filepath = 'C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\05_Thesis_code\\apophis_data.csv'
observation_extraction_method = 'instance'

order = 4                   #Order of Taylor Polynminal Expansion

################################# Observation Extraction and Formatting ########################################

#Extraction of Data
NEA = obs.load_observation_file(filepath)
obs_date, RA, DEC, RA_sigma, DEC_sigma = obs.extract_obs(NEA, extraction_method=observation_extraction_method, N_lower=0, N_upper=3)

#Convert from Pandas to Numpy + Convert From MPC2Degrees
# Convert all numeric columns to float
numeric_columns = {
    'obs_date': ['DD.dddddddddd'],
    'RA': ['HH', 'MM_2', 'SS.sss'],
    'DEC': ['sDD', 'MM_3', 'SS.ss'],
    'RA_sigma': ['Accuracy_2'],
    'DEC_sigma': ['Accuracy_3']
}

for df_name, columns in numeric_columns.items():
    df = locals()[df_name]
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

obs_day_time_np, RA_deg, DEC_deg, RA_sigma_deg, DEC_sigma_deg = obs.convert_obs(obs_date, RA, DEC, RA_sigma, DEC_sigma)

obs_YYYYMM_np = obs_date[["YYYY","MM"]].to_numpy()
obs_date_np = np.column_stack((obs_YYYYMM_np, obs_day_time_np))


#Define in terms of Astropy Objects for ease of Conversion
RA = RA_deg * u.deg                 #Right Ascension
DEC = DEC_deg * u.deg               #Declination
RA_sigma = (RA_sigma.to_numpy().T) * u.deg         #Right Ascension Error
DEC_sigma = (DEC_sigma.to_numpy().T) * u.deg       #Declination Error

obs_times = at.Time(
    [f"{int(year):04d}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{second:06.3f}" 
     for year, month, day, hour, minute, second in obs_date_np],
    scale='ut1'
)
#Observatory location: Measurements in Equatorial RA DEC so not needed

#Get Observation times in J2000 epoch for Guass IOD
j0 = obs_times.jd   # Julian Date
Jd_2000 = 2451545.0 # J2000 epoch in Julian Days
obs_J2000 = j0 - Jd_2000    #Observation Days since J2000 epoch

########################## Define Heliocentric Frame and Earth Location within it ##############

mu = ac.G * ac.M_sun
mu = mu.to('km**3 / s**2')


epochs = at.Time(obs_times, scale='tdb')
acoords.solar_system_ephemeris.set("builtin")

pv = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs] # pv[i] = (pos, vel)


earth_pos_helio = np.zeros((len(epochs), 3))
earth_vel_helio = np.zeros((len(epochs), 3))
ICRS_AU = np.zeros(3)
# Stack into arrays and convert to desired units
for i, (pos, vel) in enumerate(pv):
    #Convert positions from AU to km
    pos_vel = CartesianRepresentation(
    x=pos.x.to(u.km),
    y=pos.y.to(u.km),
    z=pos.z.to(u.km),
        differentials={
            's': CartesianDifferential(
                d_x=vel.x.to(u.km/u.s),
                d_y=vel.y.to(u.km/u.s),
                d_z=vel.z.to(u.km/u.s)
            )
        }
    )
    
    # Convert velocities from AU/day to km/s
    earth_vel_helio[i] = [
        vel.x.to(u.km/u.s).value,
        vel.y.to(u.km/u.s).value,
        vel.z.to(u.km/u.s).value
    ]
    ICRS = SkyCoord(pos_vel, frame='icrs', representation_type='cartesian', differential_type='cartesian', obstime=epochs[i])
    helio = ICRS.transform_to(HeliocentricEclipticJ2000).cartesian
    
    earth_pos_helio[i] = helio.xyz.to(u.km) # (Nx3)
    earth_vel_helio[i] = helio.differentials["s"].d_xyz.to(u.km/ u.s) #Nx3



########################### DAIOD #############################

# Convert RA, DEC measurements to radians.
RA_rad = RA_deg.to(u.rad).value
DEC_rad = DEC_deg.to(u.rad).value
RA_sigma_rad = RA_sigma_deg.to(u.rad).value
DEC_sigma_rad = DEC_sigma_deg.to(u.rad).value

#Convert time arrays to all seconds Numpy for Guass IOD
time_sec = obs_J2000.to(u.s).value 

assert len(time_sec) == 3, f"ERROR:\nOnly 3 observations times can be used.\n{len(time_sec)} observations were stored"

# Generate Nominal Guass Range
observer_position_geocentric = np.zeros_like(RA_rad) # The Observer is at the centre of the Earth as RA+DEC are geocentric
i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad) #Define geocentric range vector [ρ̂x, ρ̂y, ρ̂z]ᵢ'
pos_guass, range_guass, range_mag = iod.Guass_8th_seed(observer_position_geocentric, i_rho, time_sec, mu=mu.value)

# DAIOD part 1: Conuduct Position Refinement (DA iterative refininment for Guass IOD Range magnitude)
DA.init(order, 3)
range_mag_DA = array([range_mag[0] + DA(1), range_mag[1] + DA(2), range_mag[2] + DA(3)])
_, range_mag_DAIOD = DAIOD(RA, DEC, range_mag_DA, observer_position_geocentric, time_sec, mu=mu.value)

# DAIOD part 2: Conduct Velocity Refinement (Lambert central vleocity condition refinement)
DA.init(order, 6)
RA_DA = array([RA[i] + 3*RA_sigma_rad*DA(i + 1) for i in range(3)])
DEC_DA = array([DEC[i] + 3*DEC_sigma_rad*DA(i + 4) for i in range(3)])    # Scale RA and DEC DA part so that when evaluated its within the [-1,1] range

X_0_ECI_CC, _ = DAIOD(RA_DA, DEC_DA, range_mag_DAIOD, observer_position_geocentric, time_sec, mu=mu.value) 

#X_0 = Central Orbit state (ECI-CC) at Epoch.

####################### DAIOD ECI TO HELIOCENTRIC CONVERSION #####################################
print("Translating ECI-Cartesian DAIOD output to Heliocentric-Cartesian")

#Check X_0 is compatible with ROT1 function (much be a column)
print(f"X_0_ECI_CC Shape:\n{X_0_ECI_CC.shape}")

#Equator to Ecliptic tilt
eps = np.deg2rad(23.439291)
rot = time_ref.ROT1(eps)
pos_ecliptic_geocentric = rot @ X_0_ECI_CC[:3]
vel_ecliptic_geocentric = rot @ X_0_ECI_CC[3:]

#Vector sum for position and velocity
NEO_pos_helio = pos_ecliptic_geocentric + earth_pos_helio 
NEO_vel_helio = vel_ecliptic_geocentric + earth_vel_helio
#Epoch is the time of central observation in arc.
X_0_Helio_CC = op.concat(NEO_pos_helio,NEO_vel_helio)

############################### ADS Propagation ####################################

X_0_Helio_MEE = time_ref.CC2MEE(X_0_Helio_CC[:3], X_0_Helio_CC[3:], mu=mu.value)

# Preperation for ADS
domain0 = X_0_Helio_MEE.copy()
r_tol = np.array([1,1,1]) # Tolerance for position [m]
v_tol = np.array([1e-3, 1e-3, 1e-3]) # Tolerance for velocity [m/s]
perturbations = np.zeros(3)
tol = np.concatenate((r_tol, v_tol))
Nmax = 10 # Maximum number of splits in ADS
tgrid = np.linspace(t0 , tf, 100)  # Time grid for propagation

init_domain = ADS(domain0, [])
init_list = [init_domain]
final_lists = []
final_list = X_0_Helio_MEE.copy
final_lists.append(final_list)

#ADS Domain Splitting Propagation
start_advanced = time.time()
for i in range(len(tgrid) - 1):
    final_list = ADS.eval(
        final_list, tol, Nmax, 
        lambda domain: 
        prop.advanced_propagationADS(domain, tgrid[i], tgrid[i+1], dynamics.TBP_MEE_DA(domain, mu, perturbations, tgrid[i]))
        )
    final_lists.append(final_list)
    print('time ', tgrid[i+1], 'reached!')

propagation_duration = time.time() - start_advanced
print(f"ADS propagation completed in {propagation_duration:.2f} seconds")

############################ Post Propagation Processing ################################


# Evaluate Perimeter of Manifolds and Domain function (positiion 3D)

# Evaluate the Perimeter of Manifolds and Domain function (pos and vel 6D)

# Wittig style Manifold vs Domain figures 

# Propagation Time vs Number of Images

# Arc Seperatation time vs Number of Images 

# Error in DA State vs Propagation Time

# Error in DA State vs Arc Seperation (Time)
