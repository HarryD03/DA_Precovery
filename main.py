#import packages
from utils import post_process
import utils.observation as obs
import utils.time_reference as time_ref
import utils.propagation as prop
import utils.iod as iod
import utils.dynamics as dynamics
from utils.lambert_izzo import lambert_izzo
import utils.admissable_region as ar


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
obs_date, RA, DEC, RA_sigma, DEC_sigma = obs.extract_obs(NEA, extraction_method=observation_extraction_method, N_lower=72, N_upper=240)     #set this to one arc length

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

#Get Observation times in J2000 epoch for Guass IOD
j0 = obs_times.jd   # Julian Date
Jd_2000 = 2451545.0 # J2000 epoch in Julian Days
obs_J2000 = (j0 - Jd_2000) * u.day    #Observation Days since J2000 epoch

# Delete all observations but the start, end and the center observation 

obs_times, RA, DEC, RA_sigma, DEC_sigma, obs_J2000= obs.filter_observations_to_three(obs_times, RA, DEC, RA_sigma, DEC_sigma, obs_J2000)
#Define Sky Coordinates 
coord1 = acoords.TETE(ra=RA[0], dec=DEC[0], obstime=obs_times[0])
coord2 = acoords.TETE(ra=RA[1], dec=DEC[1], obstime=obs_times[1])
coord3 = acoords.TETE(ra=RA[2], dec=DEC[2], obstime=obs_times[2])

########################## Define Heliocentric Frame and Earth Location within it ##############
#THIS IS CORRECT
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
    helio_manual = time_ref.ROT1(np.deg2rad(24.3))
    
    helio = ICRS.transform_to(HeliocentricEclipticJ2000).cartesian
    
    earth_pos_helio[i] = helio.xyz.to(u.km) # (Nx3)
    earth_vel_helio[i] = helio.differentials["s"].d_xyz.to(u.km/ u.s) #Nx3

X_earth_helio = np.concatenate((earth_pos_helio, earth_vel_helio))

#Observation Coordinate Conversion
#coord1_helio = coord1.transform_to(HeliocentricEclipticJ2000)
#coord2_helio = coord2.transform_to(HeliocentricEclipticJ2000)
#coord3_helio = coord3.transform_to(HeliocentricEclipticJ2000)

#################### Obtain Angles #####################

# Convert RA, DEC measurements to radians.
RA_rad = RA.to(u.rad).value
DEC_rad = DEC.to(u.rad).value
RA_sigma_rad = RA_sigma.to(u.rad).value
DEC_sigma_rad = DEC_sigma.to(u.rad).value


#Convert time arrays to all seconds Numpy for Guass IOD
time_sec = obs_J2000.to(u.s).value 

assert len(time_sec) == 3, f"ERROR:\nOnly 3 observations times can be used.\n{len(time_sec)} observations were stored"

# Generate Nominal Guass Range
observer_position_helio = earth_pos_helio.T # The Observer is at the centre of the Earth as RA+DEC are equatorial RA DEC
i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad) #Define geocentric range vector [ρ̂x, ρ̂y, ρ̂z]ᵢ' Geocentric range is equatorial
i_rho = i_rho.astype(np.float64)
#rotate direction vector directly - equatorial to ecliptic 
i_rho_helio = (time_ref.ROT1(np.deg2rad(23.5)) * i_rho)

########################## Admissable Region #############################

#ar.initiate_admissible_region(RA.value, DEC.value, obs_times, apparent_magnitude=18)

########################### Gauss IOD #########################################

r_IOD, _, _, v_IOD = iod.Guass_8th_seed(observer_position_helio, i_rho_helio, time_sec, mu=mu.value)
X0_GUASS = np.concatenate((r_IOD, v_IOD))

########################### DAIOD #############################################

# Convert RA, DEC measurements to radians.
RA_rad = RA.to(u.rad).value
DEC_rad = DEC.to(u.rad).value
RA_sigma_rad = RA_sigma.to(u.rad).value
DEC_sigma_rad = DEC_sigma.to(u.rad).value


#Convert time arrays to all seconds Numpy for Guass IOD
time_sec = obs_J2000.to(u.s).value 

assert len(time_sec) == 3, f"ERROR:\nOnly 3 observations times can be used.\n{len(time_sec)} observations were stored"

# Generate Nominal Guass Range
observer_position_helio = earth_pos_helio.T # The Observer is at the centre of the Earth as RA+DEC are equatorial RA DEC
i_rho = time_ref.create_da_los_vectors(RA_rad, DEC_rad) #Define geocentric range vector [ρ̂x, ρ̂y, ρ̂z]ᵢ' Geocentric range is equatorial
i_rho = i_rho.astype(np.float64)

#rotate direction vector directly - equatorial to ecliptic 
i_rho_helio = (time_ref.ROT1(np.deg2rad(23.5)) * i_rho)
pos_guass, range_guass, range_mag_gauss = iod.Guass_8th_seed(observer_position_helio, i_rho_helio, time_sec, mu=mu.value) # The position interval seems okay but the range is fucked

#Show Transfer Angles
cos_dnu12 = np.dot(pos_guass[:,0], pos_guass[:,1]) / (np.linalg.norm(pos_guass[:,0]) * np.linalg.norm(pos_guass[:,1]))
cos_dnu23 = np.dot(pos_guass[:,1], pos_guass[:,2]) / (np.linalg.norm(pos_guass[:,1]) * np.linalg.norm(pos_guass[:,2]))

print("================GAUSS TRANSFER ANGLES==================")
print(f"cos(Δν12) = {cos_dnu12}")
print(f"cos(Δν23) = {cos_dnu23}")
if cos_dnu12 < 2 or cos_dnu23 < 2:
    print("WARNING: Small Angle Transfer. Short Arc Detected")

# DAIOD part 1: Conuduct Position Refinement (DA iterative refininment for Guass IOD Range magnitude)
range_mag_DArange, Jacobian_dv = iod.DAIOD_1Scipy_invert(range_mag_gauss, i_rho_helio, time_sec, order, r_obs_heliocentric=observer_position_helio, mu=mu.value)

# DAIOD part 2: Conduct Velocity Refinement (Lambert central vleocity condition refinement)
DA.init(order, 6)
DA.Eps(1e-16)

RA_DA = array([RA_rad[i] + 3*RA_sigma_rad*DEC_rad(i + 1) for i in range(3)])
DEC_DA = array([DEC_rad[i] + 3*DEC_sigma_rad*DEC_rad(i + 4) for i in range(3)])    # Scale RA and DEC DA part so that when evaluated its within the [-1,1] range


range_mag_DAangles = iod.DAIOD_2(range_mag_DArange, RA_DA, DEC_DA, time_sec, order, r_obs_heliocentric=observer_position_helio, mu=mu.value, J=Jacobian_dv)

#ADD ADS FOR SPLITTING



# DAIOD part 3: Convert the range_magnitude Map to Cartesian State

i_rho_DA = array(time_ref.create_da_los_vectors(RA_DA, DEC_DA))
i_rho_ecl_DA = time_ref.ROT1(np.deg2rad(23.5) * i_rho_DA)
range_vec = i_rho_ecl_DA * range_mag_DAangles
pos_vec = range_vec + observer_position_helio   #Position Vector (CC) Heliocentric Ecliptic

dt1 = time_sec[1] - time_sec[0]
dt2 = time_sec[2] - time_sec[1]

vel = []
velocities1 = lambert_izzo(pos_vec[:,0], pos_vec[:,1], dt1, mu.value, mult_revs= 0, prograde=False)
vel.append(velocities1[0])  # Unpack the first solution

velocities2 = lambert_izzo(pos_vec[:,1], pos_vec[:,2], dt2, mu.value, mult_revs= 0, prograde=False)
vel.append(velocities2[0])  # Unpack the first solution

v1 = vel[0]
v2 = vel[1]
v3 = vel[3]     #Velocity Vector (CC) Heliocentric Ecliptic
vel_vec = v1.concat(v2).concat(v3)  #Concatenate the velocity vectors

if vel_vec.shape != (3,3):
    print(f"ERROR: Velocity vector shape mismatch\nreshaping...\n")
    vel_vec = np.array(vel_vec, dtype=object)
    vel_vec = vel_vec.reshape((3,3))
    vel_vec = array(vel_vec)

X0_CC_Eclip_Helio = pos_vec.concat(vel_vec)   # NEO Cartesian State at all positions

X0_CC_Eclip_Helio_Epoch = X0_CC_Eclip_Helio[:,1].copy() #Epoch Cartesian State

print(f"The Epoch Cartesian State (Heliocentric Ecliptic) is:\n{X0_CC_Eclip_Helio_Epoch.cons()}\n")
print(f"The Time at Epoch is:\n{obs_J2000[1]} @ J2000")
print(f"The Time at Epoch is:\n{obs_times[1]} @ UT1")

X0_CC_Eclip_Helio_zeroth = X0_CC_Eclip_Helio.cons()         #Constant (Zeroth Order) terms in Taylor Map
cos_dnu12 = np.dot(X0_CC_Eclip_Helio_zeroth[:3,0], X0_CC_Eclip_Helio_zeroth[:3,1]) / (np.linalg.norm(X0_CC_Eclip_Helio_zeroth[:3,0]) * np.linalg.norm(X0_CC_Eclip_Helio_zeroth[:3,1]))
cos_dnu23 = np.dot(X0_CC_Eclip_Helio_zeroth[:3,1], X0_CC_Eclip_Helio_zeroth[:3,2]) / (np.linalg.norm(X0_CC_Eclip_Helio_zeroth[:3,1]) * np.linalg.norm(X0_CC_Eclip_Helio_zeroth[:3,2]))

print(f"================REFINED TRANSFER ANGLES==================")
print(f"cos(Δν12) = {cos_dnu12}")
print(f"cos(Δν23) = {cos_dnu23}")

############################### FoR Conversions w/ DA ####################################

X0_CC_ECI_Epoch = time_ref.Helio2ECIJ2000(X0_CC_Eclip_Helio_Epoch, X_earth_helio)
X0_Obs_ECI_Epoch = time_ref.CC2obs(X0_CC_ECI_Epoch) #RA, DEC, Range at Epoch

# X0_Helio_MEE_Epoch = time_ref.CC2MEE(X0_CC_Eclip_Helio_Epoch[:3], X0_CC_Eclip_Helio_Epoch[3:], mu=mu.value)    #Conversion for propagation

############################# Obtain Box Perimeter of Initial Observation State #############################

Nmax = 10 # Maximum number of splits in ADS
t0 = time_sec[1]            # Epoch Time [Seconds]
tf_days = 10                # Final Time [Days]
tf = tf_days * 86400        # Final Time [Seconds]
Ts = 100                    # Number of Time Samples
Ns = 50                     # Number of Spatial Samples

tgrid = np.linspace(t0 , tf, Ts)  # Time grid for propagation

avg_sigma = 1/2 * (RA_sigma_rad + DEC_sigma_rad)  # Average sigma for the perimeter generation
obs_perimeter_norm, obs_perimeter = post_process.generate_perimeter3D(3 * avg_sigma, Ns)
obs_perimeter_tmp = obs_perimeter + np.zeros_like(obs_perimeter)

#Observation Equorital to Heliocentric Ecliptic perimeter for Propagaiton
CC_perimeter_tmp = time_ref.Obs2CC(obs_perimeter_tmp)
CC_perimeter = time_ref.ECI2HelioJ2000(CC_perimeter_tmp, X_earth_helio)
CC_perimeter_pos = CC_perimeter[:,:3]   #Position perimeter

############################# Pointwise Propagation #################################

#Propagate ground truth perimeters (DAIOD + PW integration along bounding box) -> This will be most accurate
try: 
    with (thisfolder / 'DAIOD_PW_propagation.npy').open('rb') as f:
        XF_PW = np.load(f, allow_pickle=True)
        XF_nom = np.load(f, allow_pickle=True)
        duration_PW = np.load(f, allow_pickle=True)
        print(f"Pointwise Propagation Duration: {duration_PW:.2f} seconds")

except FileNotFoundError:
    
    #Conduct Pointwise Integration
    XF_boundary_PW = np.zeros((CC_perimeter.shape[0], 6, Ts))       #Perimeter [perimeter X states X time]
    XF_nom = np.zeros((6, Ts))                                      #Nominal solution
    
    XF_boundary_PW[:,:,0] = CC_perimeter
    
    start_PW = time.time()
    for k in range(Ts-1):
        tspan = [tgrid[k+1], tgrid[k]]
        XF_boundary_PW[:,:,k+1], XF_nom[:,k+1] = prop._propagate_bounding_box_edges_facesPW(X0_CC_Eclip_Helio_Epoch, CC_perimeter_pos, tspan, )

    end_PW = time.time()
    duration_PW = end_PW - start_PW
    with ('DAIOD_PW_propagation.npy').open('wb') as f:
        np.save(f, XF_boundary_PW, allow_pickle=True)
        np.save(f, XF_nom, allow_pickle=True)
        np.save(f, duration_PW, allow_pickle=True)

# Propagate Guass perimeters (IOD Solution + PW Integration along boundaing box) -> Compare Guass vs DAIOD

try: 
    with ('IOD_PW_propagation.npy').open('rb') as f:
        XF_IOD_PW = np.load(f, allow_pickle=True)
        XF_IOD_NOM = np.load(f, allow_pickle=True)
        duration_IOD_PW = np.load(f, allow_pickle=True)
        print(f"IOD Pointwise Propagation Duration: {duration_IOD_PW:.2f} seconds")

except FileNotFoundError:

    #Conduct Pointwise Integration 
    XF_IOD_boundary_PW = np.zeros((CC_perimeter.shape[0], 6, Ts))       #Perimeter
    XF_IOD_nom = np.zeros((6, Ts))                                      #Nominal solution
    XF_IOD_boundary_PW[:,:,0] = CC_perimeter

    start_IOD_PW = time.time()
    for k in range(Ts-1):
        tspan = [tgrid[k+1], tgrid[k]]
        XF_IOD_boundary_PW[:,:,k+1], XF_IOD_nom[:,k+1] = prop._propagate_bounding_box_edges_facesPW(X0_GUASS, CC_perimeter_pos,)

    end_IOD_PW = time.time()
    duration_IOD_PW = end_IOD_PW - start_IOD_PW

    with ('IOD_PW_propagation.npy').open('wb') as f:
        np.save(f, XF_IOD_boundary_PW, allow_pickle=True)
        np.save(f, XF_IOD_nom, allow_pickle=True)
        np.save(f, duration_IOD_PW, allow_pickle=True)

# Propagate AR perimeters (Apophis AR Perimeter Integration)



############################## ADS Propgation ######################################
# Preperation for ADS

domain0 = X0_CC_Eclip_Helio_Epoch.copy()
r_tol = np.array([1,1,1]) # Tolerance for position [km]
v_tol = np.array([1e-3, 1e-3, 1e-3]) # Tolerance for velocity [km/s]
perturbations = np.zeros(3)
tol = np.concatenate((r_tol, v_tol))

init_domain = ADS(domain0, [])
init_list = [init_domain]
final_lists = []
final_list = X0_CC_Eclip_Helio_Epoch.copy()
final_lists.append(final_list)

#ADS Domain Splitting Propagation
start_advanced = time.time()
for i in range(len(tgrid) - 1):
    final_list = ADS.eval(
        final_list, tol, Nmax, 
        lambda domain: 
        prop.advanced_propagationADS(domain, tgrid[i], tgrid[i+1], dynamics.TBP_CC_DA(domain, mu.value, tgrid[i]))
        )
    final_lists.append(final_list)
    print('time ', tgrid[i+1], 'reached!')

propagation_duration = time.time() - start_advanced
print(f"ADS propagation completed in {propagation_duration:.2f} seconds")

############################ Post Propagation Processing ################################


# Evaluate Perimeter of Manifolds and Domain function (position 3D)


post_process.evaluate_perimeter(final_lists)


# Wittig style Manifold vs Domain figures - figure6()


# Propagation Time vs Number of Images (Fixed Arc Length)

# Arc Length time vs Number of Images (Fixed Propagation Time)

# Error in DA State vs Propagation Time (Contour) (Fixed Arc Length)


# Error in DA State vs Arc Length (Time) (Contour) (Fixed Propagation Time)
