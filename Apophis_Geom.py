import os
import numpy as np
import pandas as pd
from daceypy import DA, array, ADS
from astropy.time import Time, TimeDelta
from astropy import units as u
import time
import matplotlib.pyplot as plt
import utils.optimisealpha as optalpha
import astropy.constants as ac
import astropy.coordinates as acoords

import utils.observation as obs
import utils.time_reference as time_ref_func
import utils.propagation as prop
import utils.iod as iod
import utils.dynamics as dynamics
import utils.Classical_IOD as PW
import utils.lambert_izzo as lambert_izzo
import utils.post_process as post
import utils.plotting as plot
import utils.dynamics as dynamics

#User Defined variables
DA_order = 6                              #Pirovano Defined
ang_sigma = 1/3 * (1/3600) * (np.pi/180)  # ≈ 1.61e-6 radians
mu = ac.G * ac.M_sun
filepath = 'C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\05_Thesis_code\\apophis_data.csv'
observation_extraction_method = 'instance'  #Options: 'arc', 'instance', 'custom'
N_simulations = 100         #Number of Monte Carlo Samples
Ts = 3                      #Number of propagation time steps
Ns = 2                      #Number of samples per dimension for the perimeter grid
N_arcs = [1]
prop_duration = 0.25          # days
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

obs_day_time_np, RA_deg, DEC_deg, RA_sigma_rad, DEC_sigma_rad = obs.convert_obs(obs_date, RA, DEC, RA_sigma, DEC_sigma)

obs_YYYYMM_np = obs_date[["YYYY","MM"]].to_numpy()
obs_date_np = np.column_stack((obs_YYYYMM_np, obs_day_time_np))


#Define in terms of Astropy Objects for ease of Conversion
RA = RA_deg * u.deg                                 #Right Ascension
DEC = DEC_deg * u.deg                               #Declination
RA_sigma = (RA_sigma.to_numpy().T) * u.arcsec         #Right Ascension Error
DEC_sigma = (DEC_sigma.to_numpy().T) * u.arcsec       #Declination Error

obs_times = Time(
    [f"{int(year):04d}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{second:06.3f}" 
     for year, month, day, hour, minute, second in obs_date_np],
    scale='ut1'
)

obs_times = Time(['2005-01-15T01:19:11.000', '2005-01-17T01:19:10.000', '2005-01-20T01:19:11.000'], format='isot', scale='ut1')

#Get Observation times in J2000 epoch for Guass IOD
j0 = obs_times.jd   # Julian Date
Jd_2000 = 2451545.0 # J2000 epoch in Julian Days
obs_J2000 = (j0 - Jd_2000) * u.day    #Observation Days since J2000 epoch


obs_times, RA, DEC, RA_sigma, DEC_sigma, obsJ200 = obs.filter_observations_to_three(obs_times, RA, DEC, RA_sigma, DEC_sigma, obs_J2000)
ang_sigma = (RA_sigma.value + DEC_sigma.value / 2) * u.arcsec   #Average Angular Error in Radians
ang_sigma = ang_sigma.to(u.rad)

assert len(obs_times) == 3


#Propagation times
tstart = obs_times[1]
tend = tstart + TimeDelta(prop_duration, format='jd')
tgrid = np.linspace(tstart.jd*24*3600, tend.jd*24*3600, Ts)  #seconds from Observation Epoch

#Observation times
observation_times_seconds = obs_times.jd * u.day.to(u.s)  #seconds from start time
assert len(observation_times_seconds) == 3

#Obs_perimeter_norm = post.gen_grid6D(3*ang_sigma[1].value, Ns)
Method_Clock_time = np.zeros((5,5))                                        #Structure: ( DAIOD+ADS_ADS,  DAIOD_+ADS, DAIOD_DA, PW_MC, PW_gauss) x (OD, Prop, Eval, Conversion, Query)
##################################### Observer Postion (Earth Heliocentric Elliptic) ##########################################

#THIS IS CORRECT
mu = ac.G * ac.M_sun
mu = mu.to('km**3 / s**2').value


epochs = Time(obs_times, scale='tdb')
acoords.solar_system_ephemeris.set("builtin")

#Get ICRS Equatorial position of the Earth at each observation epoch
pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]          # pv[i] = (pos, vel)

#Translate to Heliocentric Equatorial Frame
pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]  # (pos, vel) relative to Sun


for i, (p_helio, v_helio) in enumerate(pv_earth_helio):
    #Maintain Equaotorial Frame
    pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))  # [x, y, z, vx, vy, vz]

pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T  # km [3xN]
vel_obs = np.array([pv_earth_helio[i][3:] for i in range(len(pv_earth_helio))]).T  # km/s [3xN]

assert pos_obs.shape == (3, len(obs_times))
assert vel_obs.shape == (3, len(obs_times))

X_HELIO_observation_epoch = np.concatenate((pos_obs[:,1], vel_obs[:,1]))     #Observer State in ECI coordinates [6,]

## Earth Position at Propagation times
prop_times= tgrid / (24*3600)   #Julian Dates of propagation times
prop_times = Time(prop_times, format='jd', scale='tdb')

#obtain ICRS frame of Earth at propagation times
pv_earth_prop = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in prop_times]          # pv[i] = (pos, vel)
#Translate to Heliocentric Equatorial Frame
pv_sun_prop = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in prop_times]
pv_earth_helio_prop = [(pv_earth_prop[i][0] - pv_sun_prop[i][0], pv_earth_prop[i][1] - pv_sun_prop[i][1]) for i in range(len(pv_earth_prop))]  # (pos, vel) relative to Sun

for i, (p_helio, v_helio) in enumerate(pv_earth_helio_prop):
    #Maintain Equaotorial Frame
    pv_earth_helio_prop[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))  # [x, y, z, vx, vy, vz]

pos_obs_prop = np.array([pv_earth_helio_prop[i][:3] for i in range(len(pv_earth_helio_prop))]).T  # km
vel_obs_prop = np.array([pv_earth_helio_prop[i][3:] for i in range(len(pv_earth_helio_prop))]).T  # km/s

X_HELIO_observation = np.zeros((6, pos_obs_prop.shape[1]))  #Structure: ([i,j,k] x instances)
for i in range(pos_obs_prop.shape[1]):
    X_HELIO_observation[:,i] = np.concatenate((pos_obs_prop[:,i], vel_obs_prop[:,i]))     #Observer State in Heliocentric Equatorial coordinates [6xN]

pos_obs_prop[:,0] = pos_obs[:,1]  #Ensure initial position is exactly the same as the observation epoch 


################################# Orbit Determination ~##########################################################
RA_rad = RA.to(u.rad).value
DEC_rad = DEC.to(u.rad).value
eph, vec_ECI = post.load_Apophis_Ephemeris('2004-12-26','2004-12-30', '2d')
_, vec_helio_ICRS = post.load_Apophis_Ephemeris('2004-12-26','2004-12-30', '2d', location='500@sun')

RA = eph['RA']
DEC = eph['DEC']
RA_rad = RA.to(u.rad).value
DEC_rad = DEC.to(u.rad).value
observation_times_seconds = eph['datetime_jd'].to(u.s).value
RA_sigma = eph['RA_3sigma'].to(u.rad).value/3
DEC_sigma = eph['DEC_3sigma'].to(u.rad).value/3
ang_sigma = (RA_sigma[1] + DEC_sigma[1]) / 2 * u.rad   #Average Angular Error in Radians
Obs_perimeter_norm = post.gen_grid6D(3*ang_sigma.value, Ns)

time_ref_start = time.time()
X0_Obj_ECI_DAIOD_ADS = iod.DAIOD_ADS_full(RA_rad, DEC_rad, RA_sigma[1], DEC_sigma[1], pos_obs, observation_times_seconds, mu, DA_order, prograde=True) #Run DAIOD to get Object State
time_ref_end = time.time()
Method_Clock_time[0,0] = time_ref_end - time_ref_start

time_ref_start = time.time()
X0_Obj_ECI_DAIOD_DA = iod.DAIOD_full(RA_rad, DEC_rad, RA_sigma[1], DEC_sigma[1], pos_obs, observation_times_seconds, mu, DA_order, prograde=True)                                 #Run DAIOD to get Object State
time_ref_end = time.time()
Method_Clock_time[1,0] = time_ref_end - time_ref_start
Method_Clock_time[2,0] = time_ref_end - time_ref_start 

time_ref_start = time.time()
X0_Obj_ECI_DAIOD_MC, X0_Obj_ECI_GAUSS_MC = PW.monte_carlo_gauss_PWiod(N_simulations, pos_obs, RA_rad, DEC_rad, observation_times_seconds, RA_sigma[1], DEC_sigma[1], mu, prograde_bool_=True)              #Run PW Gauss IOD to get Object State
time_ref_end = time.time()
Method_Clock_time[3,0] = time_ref_end - time_ref_start
Method_Clock_time[4,0] = time_ref_end - time_ref_start  #
################################ Orbit Propagation ####################################################

# ADS Propagation - DAIOD+ADS
final_list = X0_Obj_ECI_DAIOD_ADS.copy()
final_lists_ADS=[final_list]
toll = np.array([1, 1, 1, 1e-2, 1e-2, 1e-2])
Max_split = 10

DAIOD_ADS_time = time.time()
for i in range(len(tgrid)-1):

    final_list = ADS.eval(
        final_list,
        toll,
        Max_split,
        lambda domain: prop.advanced_propagationADS(domain, tgrid[i], tgrid[i+1], dynamics.TBP_CC_DA)
    )
    final_lists_ADS.append(final_list)

    print(f"Completed propagation to t = {tgrid[i+1]} seconds")
Method_Clock_time[0,1] = time.time() - DAIOD_ADS_time

# ADS Propagation - DAIOD (Single Map Start)
DAIOD_ADS_prop_time = time.time()
init_domain = ADS(X0_Obj_ECI_DAIOD_DA, [])
init_list = [init_domain]
final_lists_ADS_DAIOD = []
final_list = init_list.copy()
final_lists_ADS_DAIOD.append(final_list) # add also initial domains

for i in range(len(tgrid)-1):

    final_list = ADS.eval(
        final_list,
        toll,
        Max_split,
        lambda domain: prop.advanced_propagationADS(domain, tgrid[i], tgrid[i+1], dynamics.TBP_CC_DA)
    )
    final_lists_ADS_DAIOD.append(final_list)

    print(f"Completed propagation to t = {tgrid[i+1]} seconds")
    print(f"CURRENT IDX: {i}")

Method_Clock_time[1,1] = time.time() - DAIOD_ADS_prop_time
# DA Propagation - DAIOD (Single Map Start)

DAIOD_DA_prop_time = time.time()
X0_Obj_ECI_DAIOD_DA_propagated = prop.advanced_propagationDA(X0_Obj_ECI_DAIOD_DA, tgrid, dynamics.TBP_CC_DA) # Everything done inside loop
Method_Clock_time[2,1] = time.time() - DAIOD_DA_prop_time
#PW Propagation - Monte Carlo (Zeroth order DAIOD)

DAIOD_MC_prop_time = time.time()
X0_Obj_ECI_MC_propagated, _ = PW.monte_carlo_propagation(X0_Obj_ECI_DAIOD_MC, tgrid[0], tgrid[-1], Ts, mu=mu)
Method_Clock_time[3,1] = time.time() - DAIOD_MC_prop_time

#PW Propagation - Gauss's solution
Gauss_MC_prop_time = time.time()
X0_Obj_ECI_GAUSS_MC_propagated, _ = PW.monte_carlo_propagation(X0_Obj_ECI_GAUSS_MC, tgrid[0], tgrid[-1], Ts, mu=mu)
Method_Clock_time[4,1] = time.time() - Gauss_MC_prop_time

################################## Observer Reference Frame Conversion ######################################

# Convert all results to the Observer's reference frame for comparison 

# DAIOD+ADS_ADS
time_ref_start = time.time()
X_ADS_Topocentric = post.ADS_Helio2GEO(final_lists_ADS, X_HELIO_observation, None)
X_ADS_Topocentric_Observation = post.ADS_Cart_2_Obs(X_ADS_Topocentric)
Method_Clock_time[0,3] = time.time() - time_ref_start

# DAIOD ADS propagation
time_ref_start = time.time()
X_DAIOD_Topocentric = post.ADS_Helio2GEO(final_lists_ADS_DAIOD, X_HELIO_observation, None)
X_DAIOD_Topocentric_Observation = post.ADS_Cart_2_Obs(X_DAIOD_Topocentric)
Method_Clock_time[1,3] = time.time() - time_ref_start

# DAIOD DA propagation
time_ref_start = time.time()
X_DAIOD_DA_Topocentric = array.zeros((6, X0_Obj_ECI_DAIOD_DA_propagated.shape[1]))  # Structure: (States x Time)
for i in range(X0_Obj_ECI_DAIOD_DA_propagated.shape[1]):
    X_DAIOD_DA_Topocentric[:,i] = X0_Obj_ECI_DAIOD_DA_propagated[:,i] - X_HELIO_observation[:,i] # [6 x len(prop_time)] - [6 x 1] -> Technically this is wrong. We should call the position of the observer at propagated time step 
    X_DAIOD_DA_Topocentric[:,i] = time_ref_func.CC2obs(X_DAIOD_DA_Topocentric[:,i])  # Convert to Observables
Method_Clock_time[2,3] = time.time() - time_ref_start

# PW Conversion - DAIOD formulation
time_ref_start = time.time()
X0_Obj_observables_MC_propagated = np.zeros_like(X0_Obj_ECI_MC_propagated)
for i in range(X0_Obj_ECI_MC_propagated.shape[2]):
    for j in range(X0_Obj_ECI_MC_propagated.shape[0]):
        X0_Obj_Topocentric_MC_propagated_Cart = X0_Obj_ECI_MC_propagated[j, :, i] - X_HELIO_observation[:,i]
        X0_Obj_observables_MC_propagated[j, :, i] = time_ref_func.CC2obs(X0_Obj_Topocentric_MC_propagated_Cart)
Method_Clock_time[3,3] = time.time() - time_ref_start

# PW Conversion - Gauss
time_ref_start = time.time()
X0_Obj_observables_GAUSS_MC_propagated = np.zeros_like(X0_Obj_ECI_GAUSS_MC_propagated)
for i in range(X0_Obj_ECI_GAUSS_MC_propagated.shape[2]):
    for j in range(X0_Obj_ECI_GAUSS_MC_propagated.shape[0]):
        X0_Obj_Topocentric_GAUSS_MC_propagated_Cart = X0_Obj_ECI_GAUSS_MC_propagated[j, :, i] - X_HELIO_observation[:,i]
        X0_Obj_observables_GAUSS_MC_propagated[j, :, i] = time_ref_func.CC2obs(X0_Obj_Topocentric_GAUSS_MC_propagated_Cart)
Method_Clock_time[4,3] = time.time() - time_ref_start   

print("Monte Carlo Propagation complete")
################################# ADS / DA Evaluation for Plotting ##########################################
#TODO: Add error ellipsoid extraction

# ADS Evaluation 

#DAIOD+ADS OD 
time_ref_start = time.time()
DAIOD_ADS_perimeter  = post.eval_perimeterADS3D(X_ADS_Topocentric_Observation, Obs_perimeter_norm, tgrid, time_idxs=None, state_dim=6) #Structure: DAIOD_ADS_perimeter['final_map'][time_idx] : (Number of perimeter points, state number, subdomain number)[time_idx]
print("ADS Evaluation 1 complete")
Method_Clock_time[0,2] = time.time() - time_ref_start

time_ref_start = time.time()
DAIOD_perimeter  = post.eval_perimeterADS3D(X_DAIOD_Topocentric_Observation, Obs_perimeter_norm, tgrid, time_idxs=None, state_dim=6)
Method_Clock_time[1,2] = time.time() - time_ref_start
print("ADS Evaluation 2 complete")
#DA Evaluation
time_ref_start = time.time()
DAIOD_DA_perimeter_instance  = np.zeros((Obs_perimeter_norm.shape[0], X_DAIOD_DA_Topocentric.shape[0], 1))  # Structure: (Points x State x 1)[Time_idx]
DAIOD_DA_perimeter = [DAIOD_DA_perimeter_instance]
for i in range(len(tgrid)):    #For i'th time 
    n_points = Obs_perimeter_norm.shape[0]
    for k in range(n_points):                   # loop over each k'th point along 6D perimeter norm
        eval_point = Obs_perimeter_norm[k,:]    
        DAIOD_DA_perimeter_instance[k,:,0] = X_DAIOD_DA_Topocentric[:,i].eval(eval_point)

    DAIOD_DA_perimeter.append(DAIOD_DA_perimeter_instance)      # List[(Perimeter Points x States)]

Method_Clock_time[2,2] = time.time() - time_ref_start  
print("ADS / DA Evaluation complete")

# Monte Carlo Evaluation / Statisitcs

print("Beginning Monte Carlo Evaluation")
#MC DAIOD
time_ref_start = time.time()
DAIOD_MC_uncertanities = PW.compute_uncertainties_all(X0_Obj_observables_MC_propagated, tgrid, X_HELIO_observation)     #
RA_DEC_Cov = DAIOD_MC_uncertanities['Cov_RADEC']                                                                          # Covariance Matricies in RA/DEC space at each time step

# P_angle has shape (Ts, 2, 2), with order [RA, DEC]
RA_sigma_DAIOD_MC_prop = DAIOD_MC_uncertanities['RA_sigma']                      # radians (needed for Query)
DEC_sigma_DAIOD_MC_prop = DAIOD_MC_uncertanities['DEC_sigma']                    # radians (needed for Query)
mean_observables_DAIOD = DAIOD_MC_uncertanities['mu_observables']                       # Mean Observables at each time step (Ts, 3) [RA, DEC, rho]

#Check if MC conversion correct.
assert np.isclose(mean_observables_DAIOD[0,0], RA_rad[1], rtol=1e-3)
assert np.isclose(mean_observables_DAIOD[1,0], DEC_rad[1], rtol=1e-3)
"""
#RSW ellipsoid definition
# 3D position ellipsoid in RSW (km)
k = -1  # Last time step
center_pos, (Xpos, Ypos, Zpos), axes_pos, V_pos = PW.rsw_cov_ellipsoid_3d(
    mu_RSW_k = DAIOD_MC_uncertanities["mu_RSW"][:, k],
    P_RSW_k  = DAIOD_MC_uncertanities["P_RSW"][k],
    conf=0.95,
    n_theta=60,
    n_phi=30,
    block="pos",
)

Method_Clock_time[3,2] = time.time() - time_ref_start

print("Monte Carlo Evaluation Gauss")
"""

#MC GAUSS
time_ref_start = time.time()
GAUSS_MC_uncertanities = PW.compute_uncertainties_all(X0_Obj_observables_GAUSS_MC_propagated, tgrid, X_HELIO_observation)
RA_sigma_GAUSS_MC_prop = GAUSS_MC_uncertanities['RA_sigma']                      # radians (needed for Query)
DEC_sigma_GAUSS_MC_prop = GAUSS_MC_uncertanities['DEC_sigma']                    # radians (needed for Query)
mean_observables_GAUSS = GAUSS_MC_uncertanities['mu_observables']                       # Mean Observables at each time step (Ts, 3) [RA, DEC, rho]

Method_Clock_time[4,2] = time.time() - time_ref_start
print("Monte Carlo Evaluation Complete")
###################################### Alphashape definition of ADS and MC perimeters for CADC Query ######################################
"""
    Single DA map does not need Alphashape as no splits in domains
"""
#ADS
#DAIOD+ADS + ADS propagation
time_ref = time.time()
alphashape_DAIODADS_ADS_list = []      # List containing the time instances
DAIOD_ADS_ADS_RADEC_area = []
RA_DEC_DAIODADS_ADS_list = []
for i in range(len(tgrid)):    #For i'th time
    manifold = DAIOD_ADS_perimeter['final_map'][i]
    points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    unique_points = points['unique points']  # Remove duplicate points

    RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    alpha = optalpha.optimizealpha(RA_DEC_CCW)

    alphashape_DAIODADS, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
    RA_DEC_DAIODADS_ADS = unique_points
    area = alphashape_DAIODADS.area

    #Append Matricies to time_idx list
    alphashape_DAIODADS_ADS_list.append(alphashape_DAIODADS)    # time instances are the list index
    DAIOD_ADS_ADS_RADEC_area.append(area)
    RA_DEC_DAIODADS_ADS_list.append(RA_DEC_DAIODADS_ADS)

Method_Clock_time[0,4] = time.time() - time_ref
#DAIOD + ADS propagation
time_ref = time.time()
alphashape_DAIOD_ADS_list = []
RA_DEC_DAIOD_ADS_list = []

for i in range(len(tgrid)):    #For i'th time
    manifold = DAIOD_perimeter['final_map'][i]
    points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    unique_points = points['unique points']  # Remove duplicate points

    RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    alpha = optalpha.optimizealpha(RA_DEC_CCW)

    alphashape_DAIOD_ADS_instance, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
    RA_DEC_DAIOD_ADS = unique_points

    RA_DEC_DAIOD_ADS_list.append(RA_DEC_DAIOD_ADS)
    alphashape_DAIOD_ADS_list.append(alphashape_DAIOD_ADS_instance)
Method_Clock_time[1,4] = time.time() - time_ref

#MC DAIOD
time_ref = time.time()
alphashape_DAIOD_MC_list = []
RA_DEC_DAIOD_MC_list = []
for i in range(len(tgrid)):    #For i'th time
    manifold = X0_Obj_observables_MC_propagated[:,:,i] 
    points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    unique_points = points['unique points']                     # Remove duplicate points


    RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    alpha = optalpha.optimizealpha(RA_DEC_CCW)
    alphashape_DAIOD_MC_instance, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
    
    #Append Matricies to lists
    RA_DEC_DAIOD_MC_list.append(RA_DEC_CCW)
    alphashape_DAIOD_MC_list.append(alphashape_DAIOD_MC_instance)

Method_Clock_time[3,4] = time.time() - time_ref

#MC GAUSS 
time_ref = time.time()
alphashape_GAUSS_MC_list = []
RA_DEC_GAUSS_MC_list = []
for i in range(len(tgrid)):    #For i'th time
    manifold = X0_Obj_observables_GAUSS_MC_propagated[:,:,i]
    points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    unique_points = points['unique points']

    RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    alpha = optalpha.optimizealpha(RA_DEC_CCW)

    alphashape_GAUSS_MC_propagated, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)

    RA_DEC_GAUSS_MC_list.append(RA_DEC_CCW)
    alphashape_GAUSS_MC_list.append(alphashape_GAUSS_MC_propagated)
Method_Clock_time[4,4] = time.time() - time_ref

#DA only
time_ref = time.time()
alphashape_DAIOD_list = []      # List containing the time instances
RA_DEC_DAIOD_DA_list =[]
for i in range(len(tgrid)):    #For i'th time
    manifold = DAIOD_DA_perimeter[i]
    points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    unique_points = points['unique points']  # Remove duplicate points

    RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    alpha = optalpha.optimizealpha(RA_DEC_CCW)
    alphashape_DA, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)

    RA_DEC_DAIOD_DA_list.append(RA_DEC_CCW)
    alphashape_DAIOD_list.append(alphashape_DA)
Method_Clock_time[2,4] = time.time() - time_ref


################################### Apophis Query  ################################
"""
    This section would Query CADC for Apophis Images.
    - It would manage the retrieval and counting of Images for MC, DAIOD+ADS_DA, DAIOD+ADS_ADS, DAIOD_DA. (Fixed Area Image Retrival) -> Total Images
    - The 'true' location via Horizon Ephemeris Collection of Apophis converted to RA-DEC coordinates. -> Defines TP
    - The Total Images within SSOIS + uncertanity -> Control/Baseline Images 'truth'
    - Define FP + TP /TP Total Images / True Image (Lowest is best), Define FN 
"""    
# Fixed Area Image Retrival
# DAIOD+ADS_ADS
tgrid_mjd = Time(np.linspace(tstart.mjd, tend.mjd, Ts), format='mjd', scale='ut1')
N_images = np.zeros((len(N_arcs), Ts, 5))  # Structure: (Arc, Time, Method)
SSOIS_Nimages = np.zeros((Ts))  # Structure: (Time)
TP = np.zeros((len(N_arcs), Ts, 5))            # Structure: (Arc, Time, Method)
FP = np.zeros((len(N_arcs), Ts, 5))            # Structure: (Arc, Time, Method)
FN = np.zeros((len(N_arcs), Ts, 5))            # Structure: (Arc, Time, Method)
precision = np.zeros((len(N_arcs), Ts, 5))
recall = np.zeros((len(N_arcs), Ts, 5))

truth_apohpis_heliocentric = np.zeros((3, Ts))  # Structure: (X, Y, Z, Time) - used to plot 'truth' projection XxY.
truth_apophis_radec = np.zeros((2, Ts))         # Structure: (RA, DEC, Time) - used to plot 'truth' projection RAxDEC coverage

for i in range(len(tgrid_mjd.value)):
    tgrid_mjd2 = tgrid_mjd[i] + TimeDelta(10, format='s')  # Add 10 seconds to end time to ensure inclusivity (avg exposure time)

    print(f"Querying for time idx {i} between {tgrid_mjd[i]} and {tgrid_mjd2}")
    for j in range(len(N_arcs)):
        print(f"Starting Arc Length {N_arcs[j]} days")

        #Obtain the Full Query Metrics for plotting @ a ith timestep + jth arc length
        #Logic: Query Alphashape Polygon for all images in timestep
        #       Query SSOIS Apophis database for known Apophis images with Apophis Position error
        #       Assess if any images overlap. 
        #           If so, define True positive
        #           Remaining Images = False Positive 

        time_start = time.time()
        DAIODADS_ADS_metrics = post.full_query(alphashape_DAIODADS_ADS_list[i], tgrid_mjd[i], tgrid_mjd2)
        #Save into arrays for later plotting 
        N_images[j,i,0] = DAIODADS_ADS_metrics['Total_Images']
        TP[j,i,0] = DAIODADS_ADS_metrics['TP']
        FP[j,i,0] = DAIODADS_ADS_metrics['FP']
        FN[j,i,0] = DAIODADS_ADS_metrics['FN']
        precision[j,i,0] = DAIODADS_ADS_metrics['precision']
        recall[j,i,0] = DAIODADS_ADS_metrics['recall']
        Method_Clock_time[0,4] += time.time() - time_start
        print(f"Completed CADC+ESO Query for DAIOD+ADS_ADS at time idx {i}")


        time_start = time.time()
        DAIOD_ADS_metrics = post.full_query(alphashape_DAIOD_ADS_list[i], tgrid_mjd[i], tgrid_mjd2)
        #Save into arrays for later plotting
        N_images[j,i,1] = DAIOD_ADS_metrics['Total_Images']
        TP[j,i,1] = DAIOD_ADS_metrics['TP']
        FP[j,i,1] = DAIOD_ADS_metrics['FP']
        FN[j,i,1] = DAIOD_ADS_metrics['FN']
        precision[j,i,1] = DAIOD_ADS_metrics['precision']
        recall[j,i,1] = DAIOD_ADS_metrics['recall']
        Method_Clock_time[1,4] += time.time() - time_start
        print(f"Completed CADC+ESO Query for DAIOD+ADS at time idx {i}")

        time_start = time.time()
        DAIOD_metrics = post.full_query(alphashape_DAIOD_list[i], tgrid_mjd[i], tgrid_mjd2)
        N_images[j,i,1] = DAIOD_metrics['Total_Images']
        TP[j,i,1] = DAIOD_metrics['TP']
        FP[j,i,1] = DAIOD_metrics['FP']
        FN[j,i,1] = DAIOD_metrics['FN']
        precision[j,i,1] = DAIOD_metrics['precision']
        recall[j,i,1] = DAIOD_metrics['recall']
        Method_Clock_time[2,4] += time.time() - time_start
        print(f"Completed CADC+ESO Query for DAIOD at time idx {i}")

        time_start = time.time()
        DAIOD_MC_metrics = post.full_query(alphashape_DAIOD_MC_list[i], tgrid_mjd[i], tgrid_mjd2)
        N_images[j,i,3] = DAIOD_MC_metrics['Total_Images']
        TP[j,i,1] = DAIOD_MC_metrics['TP']
        FP[j,i,1] = DAIOD_MC_metrics['FP']
        FN[j,i,1] = DAIOD_MC_metrics['FN']
        precision[j,i,1] = DAIOD_MC_metrics['precision']
        recall[j,i,1] = DAIOD_MC_metrics['recall']
        Method_Clock_time[3,4] += time.time() - time_start
        print(f"Completed CADC+ESO Query for DAIOD_MC time idx {i}")

        GAUSS_MC_metrics = post.full_query(alphashape_GAUSS_MC_list[i], tgrid_mjd[i], tgrid_mjd2)
        N_images[j,i,4] = GAUSS_MC_metrics['Total_Images']
        TP[j,i,4] = GAUSS_MC_metrics['TP']
        FP[j,i,4] = GAUSS_MC_metrics['FP']
        FN[j,i,4] = GAUSS_MC_metrics['FN']
        precision[j,i,4] = GAUSS_MC_metrics['precision']
        recall[j,i,4] = GAUSS_MC_metrics['recall']
        Method_Clock_time[4,4] += time.time() - time_start
        print(f"Completed CADC+ESO Query for GAUSS_MC time idx {i}")
        

        #Obtain Apophis 'truth' Ephermis For RAxDEC plots

        #Convert tstart from isot to 'YYYY-MM-DD'
        t0 = tgrid.isot
        t0 = f"{t0.ymd}"
        tf = tgrid_mjd2.isot
        tf = f"{tf.ymd}"
        t_step = tf

        eph_apophis, vec_apophis = post.load_Apophis_Ephemeris(t0, tf, t_step)
        eph_apophis_helio, vec_apophis_helio = post.load_Apophis_Ephemeris(t0, tf, t_step, location='500@10')
        
        #save in numpy array for plots
        X = vec_apophis_helio['X']
        Y = vec_apophis_helio['Y']
        Z = vec_apophis_helio['Z']

        RA_apophis = eph_apophis_helio['RA']
        DEC_apophis = eph_apophis_helio['DEC']

        truth_apohpis_heliocentric[:,i] = np.array([X[0].to(u.km).value, Y[0].to(u.km).value, Z[0].to(u.km).value])
        truth_apophis_radec[:,i] = np.array([RA_apophis[0].to(u.deg).value, DEC_apophis[0].to(u.deg).value])
        print(f"Completed Apophis Ephemeris for time idx {i}")


    

#################################### Plots - Fixed Arc Length ####################################


#ADS nsplit history in propagation -> Observe Computional Intensitiy (Wittig style)
save_dir_nsplit = "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\nsplit_history"
filename = "ADS_nsplit_history.png"
nsplit_DAIOD_ADS_ADS = np.zeros((Ts))
nsplit_DAIOD_ADS = np.zeros((Ts))

nsplit_DAIOD_ADS_ADS[0]= len(X_ADS_Topocentric_Observation[0])
nsplit_DAIOD_ADS[0] = 1
for i in range(Ts-1):
    nsplit_DAIOD_ADS_ADS[i+1]=len(X_ADS_Topocentric_Observation[i]) #Domain Split history for DAIOD+ADS OD ADS prop: Count number of subdomains
    nsplit_DAIOD_ADS[i+1] = len(final_lists_ADS_DAIOD[i])           #Domain Split History for DAIOD OD ADS prop: Count number of subdomains

fig, ax = plt.subplots()
ax.plot(tgrid, nsplit_DAIOD_ADS, 'b--')
ax.plot(tgrid, nsplit_DAIOD_ADS_ADS, 'r-')
ax.grid('minor',  linestyle=':')

ax.set_xlabel('propagation time (s)')
ax.set_ylabel('number of domains (-)')
ax.set_title('Domain Split History during ADS Propagation')
ax.legend(['DAIOD+ADS Orbit Determination', 'DAIOD Orbit Determination'])

#save 
os.makedirs(save_dir_nsplit, exist_ok=True)
out_path = os.path.join(save_dir_nsplit, filename)
fig.savefig(out_path, dpi=300, bbox_inches="tight")


#Wittig Style but for RA-DEC plane @ time instance -> Observe Coverage Growth (Download to Figures folder)
    #Plot points
    #Plot Alphashape geometry
def plot_RA_DEC(RA_DEC_list: list[np.ndarray], alphashape_list, date: str, filename: str, save_dir: str, panel_titles=("DAIOD+ADS OD", "DAIOD OD ADS Propagation", "DAIOD DA", "DAIOD MC", "GAUSS MC"), alphas=None, xlim=(-90, 90), ylim=(0,360), show=False):

    """
        Plot the data points on the RA DEC plane 
        - Scatter plot of RA vs DEC
        - Overlay of alpha-shape geometry if it exists
    :params RA_DEC_list: A list of RA_DEC points. Structure: List[(RA, DEC)] shape: list[(n,2)]
    """

    assert len(RA_DEC_list) == 5 and len(panel_titles) == 5, \
        "Provide exactly 5 items for points_list, shapes_list, and panel_titles."
    


    if alphas is None:
        alphas = [None]*5 

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax in axes: ax.set_visible(False)

    for i in range(5):
        ax = axes[i]
        ax.set_visible(True)

        #Points
        ax.plot(RA_DEC_list[i][:,1], RA_DEC_list[i][:, 0], 'o', label='input points')
        if alphashape_list[i] is not None:
            ax.plot(*alphashape_list[i].exterior.xy, 'r--', label = f'alphashape: {0}')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(panel_titles[i])
        ax.set_xlabel('DEC (deg)')
        ax.set_ylabel('RA (deg)')
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.set_aspect('equal')
        ax.legend(loc='best')

    fig.suptitle(f"Right Ascension and Declination Coverage @ {date}")

    #save 
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path

save_dir = "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\RA_DEC_Coverage"
for i in range(len(tgrid)):                 #For each time instance plot the RAxDEC plane (alphashape + scatter) in subplots
    RA_DEC_list = [RA_DEC_DAIODADS_ADS_list[i], RA_DEC_DAIOD_ADS_list[i], RA_DEC_DAIOD_DA_list[i], RA_DEC_DAIOD_MC_list[i], RA_DEC_GAUSS_MC_list[i]] # Obtain RA-DEC points at this time instance
    alphashape_list = [alphashape_DAIODADS_ADS_list[i], alphashape_DAIOD_ADS_list[i], alphashape_DAIOD_list[i], alphashape_DAIOD_MC_list[i], alphashape_GAUSS_MC_list[i]] # Obtain Alphashape geometry at this time instance

    # Convert tgrid to string date
    date = (tstart + TimeDelta(tgrid[i]*u.s)).iso.split(' ')[0]
    filename = f"RAxDEC_{date}.png"
    plot_RA_DEC(RA_DEC_list, alphashape_list, date, filename, save_dir, show=False)

# panel_titles=("DAIOD+ADS OD", "DAIOD OD ADS Propagation", "DAIOD DA", "DAIOD MC", "GAUSS MC")

#Witting Style 2D Orbit projection Cartesian uncertanity
    # Convert Evaluated / MC points from observer (Topocentric) frame to Cartesian ECI frame
    # Rotate so that z axis = ang momentum vector -> plot.state_to_orbital_frame
    # Plot each timestep on same figure till tgrid = period -> time_ref.CC2COE() = COE 
    # New figure if tgrid[i] == period
    # Plot the central body at 0,0
def topocentric_to_eci_cartesian(topocentric_points, observer_eci_position):
    """
    Convert topocentric Cartesian points to ECI Cartesian.
    
    :param topocentric_points: Array of shape (6, n_points) or (n_points, 6) - [x, y, z, vx, vy, vz]
    :param observer_eci_position: Observer's ECI position [6,] - [x, y, z, vx, vy, vz]
    :return: ECI Cartesian points [6, n_points]
    """
    if topocentric_points.ndim == 1:
        topocentric_points = topocentric_points.reshape(-1, 1)
    elif topocentric_points.shape[0] != 6:
        topocentric_points = topocentric_points.T  # Ensure shape (6, n_points)
    
    eci_points = topocentric_points + observer_eci_position.reshape(-1, 1)
    return eci_points

def compute_orbital_period(eci_state, mu):
    coe = time_ref_func.CC2COE(eci_state[:3], eci_state[3:], mu)
    a = coe[0]
    period = 2*np.pi*np.sqrt(a**3/mu)
    return period



def plot_orbital_projection(method_data, method_name, tgrid, observer_eci_position, mu, method_dir):
    """
    :params method_data: the propagate manifolds List[(Perimeter_points, 6, N_subdomains)]
    :params method_name: The technique used
    :params tgrid: propagation grid
    :params observer_eci_position: Observer's ECI position
    :params mu: Gravitational parameter
    :params method_dir: Directory to save plots
    """
    
    if not isinstance(method_data, list):
        #Extract 2D matrix
        for i in range(method_data.shape[2]):       #Cycle Through Manifolds
            method_data_2d = method_data[:,:,i]
            method_data_2d = method_data_2d.reshape(method_data_2d.shape[0], 6, 1)  #Get X,Y
            method_data.append(method_data_2d)

    manifold_0 = method_data[0]
    manifold_0 = manifold_0.reshape(-1, 6)    
    for i in range(manifold_0.shape[0]):
        eci_initial = topocentric_to_eci_cartesian(manifold_0[i,:], observer_eci_position)
    
    eci_initial_mean = np.mean(eci_initial, axis=1)
    period = compute_orbital_period(eci_initial_mean, mu)

    fig_num = 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(f'2D Orbital Projection - {method_name} (Figure {fig_num})')
    ax.grid(True)
    ax.axis('equal')
    ax.plot(0, 0, 'ko', markersize=10, label='Central Body')  # Central body at origin
    
    cumulative_time = 0
    for i in range(len(tgrid)):
        timestep_time = tgrid[i] - (tgrid[i-1] if i > 0 else 0)
        cumulative_time += timestep_time

        if cumulative_time > period:
            #Save current figure and start a new one
            filepath = os.path.join(method_dir, f'Figure_{fig_num}.png')
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig_num += 1
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_title(f'2D Orbital Projection - {method_name} (Figure {fig_num})')
            ax.grid(True)
            ax.axis('equal')
            ax.plot(0, 0, 'ko', markersize=10, label='Central Body')  # Central body at origin
            cumulative_time = timestep_time  # Reset cumulative time to current timestep
        
        manifold = method_data[i]       #Topocentric Spgerical
        # Convert to ECI Cartesian
        eci_manifold = np.zeros_like(manifold)
        eci_CC_manifold = np.zeros_like(manifold)
        eci_CC_manifold_2d = np.zeros((manifold.shape[0], 2))  # For 2D points
        for k in range(manifold.shape[2]):  # For each subdomain
            for j in range(manifold.shape[0]): #each point
                # Spherical to Cartesian
                tmp_manifold= time_ref_func.obs2CC(manifold[j,:,k])     #cartesian topocentric
                
                eci_tmp = topocentric_to_eci_cartesian(tmp_manifold, observer_eci_position) #eci cartesian
                eci_CC_manifold[j,:,k] = eci_tmp.reshape(-1)  # Ensure shape is (6,)

                #Rotate to Angular Momentum vector
                eci_CC_manifold[j,:,k], _ = plot.state_to_orbital_frame(eci_CC_manifold[j,:,k], mu)

        points_2d = post.extract_2D_points(eci_CC_manifold, 0, [0,1], False)  # Ensure shape is (N_points, 2)
        eci_CC_manifold_2d = points_2d['unique points']  # Remove duplicate points
        label = "{:.2f}".format(tgrid[i]/(60/60)) + " hrs"
        ax.plot(eci_CC_manifold_2d[:, 0], eci_CC_manifold_2d[:, 1], label=label)

        #Plot Apophis 'Truth'
        ax.plot(truth_apohpis_heliocentric[0,i], truth_apohpis_heliocentric[1,i], 'gx', markersize=8, label='Apophis Truth' if i==0 else "")

        if eci_CC_manifold_2d.shape[0] > 0:
            ax.annotate(label, 
                        (eci_CC_manifold_2d[0, 0], eci_CC_manifold_2d[0, 1]), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=8)

    filepath = os.path.join(method_dir, f'Orbital_Projection_{method_name}_Figure_{fig_num}.png')
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

#plot_orbital_projection(DAIOD_ADS_perimeter['final_map'], "DAIOD+ADS ADS Propagation", tgrid, X_ECI_observation_epoch,mu, "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\Orbit_projection\\DAIOD+ADS ADS")
#plot_orbital_projection(DAIOD_perimeter['final_map'], "DAIOD ADS Propagation", tgrid, X_ECI_observation_epoch,mu, "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\Orbit_projection\\DAIOD ADS")
# plot_orbital_projection(DAIOD_DA_perimeter['final_map'], "DAIOD_DA Propagation", tgrid, X_ECI_observation_epoch,mu, "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\Orbit_projection\\DAIOD DA")
#plot_orbital_projection(X0_Obj_observables_MC_propagated, "PW DAIOD MC Propagation", tgrid, X_ECI_observation_epoch,mu, "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\Orbit_projection\\DAIOD MC")
#plot_orbital_projection(X0_Obj_observables_GAUSS_MC_propagated, "PW Gauss MC Propagation", tgrid, X_ECI_observation_epoch,mu, "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\Orbit_projection\\Gauss MC")

#Clock-time vs Method Cumilative Coloured bar chart -> Observe Method Speed Efficiency
def plot_cumulative_bar_chart(Method_Clock_time, output_dir):
    """
    Plot a stacked bar chart for cumulative processing times.
    
    :param Method_Clock_time: 5x5 cumulative array
    :param output_dir: Directory to save the plot
    """
    methods = ['DAIOD+ADS_ADS', 'DAIOD_+ADS', 'DAIOD_DA', 'PW_MC', 'PW_gauss']
    phases = ['OD', 'Prop', 'Eval', 'Conversion', 'Query']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(methods))
    
    for phase_idx, phase in enumerate(phases):
        phase_times = Method_Clock_time[:, phase_idx]
        ax.bar(methods, phase_times, bottom=bottom, label=phase, color=colors[phase_idx])
        bottom += phase_times
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Cumulative Clock Time (seconds)')
    ax.set_title('Cumulative Processing Times of Candidate Methods')
    ax.legend(title='Phases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'CumulativeProcessingTimesBarChart.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved cumulative bar chart to {filepath}')


output_dir = "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures"
plot_cumulative_bar_chart(Method_Clock_time, output_dir)



#Clock-time vs tgrid @ Methods -> Observe Clock time growth with simulation time
# ... (after your loops) ... TODO: Impliment this to see where within the propagation time this occurs

def plot_wall_clock_line(method_clock_times_per_step, tgrid, output_dir='./processing_times_plots/'):
    """
    Plot a line chart for cumulative wall clock time per method vs. tgrid.
    
    :param method_clock_times_per_step: List of 5x5 arrays (one per tgrid index)
    :param tgrid: Time grid array
    :param output_dir: Directory to save the plot
    """
    methods = ['DAIOD+ADS_ADS', 'DAIOD_+ADS', 'DAIOD_DA', 'PW_MC', 'PW_gauss']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For each method, plot cumulative time vs. tgrid
    for method_idx, method in enumerate(methods):
        cumulative_times_method = [times[method_idx, :].sum() for times in method_clock_times_per_step]  # Sum across phases per step
        ax.plot(tgrid[:len(cumulative_times_method)], cumulative_times_method, label=method, color=colors[method_idx])
        name = str(len(cumulative_times_method))

    ax.set_xlabel('Propagation Time (seconds)')
    ax.set_ylabel('Cumulative Wall Clock Time (seconds)')
    ax.set_title('Wall Clock Time vs. Propagation Time for Each Method')
    ax.legend(title='Methods')
    ax.grid(True)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'WallClockTimeLinePlot_{name}.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved line plot to {filepath}')

# Call the function
output = "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\clocktime_proptime"

#Error = DAIOD ADS vs MC
def plot_DAIOD_error(DAIOD_perimeter, MC_perimeter, tgrid, output_dir):
    """
    Plot the error between DAIOD ADS and MC perimeters over time.
    
    :param DAIOD_perimeter: List of DAIOD perimeter arrays
    :param MC_perimeter: List of MC perimeter arrays
    :param tgrid: Time grid array
    :param output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    errors = []
    for i in range(len(tgrid)):
        daiod_points = DAIOD_perimeter[i].reshape(-1, 6)  # Flatten to (n_points, 6)
        mc_points = MC_perimeter[i].reshape(-1, 6)        # Flatten to (n_points, 6)
        
        # Compute mean squared error in RA/DEC space
        daiod_ra_dec = daiod_points[:, :2]  # Assuming first two columns are RA and DEC
        mc_ra_dec = mc_points[:, :2]
        
        # Interpolate MC points to DAIOD points for fair comparison
        from scipy.spatial import cKDTree
        tree = cKDTree(mc_ra_dec)
        distances, _ = tree.query(daiod_ra_dec)
        
        mse = np.mean(distances**2)
        errors.append(mse)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tgrid, errors, marker='o')
    ax.set_xlabel('Propagation Time (seconds)')
    ax.set_ylabel('Mean Squared Error in RA/DEC (degrees^2)')
    ax.set_title('Error between DAIOD ADS and MC Perimeters Over Time')
    ax.grid(True)
    
    filepath = os.path.join(output_dir, 'DAIOD_ADS_vs_MC_Error.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved error plot to {filepath}')


#TP, FP, FN of each method vs time
def plot_classification_metrics_vs_time(TP, FP, FN, tgrid, output_dir):
    """
    Plot Precision, Recall, and F1 Score vs. time for each method.
    
    :param TP: True Positives array (N_arcs, Ts, 5)
    :param FP: False Positives array (N_arcs, Ts, 5)
    :param FN: False Negatives array (N_arcs, Ts, 5)
    :param tgrid: Time grid array
    :param output_dir: Directory to save the plots

    P
    """
    os.makedirs(output_dir, exist_ok=True)
    
    methods = ['DAIOD+ADS_ADS', 'DAIOD_+ADS', 'DAIOD_DA', 'PW_MC', 'PW_gauss']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    
    #generate precision, recall, f1 score arrays
    precision = np.zeros((len(tgrid), len(methods)))
    recall = np.zeros((len(tgrid), len(methods)))
    f1_score = np.zeros((len(tgrid), len(methods)))

    for method_idx, method in enumerate(methods):
        
        for i in range(len(tgrid)):
            tp = TP[:, i, method_idx].sum()
            fp = FP[:, i, method_idx].sum()
            fn = FN[:, i, method_idx].sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            precision[i, method_idx] = prec
            recall[i, method_idx] = rec
            f1_score[i, method_idx] = f1

        #Plot each methods precision, recall, f1 
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(tgrid, precision[:, method_idx], '--', label=methods[method_idx], color=colors[method_idx])

        fig2, ax2 = plt.subplots(figure=(10, 6))
        ax2.scatter(tgrid, recall[:, method_idx], label=methods[method_idx], color=colors[method_idx])

        fig3, ax3 = plt.subplots(figure=(10, 6))
        ax.scatter(tgrid, f1_score[:,method_idx], label=methods[method_idx], color=colors[method_idx])
           
    ax.set_xlabel('Propagation Time (seconds)')
    ax.set_ylabel('Precision (-)')
    ax.set_title(f'Level of Precision')
    ax.legend()
    ax.grid(True)

    ax2.set_xlabel('Propagation Time (seconds)')
    ax2.set_ylabel('Recall (-)')
    ax2.set_title(f'Rate of Recall')
    ax2.legend()
    ax2.grid(True)

    ax3.set_xlabel('Propagation Time (seconds)')
    ax3.set_ylabel('F1 Score (-)')
    ax3.set_title(f'F1 Score')
    ax3.legend()
    ax3.grid(True)

    #Save plots
    for method_idx, method in enumerate(methods):
        filepath = os.path.join(output_dir, f'Precision_vs_Time_{method}.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        filepath = os.path.join(output_dir, f'Recall_vs_Time_{method}.png')
        fig2.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        filepath = os.path.join(output_dir, f'F1_Score_vs_Time_{method}.png')
        fig3.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig3)


plot_classification_metrics_vs_time(TP, FP, FN, tgrid, output_dir="C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\06_Thesis_graphs\\Figures\\Metrics")


