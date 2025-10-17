import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

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
import utils.plotting as plot

import pickle
import os
from datetime import datetime
from pathlib import Path

""" Main Script for Orbit Determination and Propagation
    Compares DAIOD+ADS, DAIOD+DA, PW Gauss IOD, PW DAIOD MC
        - Orbit Determination + Propagation = Equatorial Frame
        - """
def linspace_with_required(start, end, n, required_times, *, scale="utc", dedup_tol=1e-9):
    """
    Build a Time array with n evenly spaced points between start and end,
    and ensure all required_times are included. Duplicates within dedup_tol JD are removed.
    """
    # Normalize inputs
    start = Time(start, scale=scale)
    end   = Time(end,   scale=scale)
    base  = Time(np.linspace(start.jd, end.jd, n), format="jd", scale=scale)

    # Required times (can be strings, datetimes, or Time)
    req = Time(required_times, scale=scale)
    # Keep only those inside [start, end]
    mask = (req.jd <= start.jd) & (req.jd >= end.jd)
    req  = req[mask]

    # Merge, sort, and de-dup within tolerance
    jd = np.concatenate([base.jd, req.jd])
    jd.sort()
    jd = jd[::-1]   # Ensure descending order
    out = [jd[0]]
    for v in jd[1:]:
        if abs(v - out[-1]) > dedup_tol:  # ~1e-9 JD ≈ 0.086 ms
            out.append(v)

    return Time(out, format="jd", scale=scale)


#Arc length OD Parameters
dt = np.array([0.18, 0.25, 0.3, 0.5, 1])
dt_astropy = TimeDelta(dt, format='jd')
obs_time0 = Time('2005-06-17 23:41:07', scale='utc') #Central Observation Time Time('2004-01-17 12:53:20', scale='utc')
MonteCarlo_samples = 10000 #Number of Monte Carlo Samples

#Propagation Parameters
tstart = obs_time0
tfinal = tstart - TimeDelta(183, format='jd') #Final Time for Propagation
dt_propagation = TimeDelta(1, format='jd') #Propagation time step
t_required = Time(["2005-05-17 01:20:55","2005-04-17 01:20:55","2005-03-17 01:20:55", "2005-02-17 01:20:55","2005-01-18 01:16:50", "2005-01-17 01:19:11"], scale='utc')       #Timestamps of SSOIS images
t_propagation = linspace_with_required(tstart, tfinal, 183, t_required) #Time grid for propagation including
# enforced timestamps


#Evaluation Parameters
DA_axis_points = 2 #Number of points along each axis for DA perimeter evaluation
order = 6
# Storage Arrays
Method_Clock_time = np.zeros((5,5))
script_dir = Path(__file__).parent
output_dir = script_dir / "Simulation_data"
time1 = time.time()
for m in range(len(dt)): # The Arc length
    
    print(f"Arc Length idx: {m}")
    print(f"Arc Length: {dt[m]} days")

    obs_times = Time([obs_time0 - dt_astropy[m], obs_time0, obs_time0 + dt_astropy[m]])

    #Generate Observer Positions at Observation Times
    mu = ac.G * ac.M_sun
    mu = mu.to('km**3 / s**2').value
    epochs = Time(obs_times, scale='utc')
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

    #Fetch RA, DEC, and observation uncertanity using JPL Horizons
    eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')  #ICRS Geocentric
    _, vec = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1', location='500@10') #ICRS Heliocentric

    _, vec2 = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1', location='500@10')
    eph2, _  = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')

    ra = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad)
    dec = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad)
    apophis_pos = ((np.concatenate([vec['x'], vec2['x'][1:]]) , np.concatenate([vec['y'], vec2['y'][1:]]), np.concatenate([vec['z'], vec2['z'][1:]])) * u.AU).to(u.km)
    apophis_vel = ((np.concatenate([vec['vx'], vec2['vx'][1:]]), np.concatenate([vec['vy'], vec2['vy'][1:]]), np.concatenate([vec['vz'], vec2['vz'][1:]])) * (u.AU/u.d)).to(u.km/u.s)
    apophis_vec = np.concatenate([apophis_pos.value, apophis_vel.value], axis=0) #km

    #Get error from central observation
    ra_sigma = (30*eph['RA_3sigma'][0] * u.arcsec).to(u.rad).value/3 * u.rad
    dec_sigma = (30*eph['DEC_3sigma'][0] * u.arcsec).to(u.rad).value/3 * u.rad

    #Start Orbit Determination Algoithms
    print("DAIOD + ADS")
    time_ref_start = time.time()
    X0_Obj_ECI_DAIOD_ADS = iod.DAIOD_ADS_full(ra.value, dec.value, ra_sigma.value, dec_sigma.value, pos_obs, (obs_times.mjd * u.day).to(u.s).value, mu, order, prograde=True) #Run DAIOD to get Object State
    if X0_Obj_ECI_DAIOD_ADS is None:
        print(f"DAIOD ADS failed at arc length {dt[m]} days")
        X0_Obj_ECI_DAIOD_ADS = None
        continue
    time_ref_end = time.time()
    Method_Clock_time[0,0] = time_ref_end - time_ref_start

    #DAIODADS + DA
    print("DAIOD + DA")
    time_ref_start = time.time()
    try:
        X0_Obj_ECI_DAIOD_DA = iod.DAIOD_full(ra.value, dec.value, ra_sigma.value, dec_sigma.value, pos_obs, (obs_times.mjd * u.day).to(u.s).value, mu, order, prograde=True)       #Run DAIOD to get Object State
    except Exception as e:
        print(f"DA failed at arc length {dt[m]} days: {e}")
        X0_Obj_ECI_DAIOD_DA = None
        continue
    time_ref_end = time.time()

    Method_Clock_time[1,0] = time_ref_end - time_ref_start
    Method_Clock_time[2,0] = time_ref_end - time_ref_start 

    print("Monte Carlo Gauss IOD")
    time_ref_start = time.time()
    try:
        X0_Obj_ECI_DAIOD_MC, X0_Obj_ECI_GAUSS_MC = PW.monte_carlo_gauss_PWiod(MonteCarlo_samples, pos_obs, ra.value, dec.value, (obs_times.mjd * u.day).to(u.s).value, ra_sigma.value, dec_sigma.value, mu, prograde_bool_=True)              #Run PW Gauss IOD to get Object State
    except Exception as e:
        print(f"Monte Carlo Gauss IOD failed at arc length {dt[m]} days: {e}")
        X0_Obj_ECI_DAIOD_MC = None
        X0_Obj_ECI_GAUSS_MC = None
        continue
    time_ref_end = time.time()
    Method_Clock_time[3,0] = time_ref_end - time_ref_start
    Method_Clock_time[4,0] = time_ref_end - time_ref_start 

    print("Orbit Determination Complete")
    
    
    #Propagate DAIOD_ADS, DAIOD_DA, DAIOD_MC, GAUSS_MC results to tfinal

    tgrid = (t_propagation.mjd * u.day).to(u.s).value   #Get Propagation in Programme units
    Ts = len(tgrid)

    #ADS Propagation - DAIOD+ADS

    # final_list = X0_Obj_ECI_DAIOD_ADS.copy()
    # final_lists_ADS=[final_list]
    toll_obs = np.array([ra_sigma.value, ra_sigma.value, ra_sigma.value, 0.1*ra_sigma.value, 0.1*ra_sigma.value, 0.1*ra_sigma.value])
    toll = np.array([100, 100, 100, 0.1, 0.1, 0.1]) #1m and 1mm/s
    Max_split = 1
    
    # DAIOD_ADS_time = time.time()
    # try:
    #     for i in range(len(tgrid)-1):

    #         final_list = ADS.eval(
    #             final_list,
    #             toll,
    #             Max_split,
    #             lambda domain: prop.advanced_propagationADS(domain, tgrid[i], tgrid[i+1], dynamics.TBP_CC_DA)
    #         )
    #         final_lists_ADS.append(final_list)
    #         print(f"Completed propagation to t = {tgrid[i+1]} seconds")
    # except Exception as e:
    #     print(f"DAIOD ADS propagation failed at arc length {dt[m]} days: {e}")
    #     final_lists_ADS = None
    #     continue
    # Method_Clock_time[0,1] = time.time() - DAIOD_ADS_time

    #ADS propagation - DAIOD 
    # DAIOD_ADS_prop_time = time.time()
    # init_domain = ADS(X0_Obj_ECI_DAIOD_DA, [])
    # init_list = [init_domain]
    # final_lists_ADS_DAIOD = []
    # final_list = init_list.copy()
    # final_lists_ADS_DAIOD.append(final_list) # add also initial domains
    # try:
    #     for i in range(len(tgrid)-1):

    #         final_list = ADS.eval(
    #             final_list,
    #             toll,
    #             Max_split,
    #             lambda domain: prop.advanced_propagationADS(domain, tgrid[i], tgrid[i+1], dynamics.TBP_CC_DA)
    #         )
    #         final_lists_ADS_DAIOD.append(final_list)

    #         print(f"Completed propagation to t = {tgrid[i+1]} seconds")
    #         print(f"CURRENT IDX: {i}")
    # except Exception as e:
    #     print(f"DAIOD ADS propagation failed at arc length {dt[m]} days: {e}")
    #     final_lists_ADS_DAIOD = None
    #     continue

    # Method_Clock_time[1,1] = time.time() - DAIOD_ADS_prop_time

    #DA propagation DAIOD
    DAIOD_DA_prop_time = time.time()
    try:
        X0_Obj_ECI_DAIOD_DA_propagated = prop.advanced_propagationDA(X0_Obj_ECI_DAIOD_DA, tgrid, dynamics.TBP_CC_DA) # Everything done inside loop
    except Exception as e:
        print(f"DAIOD DA propagation failed at arc length {dt[m]} days: {e}")
        X0_Obj_ECI_DAIOD_DA_propagated = None
        continue
    Method_Clock_time[2,1] = time.time() - DAIOD_DA_prop_time

    #DA propagation DAIOD_ADS
    DAIOD_ADS_DA_prop_time = time.time()
    try:
        # Precondition DAIOD_ADS object for advanced_propagationDA()
        # Extract the DA manifold from the first ADS domain since advanced_propagationDA expects array, not ADS
        if isinstance(X0_Obj_ECI_DAIOD_ADS, list) and len(X0_Obj_ECI_DAIOD_ADS) > 0:
            # X0_Obj_ECI_DAIOD_ADS is a list of ADS objects, take the manifold from the first one
            X0_DAIOD_ADS_DA_input = X0_Obj_ECI_DAIOD_ADS[0].manifold
        else:
            raise ValueError("DAIOD_ADS returned invalid ADS object structure")
            
        X0_Obj_ECI_DAIOD_ADS_DA_propagated = prop.advanced_propagationDA(X0_DAIOD_ADS_DA_input, tgrid, dynamics.TBP_CC_DA) # Everything done inside loop
    except Exception as e:
        print(f"DAIOD ADS DA propagation failed at arc length {dt[m]} days: {e}")
        X0_Obj_ECI_DAIOD_ADS_DA_propagated = None
        continue
    # Note: We'll need to add a new row to Method_Clock_time or reuse an existing one
    # For now, storing in a separate variable
    DAIOD_ADS_DA_time = time.time() - DAIOD_ADS_DA_prop_time
    Method_Clock_time[0,1] = DAIOD_ADS_DA_time
    #PW Propagation - Monte Carlo (Zeroth order DAIOD)
    print("Beginning Monte Carlo Propagation")
    DAIOD_MC_prop_time = time.time()
    try:
        X0_Obj_ECI_MC_propagated, _ = PW.monte_carlo_propagation(X0_Obj_ECI_DAIOD_MC, tgrid[0], tgrid[-1], Ts)
    except Exception as e:
        print(f"DAIOD MC propagation failed at arc length {dt[m]} days: {e}")
        X0_Obj_ECI_MC_propagated = None
        continue
    Method_Clock_time[3,1] = time.time() - DAIOD_MC_prop_time

    #PW Propagation - Gauss's solution
    Gauss_MC_prop_time = time.time()
    try:
        X0_Obj_ECI_GAUSS_MC_propagated, _ = PW.monte_carlo_propagation(X0_Obj_ECI_GAUSS_MC, tgrid[0], tgrid[-1], Ts)
    except Exception as e:
        print(f"DAIOD Gauss MC propagation failed at arc length {dt[m]} days: {e}")
        X0_Obj_ECI_GAUSS_MC_propagated = None
        continue
    Method_Clock_time[4,1] = time.time() - Gauss_MC_prop_time

    print("Propagation Complete")


    ###############################################Get Observer Position at each propagated time step
    print("Getting Observer Position at each propagated time step")
    obs_times_propagation = t_propagation.tdb
    acoords.solar_system_ephemeris.set("builtin")
    #Get ICRS Equatorial position of the Earth at each observation epoch
    pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in obs_times_propagation]          # pv[i] = (pos, vel)
    #Translate to Heliocentric Equatorial Frame
    pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in obs_times_propagation] 
    pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]  # (pos, vel) relative to Sun
    
    Earth_propagated_position = np.zeros((6,Ts))
    for i, (p_helio, v_helio) in enumerate(pv_earth_helio):
    #Maintain Equaotorial Frame
        pv_earth_helio[i] = np.concatenate((p_helio.xyz.to(u.km).value, v_helio.xyz.to(u.km/u.s).value))  # [x, y, z, vx, vy, vz]
        Earth_propagated_position[:,i] = pv_earth_helio[i]
    
    print("Observer Position Complete")
    ###################################Reference Frame conversion

    # time_ref_start = time.time()
    # X_ADS_geocentric = post.ADS_Helio2GEO(final_lists_ADS, Earth_propagated_position, None)
    # X_ADS_geocentric_obs = post.ADS_Cart_2_Obs(X_ADS_geocentric)
    # Method_Clock_time[0,3] = time.time() - time_ref_start

    # # DAIOD ADS propagation
    # time_ref_start = time.time()
    # X_DAIOD_geocentric = post.ADS_Helio2GEO(final_lists_ADS_DAIOD, Earth_propagated_position, None)
    # X_DAIOD_geocentric_obs = post.ADS_Cart_2_Obs(X_DAIOD_geocentric)
    # Method_Clock_time[1,3] = time.time() - time_ref_start

    # DAIOD DA propagation
    time_ref_start = time.time()
    X_DAIOD_DA_geocentric_obs = array.zeros((6, X0_Obj_ECI_DAIOD_DA_propagated.shape[1]))  # Structure: (States x Time)
    for i in range(X0_Obj_ECI_DAIOD_DA_propagated.shape[1]):    #Cyle through time steps
        X_DAIOD_DA_geocentric = X0_Obj_ECI_DAIOD_DA_propagated[:,i] - Earth_propagated_position[:,i] # [6 x len(prop_time)] 
        X_DAIOD_DA_geocentric_obs[:,i] = time_ref_func.CC2obs(X_DAIOD_DA_geocentric)  # Convert to Observables
        Method_Clock_time[2,3] = time.time() - time_ref_start

    # DAIOD_ADS DA propagation
    time_ref_start = time.time()
    X_DAIOD_ADS_DA_geocentric_obs = array.zeros((6, X0_Obj_ECI_DAIOD_ADS_DA_propagated.shape[1]))  # Structure: (States x Time)
    for i in range(X0_Obj_ECI_DAIOD_ADS_DA_propagated.shape[1]):    #Cyle through time steps
        X_DAIOD_ADS_DA_geocentric = X0_Obj_ECI_DAIOD_ADS_DA_propagated[:,i] - Earth_propagated_position[:,i] # [6 x len(prop_time)] 
        X_DAIOD_ADS_DA_geocentric_obs[:,i] = time_ref_func.CC2obs(X_DAIOD_ADS_DA_geocentric)  # Convert to Observables
    # Note: Time will be stored in DAIOD_ADS_DA_frame_conversion_time for consistency
    DAIOD_ADS_DA_frame_conversion_time = time.time() - time_ref_start
    Method_Clock_time[0,3] = DAIOD_ADS_DA_frame_conversion_time
    try:
    # PW Conversion - DAIOD formulation
        time_ref_start = time.time()
        X_DAIOD_MC_geocentric_obs = np.zeros_like(X0_Obj_ECI_MC_propagated)
        for i in range(X0_Obj_ECI_MC_propagated.shape[2]):
            for j in range(X0_Obj_ECI_MC_propagated.shape[0]):
                X_DAIOD_MC_geocentric_Cart = X0_Obj_ECI_MC_propagated[j, :, i] - Earth_propagated_position[:,i]
                X_DAIOD_MC_geocentric_obs[j, :, i] = time_ref_func.CC2obs(X_DAIOD_MC_geocentric_Cart)
    except Exception as e:
        print(f"DAIOD MC conversion to obs failed at arc length {dt[m]} days: {e}")
        X_DAIOD_MC_geocentric_obs = np.nan * np.ones((1,6,1))

    Method_Clock_time[3,3] = time.time() - time_ref_start

    # PW Conversion - Gauss
    time_ref_start = time.time()
    try:
        X_GAUSS_MC_geocentric_obs = np.zeros_like(X0_Obj_ECI_GAUSS_MC_propagated)
        for i in range(X0_Obj_ECI_GAUSS_MC_propagated.shape[2]):
                for j in range(X0_Obj_ECI_GAUSS_MC_propagated.shape[0]):
                    X0_GAUSS_MC_geocentric_Cart = X0_Obj_ECI_GAUSS_MC_propagated[j, :, i] - Earth_propagated_position[:,i]
                    X_GAUSS_MC_geocentric_obs[j, :, i] = time_ref_func.CC2obs(X0_GAUSS_MC_geocentric_Cart)
    except Exception as e:
        print(f"Gauss MC conversion to obs failed at arc length {dt[m]} days: {e}")
        X_GAUSS_MC_geocentric_obs = np.nan * np.ones((1,6,1))
        
    Method_Clock_time[4,3] = time.time() - time_ref_start

    ######### DA / MC Apophis Propagation error #########

    # Get "True" Apophis Position at each propagated time step
    #for i in range(len(tgrid)):
    #    t_propagation[i].format = 'iso'
    #    Apo = post.load_Apophis_Ephemeris(t_propagation[i], t_propagation[i], step='1', location='500@10')  #ICRS Geocentric

    #    DA_cons = X0_Obj_ECI_DAIOD_DA_propagated.cons()
    #    ADS_cons = post.calculate_ads_mean_single_time(final_lists_ADS)
    #    ADS_DAIOD_cons = post.calculate_ads_mean_single_time(final_lists_ADS_DAIOD)
    #    error_DA = np.linalg.norm(Apo[:3] - DA_cons[:3])
    #    error_ADS = np.linalg.norm(Apo[:3] - ADS_cons[:3])


    ######### ADS + DA Evaluation #########
    print("Beginning ADS / DA Evaluation")
    # need to update the ra and dec to be at the propagated time steps or centre points
    Obs_perimeter_norm = post.gen_grid6D(ra_sigma.value, dec_sigma.value, DA_axis_points)

    #DAIOD+ADS OD 
    # time_ref_start = time.time()
    # DAIOD_ADS_perimeter  = post.eval_perimeterADS3D(X_ADS_geocentric_obs, Obs_perimeter_norm, tgrid, time_idxs=None, state_dim=6) #Structure: DAIOD_ADS_perimeter['final_map'][time_idx] : (Number of perimeter points, state number, subdomain number)[time_idx]
    # print("ADS Evaluation 1 complete")
    # Method_Clock_time[0,2] = time.time() - time_ref_start

    # time_ref_start = time.time()
    # DAIOD_perimeter  = post.eval_perimeterADS3D(X_DAIOD_geocentric_obs, Obs_perimeter_norm, tgrid, time_idxs=None, state_dim=6)
    # Method_Clock_time[1,2] = time.time() - time_ref_start
    # print("ADS Evaluation 2 complete")
    
    #DA Evaluation
    time_ref_start = time.time()
    DAIOD_DA_perimeter_instance  = np.zeros((Obs_perimeter_norm.shape[0], X_DAIOD_DA_geocentric_obs.shape[0], 1))  # Structure: (Points x State x 1)[Time_idx]
    DAIOD_DA_perimeter = []

    for i in range(len(tgrid)):    #For i'th time 
        n_points = Obs_perimeter_norm.shape[0]
        
        DAIOD_DA_perimeter_instance  = np.zeros((n_points, X_DAIOD_DA_geocentric_obs.shape[0], 1))  # Structure: (Points x State x 1)[Time_idx]

        for k in range(n_points):                   # loop over each k'th point along 6D perimeter norm
            eval_point = Obs_perimeter_norm[k,:]    
            DAIOD_DA_perimeter_instance[k,:,0] = X_DAIOD_DA_geocentric_obs[:,i].eval(eval_point)

        DAIOD_DA_perimeter.append(DAIOD_DA_perimeter_instance)      # List[(Perimeter Points x States)]
    Method_Clock_time[1,2] = time.time() - time_ref_start

    print("DA Evaluation 1 complete")
    
    Method_Clock_time[2,2] = time.time() - time_ref_start  

    #DA Evaluation for DAIOD_ADS DA propagation
    time_ref_start = time.time()
    DAIOD_ADS_DA_perimeter_instance  = np.zeros((Obs_perimeter_norm.shape[0], X_DAIOD_ADS_DA_geocentric_obs.shape[0], 1))  # Structure: (Points x State x 1)[Time_idx]
    DAIOD_ADS_DA_perimeter = []

    for i in range(len(tgrid)):    #For i'th time 
        n_points = Obs_perimeter_norm.shape[0]
        
        DAIOD_ADS_DA_perimeter_instance  = np.zeros((n_points, X_DAIOD_ADS_DA_geocentric_obs.shape[0], 1))  # Structure: (Points x State x 1)[Time_idx]

        for k in range(n_points):                   # loop over each k'th point along 6D perimeter norm
            eval_point = Obs_perimeter_norm[k,:]    
            DAIOD_ADS_DA_perimeter_instance[k,:,0] = X_DAIOD_ADS_DA_geocentric_obs[:,i].eval(eval_point)

        DAIOD_ADS_DA_perimeter.append(DAIOD_ADS_DA_perimeter_instance)      # List[(Perimeter Points x States)]
    Method_Clock_time[0,2] = time.time() - time_ref_start
    

    print("DAIOD_ADS DA Evaluation complete")
    
    DAIOD_ADS_DA_evaluation_time = time.time() - time_ref_start
    print("ADS / DA Evaluation complete")

    # Monte Carlo Evaluation / Statistics

    print("Beginning Monte Carlo Evaluation")

    #MC DAIOD
    time_ref_start = time.time()
    DAIOD_MC_uncertanities = PW.compute_uncertainties_all(X_DAIOD_MC_geocentric_obs, tgrid, Earth_propagated_position)     #
    RA_DEC_Cov = DAIOD_MC_uncertanities['Cov_RADEC']                                                                          # Covariance Matricies in RA/DEC space at each time step

        # P_angle has shape (Ts, 2, 2), with order [RA, DEC]
    RA_sigma_DAIOD_MC_prop = DAIOD_MC_uncertanities['RA_sigma']                      # radians (needed for Query)
    DEC_sigma_DAIOD_MC_prop = DAIOD_MC_uncertanities['DEC_sigma']                    # radians (needed for Query)
    mean_observables_DAIOD = DAIOD_MC_uncertanities['mu_observables']                       # Mean Observables at each time step (Ts, 3) [RA, DEC, rho]
    Method_Clock_time[3,2] = time.time() - time_ref_start

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

    time_ref_start = time.time()
    GAUSS_MC_uncertanities = PW.compute_uncertainties_all(X_GAUSS_MC_geocentric_obs, tgrid, Earth_propagated_position)     #
    RA_sigma_GAUSS_MC_prop = GAUSS_MC_uncertanities['RA_sigma']                      # radians (needed for Query)
    DEC_sigma_GAUSS_MC_prop = GAUSS_MC_uncertanities['DEC_sigma']                    # radians (needed for Query)
    mean_observables_GAUSS = GAUSS_MC_uncertanities['mu_observables']  
    Method_Clock_time[4,2] = time.time() - time_ref_start
    print("Monte Carlo Evaluation Complete")
                     # Mean Observables at each time step (Ts, 3) [RA, DEC, rho]


    # ...existing code...

    # Plot for debugging - Final propagated state range/range-rate comparison
    final_time_idx = -1  # Last time step (final propagated state)
    
    # Extract final propagated states for range/range-rate
    # range_DAIODADS_ADS = []
    # rangerate_DAIODADS_ADS = []
    # for i in range(DAIOD_ADS_perimeter['final_map'][final_time_idx].shape[2]):
    #     range_DAIODADS_ADS.append(DAIOD_ADS_perimeter['final_map'][final_time_idx][:,2,i])
    #     rangerate_DAIODADS_ADS.append(DAIOD_ADS_perimeter['final_map'][final_time_idx][:,5,i])

    # range_DAIOD_ADS = []
    # rangerate_DAIOD_ADS = []
    # for i in range(DAIOD_perimeter['final_map'][final_time_idx].shape[2]):
    #     range_DAIOD_ADS.append(DAIOD_perimeter['final_map'][final_time_idx][:,2,i])
    #     rangerate_DAIOD_ADS.append(DAIOD_perimeter['final_map'][final_time_idx][:,5,i])

    range_DAIOD_DA = DAIOD_DA_perimeter[final_time_idx][:,2,0]
    rangerate_DAIOD_DA = DAIOD_DA_perimeter[final_time_idx][:,5,0]

    range_DAIOD_ADS_DA = DAIOD_ADS_DA_perimeter[final_time_idx][:,2,0]
    rangerate_DAIOD_ADS_DA = DAIOD_ADS_DA_perimeter[final_time_idx][:,5,0]


    range_PW = X_DAIOD_MC_geocentric_obs[:,2,final_time_idx]
    rangerate_PW = X_DAIOD_MC_geocentric_obs[:,5,final_time_idx]


    range_GAUSS = X_GAUSS_MC_geocentric_obs[:,2,final_time_idx]
    rangerate_GAUSS = X_GAUSS_MC_geocentric_obs[:,5,final_time_idx]


    # Get NASA Horizons truth data for Apophis at final time
    final_time = t_propagation[final_time_idx]
    final_time.format = 'iso'
    test_time= final_time + TimeDelta(1, format='jd')
    eph_final, _ = post.load_Apophis_Ephemeris(final_time, test_time, step='1')  # Geocentric
    
    # Convert NASA truth to range/range-rate
    apophis_ra_final = (eph_final['RA'][0] * u.deg).to(u.rad).value
    apophis_dec_final = (eph_final['DEC'][0] * u.deg).to(u.rad).value
    apophis_range_final = eph_final['delta'][0]  # AU, convert to km
    apophis_range_final_km = (apophis_range_final * u.AU).to(u.km).value
    apophis_range_rate_final = eph_final['delta_rate'][0]  # km/s
    
    # Create IOD_plots directory
    iod_plots_dir = script_dir / "IOD_plots"
    iod_plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12,8))
    plt.scatter(range_PW, rangerate_PW, label='DAIOD MC (Final)', alpha=0.6, s=30)
    #plt.scatter(range_DAIODADS_ADS, rangerate_DAIODADS_ADS, label='DAIOD+ADS (ADS OD, Final)', alpha=0.6, s=30)
    #plt.scatter(range_DAIOD_ADS, rangerate_DAIOD_ADS, label='DAIOD+ADS (ADS Prop, Final)', alpha=0.6, s=30)
    plt.scatter(range_DAIOD_DA, rangerate_DAIOD_DA, label='DAIOD+DA (Final)', alpha=0.6, s=30)
    plt.scatter(range_DAIOD_ADS_DA, rangerate_DAIOD_ADS_DA, label='DAIOD+ADS+DA (Final)', alpha=0.6, s=30)
    #plt.scatter(range_GAUSS, rangerate_GAUSS, label='Gauss MC (Final)', alpha=0.6, s=30)
    
    # Plot NASA Horizons truth as a red star
    plt.scatter(apophis_range_final_km, apophis_range_rate_final, 
               color='red', marker='*', s=200, 
               label='NASA Horizons Truth (Final)', 
               edgecolors='black', linewidth=1, zorder=10)
    
    plt.xlabel('Range (km)', fontsize=12)
    plt.ylabel('Range Rate (km/s)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.title(f"Final Propagated State: Range vs Range-Rate\nFinal Time: {final_time.iso}", fontsize=14)
    
    # Save plot instead of showing
    plot_filename = f"final_range_rangerate_arc_{dt[m]:.3f}.png"
    plot_path = iod_plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    print(f"Final propagated range/range-rate plot saved to: {plot_path}")

    # Print comparison statistics
    print(f"\nFinal State Comparison:")
    print(f"NASA Horizons Truth - Range: {apophis_range_final_km:.3f} km, Range-rate: {apophis_range_rate_final:.6f} km/s")
    if len(range_PW) > 0:
        mean_range_mc = np.mean(range_PW)
        mean_rangerate_mc = np.mean(rangerate_PW)
        print(f"DAIOD MC Mean - Range: {mean_range_mc:.3f} km, Range-rate: {mean_rangerate_mc:.6f} km/s")
        print(f"DAIOD MC Range Error: {abs(mean_range_mc - apophis_range_final_km):.3f} km")
        print(f"DAIOD MC Range-rate Error: {abs(mean_rangerate_mc - apophis_range_rate_final):.6f} km/s")

    # Plot for debugging - Initial state range/range-rate comparison
    initial_time_idx = 0  # First time step (initial state)
    
    # Extract initial states for range/range-rate
    # range_DAIODADS_ADS_init = []
    # rangerate_DAIODADS_ADS_init = []
    # for i in range(DAIOD_ADS_perimeter['final_map'][initial_time_idx].shape[2]):
    #     range_DAIODADS_ADS_init.append(DAIOD_ADS_perimeter['final_map'][initial_time_idx][:,2,i])
    #     rangerate_DAIODADS_ADS_init.append(DAIOD_ADS_perimeter['final_map'][initial_time_idx][:,5,i])

    # range_DAIOD_ADS_init = []
    # rangerate_DAIOD_ADS_init = []
    # for i in range(DAIOD_perimeter['final_map'][initial_time_idx].shape[2]):
    #     range_DAIOD_ADS_init.append(DAIOD_perimeter['final_map'][initial_time_idx][:,2,i])
    #     rangerate_DAIOD_ADS_init.append(DAIOD_perimeter['final_map'][initial_time_idx][:,5,i])

    
    range_DAIOD_DA_init = DAIOD_DA_perimeter[initial_time_idx][:,2,0]
    rangerate_DAIOD_DA_init = DAIOD_DA_perimeter[initial_time_idx][:,5,0]

    range_DAIOD_ADS_DA_init = DAIOD_ADS_DA_perimeter[initial_time_idx][:,2,0]
    rangerate_DAIOD_ADS_DA_init = DAIOD_ADS_DA_perimeter[initial_time_idx][:,5,0]


    range_PW_init = X_DAIOD_MC_geocentric_obs[:,2,initial_time_idx]
    rangerate_PW_init = X_DAIOD_MC_geocentric_obs[:,5,initial_time_idx]

    range_GAUSS_init = X_GAUSS_MC_geocentric_obs[:,2,initial_time_idx]
    rangerate_GAUSS_init = X_GAUSS_MC_geocentric_obs[:,5,initial_time_idx]


    # Get NASA Horizons truth data for Apophis at initial time
    initial_time = t_propagation[initial_time_idx]
    initial_time.format = 'iso'
    test_time_init = initial_time + TimeDelta(1, format='jd')
    eph_initial, _ = post.load_Apophis_Ephemeris(initial_time, test_time_init, step='1')  # Geocentric
    
    # Convert NASA truth to range/range-rate for initial time
    apophis_ra_initial = (eph_initial['RA'][0] * u.deg).to(u.rad).value
    apophis_dec_initial = (eph_initial['DEC'][0] * u.deg).to(u.rad).value
    apophis_range_initial = eph_initial['delta'][0]  # AU, convert to km
    apophis_range_initial_km = (apophis_range_initial * u.AU).to(u.km).value
    apophis_range_rate_initial = eph_initial['delta_rate'][0]  # km/s

    plt.figure(figsize=(12,8))
    plt.scatter(range_PW_init, rangerate_PW_init, label='DAIOD MC (Initial)', alpha=0.6, s=30)
    #plt.scatter(range_DAIODADS_ADS_init, rangerate_DAIODADS_ADS_init, label='DAIOD+ADS (ADS OD, Initial)', alpha=0.6, s=30)
    #plt.scatter(range_DAIOD_ADS_init, rangerate_DAIOD_ADS_init, label='DAIOD+ADS (ADS Prop, Initial)', alpha=0.6, s=30)
    plt.scatter(range_DAIOD_DA_init, rangerate_DAIOD_DA_init, label='DAIOD+DA (Initial)', alpha=0.6, s=30)
    plt.scatter(range_DAIOD_ADS_DA_init, rangerate_DAIOD_ADS_DA_init, label='DAIOD+ADS+DA (Initial)', alpha=0.6, s=30)
    #plt.scatter(range_GAUSS_init, rangerate_GAUSS_init, label='Gauss MC (Initial)', alpha=0.6, s=30)
    
    # Plot NASA Horizons truth as a red star
    plt.scatter(apophis_range_initial_km, apophis_range_rate_initial, 
               color='red', marker='*', s=200, 
               label='NASA Horizons Truth (Initial)', 
               edgecolors='black', linewidth=1, zorder=10)
    
    plt.xlabel('Range (km)', fontsize=12)
    plt.ylabel('Range Rate (km/s)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.title(f"Initial State: Range vs Range-Rate\nInitial Time: {initial_time.iso}", fontsize=14)
    
    # Save plot instead of showing
    plot_filename_init = f"initial_range_rangerate_arc_{dt[m]:.3f}.png"
    plot_path_init = iod_plots_dir / plot_filename_init
    plt.savefig(plot_path_init, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    print(f"Initial range/range-rate plot saved to: {plot_path_init}")

    # Print comparison statistics for initial state
    print(f"\nInitial State Comparison:")
    print(f"NASA Horizons Truth - Range: {apophis_range_initial_km:.3f} km, Range-rate: {apophis_range_rate_initial:.6f} km/s")
    if len(range_PW_init) > 0 and not np.all(np.isnan(range_PW_init)):
        mean_range_mc_init = np.mean(range_PW_init)
        mean_rangerate_mc_init = np.mean(rangerate_PW_init)
        print(f"DAIOD MC Mean - Range: {mean_range_mc_init:.3f} km, Range-rate: {mean_rangerate_mc_init:.6f} km/s")
        print(f"DAIOD MC Range Error: {abs(mean_range_mc_init - apophis_range_initial_km):.3f} km")
        print(f"DAIOD MC Range-rate Error: {abs(mean_rangerate_mc_init - apophis_range_rate_initial):.6f} km/s")



    # ...existing code...
    
    #split_stat = plot.plot_split_history(final_lists_ADS, final_lists_ADS_DAIOD, tgrid, dt[m], iod_plots_dir)
    #print(f"Split history plot saved to: {split_stat}")
    ############### alpha shape creation ################

        #ADS
    #DAIOD+ADS + ADS propagation
    # time_ref = time.time()
    # alphashape_DAIODADS_ADS_list = []      # List containing the time instances
    # DAIOD_ADS_ADS_RADEC_area = []
    # RA_DEC_DAIODADS_ADS_list = []
    
    # for i in range(len(tgrid)):    #For i'th time
    #     manifold = DAIOD_ADS_perimeter['final_map'][i]
    #     points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    #     unique_points = points['unique points']  # Remove duplicate points

    #     RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    #     RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    #     alpha = optalpha.optimizealpha(RA_DEC_CCW, upper=10, lower=0.01)

    #     alphashape_DAIODADS, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
    #     RA_DEC_DAIODADS_ADS = unique_points
    #     area = alphashape_DAIODADS.area

    #     #Append Matricies to time_idx list
    #     alphashape_DAIODADS_ADS_list.append(alphashape_DAIODADS)    # time instances are the list index
    #     DAIOD_ADS_ADS_RADEC_area.append(area)
    #     RA_DEC_DAIODADS_ADS_list.append(RA_DEC_DAIODADS_ADS)

    # Method_Clock_time[0,4] = time.time() - time_ref
    
    #DAIOD + ADS propagation
    # time_ref = time.time()
    # alphashape_DAIOD_ADS_list = []
    # RA_DEC_DAIOD_ADS_list = []
    # DAIOD_ADS_RADEC_area = []
    # for i in range(len(tgrid)):    #For i'th time
    #     manifold = DAIOD_perimeter['final_map'][i]
    #     points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
    #     unique_points = points['unique points']  # Remove duplicate points

    #     RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
    #     RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
    #     alpha = optalpha.optimizealpha(RA_DEC_CCW, upper=10, lower=0.01)

    #     alphashape_DAIOD_ADS_instance, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
    #     RA_DEC_DAIOD_ADS = unique_points
    #     area = alphashape_DAIOD_ADS_instance.area

    #     RA_DEC_DAIOD_ADS_list.append(RA_DEC_DAIOD_ADS)
    #     alphashape_DAIOD_ADS_list.append(alphashape_DAIOD_ADS_instance)
    #     DAIOD_ADS_RADEC_area.append(area)

    # Method_Clock_time[1,4] = time.time() - time_ref

    #MC DAIOD
    time_ref = time.time()
    alphashape_DAIOD_MC_list = []
    RA_DEC_DAIOD_MC_list = []
    DAIOD_MC_RADEC_area = []
    for i in range(len(tgrid)):    #For i'th time
        manifold = X_DAIOD_MC_geocentric_obs[:,:,i] 
        points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
        unique_points = points['unique points']                     # Remove duplicate points


        RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
        RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
        alpha = optalpha.optimizealpha(RA_DEC_CCW, upper=12, lower=0.01)
        alphashape_DAIOD_MC_instance, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
        area = alphashape_DAIOD_MC_instance.area
        #Append Matricies to lists
        RA_DEC_DAIOD_MC_list.append(RA_DEC_CCW)
        alphashape_DAIOD_MC_list.append(alphashape_DAIOD_MC_instance)
        DAIOD_MC_RADEC_area.append(area)
        print(f"Time step {i+1}/{len(tgrid)} complete")
    Method_Clock_time[3,4] = time.time() - time_ref

    print("DAIOD MC Alphashape complete")
    #MC GAUSS 
    time_ref = time.time()
    alphashape_GAUSS_MC_list = []
    RA_DEC_GAUSS_MC_list = []
    GAUSS_MC_MC_RADEC_area = []
    for i in range(len(tgrid)):    #For i'th time
        manifold = X_GAUSS_MC_geocentric_obs[:,:,i]
        points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
        unique_points = points['unique points']

        RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
        RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
        alpha = optalpha.optimizealpha(RA_DEC_CCW, upper=12, lower=0.01)

        alphashape_GAUSS_MC_propagated, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
        area = alphashape_GAUSS_MC_propagated.area
        RA_DEC_GAUSS_MC_list.append(RA_DEC_CCW)
        alphashape_GAUSS_MC_list.append(alphashape_GAUSS_MC_propagated)
        GAUSS_MC_MC_RADEC_area.append(area)
        print(f"Time step {i+1}/{len(tgrid)} complete")

    Method_Clock_time[4,4] = time.time() - time_ref
    print("Gauss MC Alphashape complete")
    #DA only
    time_ref = time.time()
    alphashape_DAIOD_list = []      # List containing the time instances
    RA_DEC_DAIOD_DA_list =[]
    DAIOD_DA_RADEC_area = []

    for i in range(len(tgrid)):    #For i'th time
        manifold = DAIOD_DA_perimeter[i]
        points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
        unique_points = points['unique points']  # Remove duplicate points

        RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
        RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
        alpha = optalpha.optimizealpha(RA_DEC_CCW, upper=12, lower=0.01)
        alphashape_DA, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
        area = alphashape_DA.area
        RA_DEC_DAIOD_DA_list.append(RA_DEC_CCW)
        alphashape_DAIOD_list.append(alphashape_DA)
        DAIOD_DA_RADEC_area.append(area)
    Method_Clock_time[2,4] = time.time() - time_ref
    print("DAIOD DA Alphashape complete")
    #DAIOD_ADS DA only
    time_ref = time.time()
    alphashape_DAIOD_ADS_DA_list = []      # List containing the time instances
    RA_DEC_DAIOD_ADS_DA_list = []
    DAIOD_ADS_DA_RADEC_area = []

    for i in range(len(tgrid)):    #For i'th time
        manifold = DAIOD_ADS_DA_perimeter[i]
        points = post.extract_2D_points(manifold, 0, [0, 1], True) # Extract 2D points from 6D perimeter for Alphashape
        unique_points = points['unique points']  # Remove duplicate points

        RA_CCW, DEC_CCW = post.CADC_sort_CCW(unique_points[:,0], unique_points[:,1])
        RA_DEC_CCW = np.column_stack((RA_CCW, DEC_CCW))
        alpha = optalpha.optimizealpha(RA_DEC_CCW, upper=12, lower=0.01)
        alphashape_ADS_DA, _, _ = post.create_alphashape2D(RA_DEC_CCW, alpha)
        area = alphashape_ADS_DA.area
        RA_DEC_DAIOD_ADS_DA_list.append(RA_DEC_CCW)
        alphashape_DAIOD_ADS_DA_list.append(alphashape_ADS_DA)
        DAIOD_ADS_DA_RADEC_area.append(area)
    DAIOD_ADS_DA_alphashape_time = time.time() - time_ref
    
    Method_Clock_time[0,4] = time.time() - time_ref
    print("DAIOD ADS DA Alphashape complete")
    print(f"Arc {m} Complete")

    ################################################ SAVE Results with pickle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "Simulation_data"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    
    # Collect all variables needed for querying
    results_data = {
        # Time information
        'obs_time0': obs_time0,
        'obs_times': obs_times,
        'tstart': tstart,
        'tfinal': tfinal,
        'dt_propagation': dt_propagation,
        't_propagation': t_propagation,
        'tgrid': tgrid,
        'Ts': Ts,
        'dt_values': dt,
        'current_arc_index': m,
        'current_dt': dt[m],
        
        # Observer positions
        'pos_obs': pos_obs,
        'Earth_propagated_position': Earth_propagated_position,
        
        # Observation data and truth
        'ra': ra,  # Save as numpy array
        'dec': dec,
        'ra_1sigma': ra_sigma,
        'dec_1sigma': dec_sigma,
        'apophis_vec': apophis_vec,
        'apophis_pos': apophis_pos.value,
        'apophis_vel': apophis_vel.value,
        
        # Initial orbit determination results (ADS objects preserved)
        'X0_Obj_ECI_DAIOD_ADS': X0_Obj_ECI_DAIOD_ADS,
        'X0_Obj_ECI_DAIOD_DA': X0_Obj_ECI_DAIOD_DA,
        'X0_Obj_ECI_DAIOD_MC': X0_Obj_ECI_DAIOD_MC,
        'X0_Obj_ECI_GAUSS_MC': X0_Obj_ECI_GAUSS_MC,
        
        # Propagated results (ADS objects preserved)
        # 'final_lists_ADS': final_lists_ADS,
        # 'final_lists_ADS_DAIOD': final_lists_ADS_DAIOD,
        'X0_Obj_ECI_DAIOD_DA_propagated': X0_Obj_ECI_DAIOD_DA_propagated,
        'X0_Obj_ECI_DAIOD_ADS_DA_propagated': X0_Obj_ECI_DAIOD_ADS_DA_propagated,
        'X0_Obj_ECI_MC_propagated': X0_Obj_ECI_MC_propagated,
        'X0_Obj_ECI_GAUSS_MC_propagated': X0_Obj_ECI_GAUSS_MC_propagated,
        
        # Converted to observational coordinates (ADS/DA objects preserved)
        # 'X_ADS_geocentric_obs': X_ADS_geocentric_obs,
        # 'X_DAIOD_geocentric_obs': X_DAIOD_geocentric_obs,
        'X_DAIOD_DA_geocentric_obs': X_DAIOD_DA_geocentric_obs,
        'X_DAIOD_ADS_DA_geocentric_obs': X_DAIOD_ADS_DA_geocentric_obs,
        'X_DAIOD_MC_geocentric_obs': X_DAIOD_MC_geocentric_obs,
        'X_GAUSS_MC_geocentric_obs': X_GAUSS_MC_geocentric_obs,
        
        # Evaluated perimeters (ready for querying)
        # 'DAIOD_ADS_perimeter': DAIOD_ADS_perimeter,
        # 'DAIOD_perimeter': DAIOD_perimeter,
        'DAIOD_DA_perimeter': DAIOD_DA_perimeter,
        'DAIOD_ADS_DA_perimeter': DAIOD_ADS_DA_perimeter,
        'Obs_perimeter_norm': Obs_perimeter_norm,
        
        # Monte Carlo uncertainty analysis
        'DAIOD_MC_uncertanities': DAIOD_MC_uncertanities,
        'GAUSS_MC_uncertanities': GAUSS_MC_uncertanities,
        'RA_sigma_DAIOD_MC_prop': RA_sigma_DAIOD_MC_prop,
        'DEC_sigma_DAIOD_MC_prop': DEC_sigma_DAIOD_MC_prop,
        'mean_observables_DAIOD': mean_observables_DAIOD,
        'RA_sigma_GAUSS_MC_prop': RA_sigma_GAUSS_MC_prop,
        'DEC_sigma_GAUSS_MC_prop': DEC_sigma_GAUSS_MC_prop,
        'mean_observables_GAUSS': mean_observables_GAUSS,
        
        # Alphashapes and RA/DEC points (ready for database queries)
        'alphashapes': {
            # 'DAIOD_ADS_ADS': alphashape_DAIODADS_ADS_list,
            # 'DAIOD_ADS': alphashape_DAIOD_ADS_list,
            'DAIOD_DA': alphashape_DAIOD_list,
            'DAIOD_ADS_DA': alphashape_DAIOD_ADS_DA_list,
            'DAIOD_MC': alphashape_DAIOD_MC_list,
            'GAUSS_MC': alphashape_GAUSS_MC_list
        },
        'ra_dec_points': {
            # 'DAIOD_ADS_ADS': RA_DEC_DAIODADS_ADS_list,
            # 'DAIOD_ADS': RA_DEC_DAIOD_ADS_list,
            'DAIOD_DA': RA_DEC_DAIOD_DA_list,
            'DAIOD_ADS_DA': RA_DEC_DAIOD_ADS_DA_list,
            'DAIOD_MC': RA_DEC_DAIOD_MC_list,
            'GAUSS_MC': RA_DEC_GAUSS_MC_list
        },
        'alphashape_areas': {
            # 'DAIOD_ADS_ADS': DAIOD_ADS_ADS_RADEC_area,
            # 'DAIOD_ADS': DAIOD_ADS_RADEC_area,
            'DAIOD_DA': DAIOD_DA_RADEC_area,
            'DAIOD_ADS_DA': DAIOD_ADS_DA_RADEC_area,
            'DAIOD_MC': DAIOD_MC_RADEC_area,
            'GAUSS_MC': GAUSS_MC_MC_RADEC_area
        },
        
        # Processing times
        'Method_Clock_time': Method_Clock_time.copy(),
        
        # Algorithm parameters (including DA initialization info)
        'algorithm_params': {
            'mu': mu,
            'DA_axis_points': DA_axis_points,
            'MonteCarlo_samples': MonteCarlo_samples,
            'toll': toll,
            'Max_split_propagation': Max_split,
            'da_order': order,  # From your DAIOD calls
            'da_nvars': 6   # 6D state space
        }
    }
    
    # First save DA parameters separately for pre-loading
    da_params_path = output_dir / f"arc_{m:03d}_da_params_{timestamp}.pkl"
    da_params_only = {
        'da_order': order,
        'da_nvars': 6,
        'arc_index': m,
        'timestamp': timestamp
    }
    
    try:
        with open(da_params_path, 'wb') as f:
            pickle.dump(da_params_only, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ DA parameters saved to: {da_params_path}")
    except Exception as e:
        print(f"✗ Error saving DA parameters: {e}")

    # Save to main pickle file with descriptive name
    save_path = output_dir / f"arc_{m:03d}_dt_{dt[m]:.3f}d_complete_results_{timestamp}.pkl"

    try:
        with open(save_path, 'wb') as f:
            pickle.dump(results_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = save_path.stat().st_size / (1024**2)
        print(f"✓ Arc {m} results saved to: {save_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Contains {len(tgrid)} time steps with all alphashapes ready for querying")
        
    except Exception as e:
        print(f"✗ Error saving Arc {m} results: {e}")
    
    # Optional: Save a quick summary CSV for this arc
    summary_data = {
        'arc_index': m,
        'dt_days': dt[m],
        'obs_time_center': obs_time0.iso,
        'prop_start': tstart.iso,
        'prop_end': tfinal.iso,
        'n_time_steps': Ts,
        'total_processing_time_sec': Method_Clock_time.sum(),
        'oda_ads_time': Method_Clock_time[0, :].sum(),
        'oda_da_time': Method_Clock_time[1, :].sum(),
        'mc_time': Method_Clock_time[3:, :].sum(),
        'save_path': str(save_path),
        'timestamp': timestamp
    }
    timestop = time.time() - time1
    print(f"Total time for arc {m}: {timestop} seconds")
    summary_df = pd.DataFrame([summary_data])
    summary_csv_path = output_dir / f"arc_{m:03d}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Arc {m} processing complete.\n")
    
print("All arcs processed.")
    











