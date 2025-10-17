import os
import numpy as np
import pandas as pd
from daceypy import DA, array, ADS
from astropy.time import Time, TimeDelta
from astropy import units as u
import time
import matplotlib.pyplot as plt
import utils.optimisealpha as optalpha

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
mu = 3.986e5
observation_times_seconds = np.array([0, 118.10, 237.58])  # Convert to seconds
RA_rad = np.array([np.deg2rad(43.537), np.deg2rad(54.420), np.deg2rad(64.318)]) #Topocentric
DEC_rad = np.array([np.deg2rad(-8.7833), np.deg2rad(-12.074), np.deg2rad(-15.105)])
N_simulations = 100
tstart = Time("2023-01-01T00:00:00", scale='utc')
tend = Time("2023-01-01T00:02:00", scale='utc')

#Example Set Up: Curtis Gauss IOD example 
Obs_perimeter_norm = post.gen_grid6D(3*ang_sigma, 2)

pos_obs = np.array([[3489.8, 3460.1, 3429.9],   #ECI position (done in curtis)
                    [3430.2, 3460.1, 3490.1],
                    [4078.5, 4078.5, 4078.5]])

earth_rot = np.array([0, 0, 7.2921159e-5])  # rad/s
vel_obs = np.cross(earth_rot, pos_obs[:,1])  # km/s

X_ECI_observation_epoch = np.concatenate((pos_obs[:,1], vel_obs))          #Observer State in ECI coordinates [6,]
Method_Clock_time = np.zeros((5,5))                                               #Structure: ( DAIOD+ADS_ADS,  DAIOD_+ADS, DAIOD_DA, PW_MC, PW_gauss) x (OD, Prop, Eval, Conversion, Query)

################################# Orbit Determination ~##########################################################
time_ref_start = time.time()
X0_Obj_ECI_DAIOD_ADS = iod.DAIOD_ADS_full(RA_rad, DEC_rad, ang_sigma, ang_sigma, pos_obs, observation_times_seconds, mu, DA_order, prograde=True) #Run DAIOD to get Object State
time_ref_end = time.time()
Method_Clock_time[0,0] = time_ref_end - time_ref_start

time_ref_start = time.time()
X0_Obj_ECI_DAIOD_DA = iod.DAIOD_full(RA_rad, DEC_rad, ang_sigma, ang_sigma, pos_obs, observation_times_seconds, mu, DA_order, prograde=True)                                 #Run DAIOD to get Object State
time_ref_end = time.time()
Method_Clock_time[1,0] = time_ref_end - time_ref_start
Method_Clock_time[2,0] = time_ref_end - time_ref_start 

time_ref_start = time.time()
X0_Obj_ECI_DAIOD_MC, X0_Obj_ECI_GAUSS_MC = PW.monte_carlo_gauss_PWiod(N_simulations, pos_obs, RA_rad, DEC_rad, observation_times_seconds, ang_sigma, ang_sigma, mu, prograde_bool_=True)              #Run PW Gauss IOD to get Object State
time_ref_end = time.time()
Method_Clock_time[3,0] = time_ref_end - time_ref_start
Method_Clock_time[4,0] = time_ref_end - time_ref_start  #
################################ Orbit Propagation ####################################################
tstart_days = tstart.jd
tend_days = tend.jd
prop_days = tend_days - tstart_days
Ts = 2
tgrid = np.linspace(0, prop_days*24*3600, Ts)  # Propagation time in seconds


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
X_ADS_Topocentric = post.ADS_ECICart_2_Topocentric(final_lists_ADS, X_ECI_observation_epoch)
X_ADS_Topocentric_Observation = post.ADS_Cart_2_Obs(X_ADS_Topocentric)
Method_Clock_time[0,3] = time.time() - time_ref_start

# DAIOD ADS propagation
time_ref_start = time.time()
X_DAIOD_Topocentric = post.ADS_ECICart_2_Topocentric(final_lists_ADS_DAIOD, X_ECI_observation_epoch)
X_DAIOD_Topocentric_Observation = post.ADS_Cart_2_Obs(X_DAIOD_Topocentric)
Method_Clock_time[1,3] = time.time() - time_ref_start

# DAIOD DA propagation
time_ref_start = time.time()
X_DAIOD_DA_Topocentric = array.zeros((6, X0_Obj_ECI_DAIOD_DA_propagated.shape[1]))  # Structure: (States x Time)
for i in range(X0_Obj_ECI_DAIOD_DA_propagated.shape[1]):
    X_DAIOD_DA_Topocentric[:,i] = X0_Obj_ECI_DAIOD_DA_propagated[:,i] - X_ECI_observation_epoch # [6 x len(prop_time)] - [6 x 1] -> Technically this is wrong. We should call the position of the observer at propagated time step 
Method_Clock_time[2,3] = time.time() - time_ref_start

# PW Conversion - DAIOD formulation
time_ref_start = time.time()
X0_Obj_observables_MC_propagated = np.zeros_like(X0_Obj_ECI_MC_propagated)
for i in range(X0_Obj_ECI_MC_propagated.shape[2]):
    for j in range(X0_Obj_ECI_MC_propagated.shape[0]):
        X0_Obj_Topocentric_MC_propagated_Cart = X0_Obj_ECI_MC_propagated[j, :, i] - X_ECI_observation_epoch
        X0_Obj_observables_MC_propagated[j, :, i] = time_ref_func.CC2obs(X0_Obj_Topocentric_MC_propagated_Cart)
Method_Clock_time[3,3] = time.time() - time_ref_start
# PW Conversion - Gauss
time_ref_start = time.time()
X0_Obj_observables_GAUSS_MC_propagated = np.zeros_like(X0_Obj_ECI_GAUSS_MC_propagated)
for i in range(X0_Obj_ECI_GAUSS_MC_propagated.shape[2]):
    for j in range(X0_Obj_ECI_GAUSS_MC_propagated.shape[0]):
        X0_Obj_Topocentric_GAUSS_MC_propagated_Cart = X0_Obj_ECI_GAUSS_MC_propagated[j, :, i] - X_ECI_observation_epoch
        X0_Obj_observables_GAUSS_MC_propagated[j, :, i] = time_ref_func.CC2obs(X0_Obj_Topocentric_GAUSS_MC_propagated_Cart)
Method_Clock_time[4,3] = time.time() - time_ref_start   

print("Monte Carlo Propagation complete")
################################# ADS / DA Evaluation for Plotting ##########################################

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

    alpha = optalpha.optimizealpha(unique_points)
    alphashape_DAIODADS, _, _ = post.create_alphashape2D(unique_points, alpha)
    RA_DEC_DAIODADS_ADS = unique_points
    #area = alphashape_DAIODADS.area

    #Append Matricies to time_idx list
    alphashape_DAIODADS_ADS_list.append(alphashape_DAIODADS)    # time instances are the list index
    #DAIOD_ADS_ADS_RADEC_area.append(area)
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

    alpha = optalpha.optimizealpha(unique_points)
    alphashape_DAIOD_ADS_instance, _, _ = post.create_alphashape2D(unique_points, alpha)
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

    alpha = optalpha.optimizealpha(unique_points)
    alphashape_DAIOD_MC_instance, _, _ = post.create_alphashape2D(unique_points, alpha)
    RA_DEC = unique_points

    RA_DEC_DAIOD_MC_list.append(RA_DEC)
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

    alpha = optalpha.optimizealpha(unique_points)
    alphashape_GAUSS_MC_propagated, _, _ = post.create_alphashape2D(unique_points, alpha)
    RA_DEC = unique_points

    RA_DEC_GAUSS_MC_list.append(RA_DEC)
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

    alpha = optalpha.optimizealpha(unique_points)
    alphashape_DA, _, _ = post.create_alphashape2D(unique_points, alpha)
    RA_DEC = unique_points

    RA_DEC_DAIOD_DA_list.append(RA_DEC)
    alphashape_DAIOD_list.append(alphashape_DA)
Method_Clock_time[2,4] = time.time() - time_ref


################################### Apophis Query N/A ################################
"""
    This section would Query CADC for Apophis Images.
    - It would manage the retrieval and counting of Images for MC, DAIOD+ADS_DA, DAIOD+ADS_ADS, DAIOD_DA. (Fixed Area Image Retrival) -> Total Images
    - The 'true' location via Horizon Ephemeris Collection of Apophis converted to RA-DEC coordinates. -> Defines TP
    - The Total Images within SSOIS + uncertanity -> Control/Baseline Images 'truth'
    - Define FP + TP /TP Total Images / True Image (Lowest is best), Define FN 
"""    

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
            ax.plot(*alphashape_list[i].exterior.xy, 'r--', label = f'alphashape: {1}')

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
        for i in range(method_data.shape[2]):
            method_data_2d = method_data[:,:,i]
            method_data_2d = method_data_2d.reshape(method_data_2d.shape[0], 6, 1)
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
        
        manifold = method_data[i]       #Topocentric CC
        # Convert to ECI Cartesian
        eci_manifold = np.zeros_like(manifold)
        eci_CC_manifold = np.zeros_like(manifold)
        eci_CC_manifold_2d = np.zeros((manifold.shape[0], 2))  # For 2D points
        for k in range(manifold.shape[2]):  # For each subdomain
            for j in range(manifold.shape[0]): #each point
                # Spherical to Cartesian
                manifold[j,:,k] = time_ref_func.obs2CC(manifold[j,:,k])
                
                eci_tmp = topocentric_to_eci_cartesian(manifold[j,:,k], observer_eci_position)
                eci_CC_manifold[j,:,k] = eci_tmp.reshape(-1)  # Ensure shape is (6,)

                #Rotate to Angular Momentum vector
                eci_CC_manifold[j,:,k], _ = plot.state_to_orbital_frame(eci_CC_manifold[j,:,k], mu)

        points_2d = post.extract_2D_points(eci_CC_manifold, 0, [0,1], False)  # Ensure shape is (N_points, 2)
        eci_CC_manifold_2d = points_2d['unique points']  # Remove duplicate points
        label = "{:.2f}".format(tgrid[i]/(60/60)) + " hrs"
        ax.plot(eci_CC_manifold_2d[:, 0], eci_CC_manifold_2d[:, 1], label=label)
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

