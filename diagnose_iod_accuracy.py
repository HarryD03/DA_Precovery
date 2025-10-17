"""
Diagnostic script to compare DA and Monte Carlo IOD solutions against NASA truth.
This will help identify systematic errors affecting both methods.
"""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import utils.iod as iod
import utils.Classical_IOD as PW
import utils.post_process as post
import astropy.coordinates as acoords
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy import constants as ac
from daceypy import DA, array
import utils.time_reference as time_ref

def diagnose_iod_accuracy():
    """Diagnose IOD accuracy issues"""
    
    print("=== DIAGNOSING IOD ACCURACY ISSUES ===\n")
    
    # Same parameters as main_final.py
    dt = np.array([1.0])  # 1 day arc
    dt_astropy = TimeDelta(dt, format='jd')
    obs_time0 = Time('2004-03-17 23:41:07', scale='utc')
    MonteCarlo_samples = 1000
    order = 6
    
    # Observation times
    obs_times = Time([obs_time0 - dt_astropy[0], obs_time0, obs_time0 + dt_astropy[0]])
    print(f"Observation times:")
    for i, t in enumerate(obs_times):
        print(f"  obs_{i+1}: {t.iso}")
    
    # Observer positions
    mu = (ac.G * ac.M_sun).to('km**3 / s**2').value
    epochs = Time(obs_times, scale='utc')
    acoords.solar_system_ephemeris.set("builtin")
    
    pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]
    pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs]
    pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]
    
    pos_obs = np.array([[pv_earth_helio[i][0].x.to(u.km).value, pv_earth_helio[i][0].y.to(u.km).value, pv_earth_helio[i][0].z.to(u.km).value] for i in range(len(pv_earth_helio))]).T
    
    print(f"\nObserver positions (heliocentric, km):")
    for i in range(3):
        print(f"  obs_{i+1}: [{pos_obs[0,i]:12.3f}, {pos_obs[1,i]:12.3f}, {pos_obs[2,i]:12.3f}]")
    
    # Load NASA Horizons truth data
    eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')
    _, vec = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1', location='500@10')
    _, vec2 = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1', location='500@10')
    eph2, _ = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')
    
    ra = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad)
    dec = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad)
    apophis_pos = ((np.concatenate([vec['x'], vec2['x'][1:]]), np.concatenate([vec['y'], vec2['y'][1:]]), np.concatenate([vec['z'], vec2['z'][1:]])) * u.AU).to(u.km)
    apophis_vel = ((np.concatenate([vec['vx'], vec2['vx'][1:]]), np.concatenate([vec['vy'], vec2['vy'][1:]]), np.concatenate([vec['vz'], vec2['vz'][1:]])) * (u.AU/u.d)).to(u.km/u.s)
    apophis_vec_truth = np.concatenate([apophis_pos.value, apophis_vel.value], axis=0)
    
    print(f"\nNASA Horizons Truth (heliocentric, central epoch):")
    truth_central = apophis_vec_truth[:, 1]  # Central observation
    print(f"  Position: [{truth_central[0]:12.3f}, {truth_central[1]:12.3f}, {truth_central[2]:12.3f}] km")
    print(f"  Velocity: [{truth_central[3]:12.6f}, {truth_central[4]:12.6f}, {truth_central[5]:12.6f}] km/s")
    
    # Observation uncertainties
    ra_sigma = (30*eph['RA_3sigma'][0] * u.arcsec).to(u.rad).value/3 * u.rad
    dec_sigma = (30*eph['DEC_3sigma'][0] * u.arcsec).to(u.rad).value/3 * u.rad
    
    print(f"\nObservation data:")
    print(f"  RA (rad):  [{ra.value[0]:12.8f}, {ra.value[1]:12.8f}, {ra.value[2]:12.8f}]")
    print(f"  DEC (rad): [{dec.value[0]:12.8f}, {dec.value[1]:12.8f}, {dec.value[2]:12.8f}]")
    print(f"  RA_sigma:  {ra_sigma.value:12.2e} rad = {ra_sigma.to(u.arcsec).value:.3f} arcsec")
    print(f"  DEC_sigma: {dec_sigma.value:12.2e} rad = {dec_sigma.to(u.arcsec).value:.3f} arcsec")
    
    print(f"\n" + "="*80)
    print("TESTING GAUSS SEED (Nominal Solution)")
    print("="*80)
    
    # Test the fundamental Gauss IOD seed that both methods use
    i_rho = time_ref.create_da_los_vectors(ra.value, dec.value)
    positions, ranges, range_mags, v_2 = iod.Guass_8th_seed(pos_obs, i_rho, (obs_times.mjd * u.day).to(u.s).value, mu)
    
    if np.isnan(positions).any() or np.isnan(ranges).any() or np.isnan(range_mags).any() or np.isnan(v_2).any():
        print("ERROR: Gauss seed failed!")
        return
    
    gauss_solution = np.concatenate([positions[:, 1], v_2])  # Central epoch solution
    print(f"Gauss seed solution (heliocentric, central epoch):")
    print(f"  Position: [{gauss_solution[0]:12.3f}, {gauss_solution[1]:12.3f}, {gauss_solution[2]:12.3f}] km")
    print(f"  Velocity: [{gauss_solution[3]:12.6f}, {gauss_solution[4]:12.6f}, {gauss_solution[5]:12.6f}] km/s")
    
    # Compare with truth
    pos_error = np.linalg.norm(gauss_solution[:3] - truth_central[:3])
    vel_error = np.linalg.norm(gauss_solution[3:] - truth_central[3:])
    print(f"\nGauss seed errors vs NASA truth:")
    print(f"  Position error: {pos_error:12.3f} km")
    print(f"  Velocity error: {vel_error:12.6f} km/s")
    
    print(f"\n" + "="*80)
    print("TESTING DA IOD (Nominal/Constant Term)")
    print("="*80)
    
    # Test DA IOD and extract nominal solution
    DA.init(order, 6)
    try:
        X0_DA = iod.DAIOD_full(ra.value, dec.value, ra_sigma.value, dec_sigma.value, pos_obs, (obs_times.mjd * u.day).to(u.s).value, mu, order, prograde=True)
        
        if X0_DA is not None:
            # Extract constant term (nominal solution)
            da_nominal = np.array([X0_DA[i].cons() for i in range(6)])
            print(f"DA nominal solution (heliocentric, central epoch):")
            print(f"  Position: [{da_nominal[0]:12.3f}, {da_nominal[1]:12.3f}, {da_nominal[2]:12.3f}] km")
            print(f"  Velocity: [{da_nominal[3]:12.6f}, {da_nominal[4]:12.6f}, {da_nominal[5]:12.6f}] km/s")
            
            # Compare with truth
            da_pos_error = np.linalg.norm(da_nominal[:3] - truth_central[:3])
            da_vel_error = np.linalg.norm(da_nominal[3:] - truth_central[3:])
            print(f"\nDA nominal errors vs NASA truth:")
            print(f"  Position error: {da_pos_error:12.3f} km")
            print(f"  Velocity error: {da_vel_error:12.6f} km/s")
            
            # Compare DA nominal with Gauss seed
            da_vs_gauss_pos = np.linalg.norm(da_nominal[:3] - gauss_solution[:3])
            da_vs_gauss_vel = np.linalg.norm(da_nominal[3:] - gauss_solution[3:])
            print(f"\nDA nominal vs Gauss seed:")
            print(f"  Position difference: {da_vs_gauss_pos:12.3f} km")
            print(f"  Velocity difference: {da_vs_gauss_vel:12.6f} km/s")
            
        else:
            print("ERROR: DA IOD failed!")
            
    except Exception as e:
        print(f"ERROR: DA IOD exception: {e}")
    
    print(f"\n" + "="*80)
    print("TESTING MONTE CARLO IOD (Mean Solution)")
    print("="*80)
    
    # Test Monte Carlo IOD
    try:
        X0_MC_DAIOD, X0_MC_GAUSS = PW.monte_carlo_gauss_PWiod(MonteCarlo_samples, pos_obs, ra.value, dec.value, (obs_times.mjd * u.day).to(u.s).value, ra_sigma.value, dec_sigma.value, mu, prograde_bool_=True)
        
        if X0_MC_DAIOD is not None and len(X0_MC_DAIOD) > 0:
            # Calculate mean solution
            mc_mean = np.mean(X0_MC_DAIOD, axis=0)
            print(f"Monte Carlo mean solution (heliocentric, central epoch):")
            print(f"  Position: [{mc_mean[0]:12.3f}, {mc_mean[1]:12.3f}, {mc_mean[2]:12.3f}] km")
            print(f"  Velocity: [{mc_mean[3]:12.6f}, {mc_mean[4]:12.6f}, {mc_mean[5]:12.6f}] km/s")
            
            # Compare with truth
            mc_pos_error = np.linalg.norm(mc_mean[:3] - truth_central[:3])
            mc_vel_error = np.linalg.norm(mc_mean[3:] - truth_central[3:])
            print(f"\nMonte Carlo mean errors vs NASA truth:")
            print(f"  Position error: {mc_pos_error:12.3f} km")
            print(f"  Velocity error: {mc_vel_error:12.6f} km/s")
            
            # Statistics
            mc_std = np.std(X0_MC_DAIOD, axis=0)
            print(f"\nMonte Carlo solution statistics:")
            print(f"  Position std: [{mc_std[0]:12.3f}, {mc_std[1]:12.3f}, {mc_std[2]:12.3f}] km")
            print(f"  Velocity std: [{mc_std[3]:12.6f}, {mc_std[4]:12.6f}, {mc_std[5]:12.6f}] km/s")
            print(f"  Number of successful samples: {len(X0_MC_DAIOD)}")
            
        else:
            print("ERROR: Monte Carlo IOD failed!")
            
    except Exception as e:
        print(f"ERROR: Monte Carlo IOD exception: {e}")
    
    print(f"\n" + "="*80)
    print("SUMMARY OF SYSTEMATIC ERRORS")
    print("="*80)
    
    if 'gauss_solution' in locals():
        print(f"All methods should be close to the Gauss seed solution if implemented correctly.")
        print(f"Large differences from NASA truth in the Gauss seed indicate:")
        print(f"  1. Observational data issues (wrong RA/DEC)")
        print(f"  2. Observer position errors")
        print(f"  3. Coordinate frame mismatches")
        print(f"  4. Time system inconsistencies")
        print(f"  5. Gravitational parameter differences")
    
    print(f"\nGravitational parameter used: μ = {mu:.6e} km³/s²")
    
    
if __name__ == "__main__":
    diagnose_iod_accuracy()