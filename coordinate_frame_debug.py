"""
Detailed coordinate frame diagnostic to compare with main_final.py exactly
"""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import utils.post_process as post
import astropy.coordinates as acoords
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy import constants as ac

def compare_coordinate_frames():
    """Compare coordinate frame handling between diagnostic and main script"""
    
    print("=== COORDINATE FRAME COMPARISON ===\n")
    
    # Exact same parameters as main_final.py
    dt = np.array([1.0])
    dt_astropy = TimeDelta(dt, format='jd')
    obs_time0 = Time('2005-06-17 23:41:07', scale='utc')
    
    obs_times = Time([obs_time0 - dt_astropy[0], obs_time0, obs_time0 + dt_astropy[0]])
    
    print("=== OBSERVER POSITION CALCULATION ===")
    print("Using EXACT same code as main_final.py...\n")
    
    # EXACT same code as main_final.py lines 97-110
    mu = ac.G * ac.M_sun
    mu = mu.to('km**3 / s**2').value
    epochs = Time(obs_times, scale='utc')
    acoords.solar_system_ephemeris.set("builtin")
    
    # Get ICRS Equatorial position of the Earth at each observation epoch
    pv_earth = [acoords.get_body_barycentric_posvel('earth', epoch) for epoch in epochs]
    # Translate to Heliocentric Equatorial Frame
    pv_sun = [acoords.get_body_barycentric_posvel('sun', epoch) for epoch in epochs] 
    pv_earth_helio = [(pv_earth[i][0] - pv_sun[i][0], pv_earth[i][1] - pv_sun[i][1]) for i in range(len(pv_earth))]
    
    # EXACT same indexing as main_final.py lines 109-110
    pos_obs = np.array([pv_earth_helio[i][:3] for i in range(len(pv_earth_helio))]).T  # km [3xN]
    
    print("Observer positions (heliocentric ICRS, km):")
    for i in range(3):
        print(f"  obs_{i+1}: [{pos_obs[0,i]:15.6f}, {pos_obs[1,i]:15.6f}, {pos_obs[2,i]:15.6f}]")
    
    print("\n=== NASA HORIZONS DATA LOADING ===")
    print("Using EXACT same code as main_final.py...\n")
    
    # EXACT same code as main_final.py lines 113-124
    eph, _ = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1')  #ICRS Geocentric
    _, vec = post.load_Apophis_Ephemeris(obs_times[0], obs_times[1], step='1', location='500@10') #ICRS Heliocentric

    _, vec2 = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1', location='500@10')
    eph2, _  = post.load_Apophis_Ephemeris(obs_times[1], obs_times[2], step='1')

    ra = (np.concatenate([eph['RA'], eph2['RA'][1:]]) * u.deg).to(u.rad)
    dec = (np.concatenate([eph['DEC'], eph2['DEC'][1:]]) * u.deg).to(u.rad)
    apophis_pos = ((np.concatenate([vec['x'], vec2['x'][1:]]) , np.concatenate([vec['y'], vec2['y'][1:]]), np.concatenate([vec['z'], vec2['z'][1:]])) * u.AU).to(u.km)
    apophis_vel = ((np.concatenate([vec['vx'], vec2['vx'][1:]]), np.concatenate([vec['vy'], vec2['vy'][1:]]), np.concatenate([vec['vz'], vec2['vz'][1:]])) * (u.AU/u.d)).to(u.km/u.s)
    apophis_vec = np.concatenate([apophis_pos.value, apophis_vel.value], axis=0) #km
    
    print("NASA Horizons queries:")
    print(f"  eph query:  ICRS Geocentric (for RA/DEC observations)")
    print(f"  vec query:  ICRS Heliocentric (location='500@10')")
    
    print(f"\nRA/DEC observations (from geocentric eph):")
    print(f"  RA (rad):  [{ra.value[0]:15.8f}, {ra.value[1]:15.8f}, {ra.value[2]:15.8f}]")
    print(f"  DEC (rad): [{dec.value[0]:15.8f}, {dec.value[1]:15.8f}, {dec.value[2]:15.8f}]")
    
    print(f"\nApophis truth state (from heliocentric vec):")
    truth_central = apophis_vec[:, 1]  # Central epoch
    print(f"  Position: [{truth_central[0]:15.6f}, {truth_central[1]:15.6f}, {truth_central[2]:15.6f}] km")
    print(f"  Velocity: [{truth_central[3]:15.8f}, {truth_central[4]:15.8f}, {truth_central[5]:15.8f}] km/s")
    
    print("\n=== COORDINATE FRAME ANALYSIS ===")
    print("Key insight: We are mixing coordinate frames!")
    print("1. Observer positions: HELIOCENTRIC ICRS")
    print("2. RA/DEC observations: GEOCENTRIC ICRS (correct for Earth-based obs)")
    print("3. Truth comparison: HELIOCENTRIC ICRS")
    print("4. IOD algorithm expects: Observer + RA/DEC in SAME frame")
    
    print("\n=== POTENTIAL ISSUE ===")
    print("The IOD algorithm is using:")
    print("  - Heliocentric observer positions")
    print("  - Geocentric RA/DEC observations")
    print("This mismatch could cause the 47M km error!")
    
    print("\n=== CHECKING GEOCENTRIC VS HELIOCENTRIC ===")
    
    # Let's check what the geocentric observer positions would be
    print("If we used GEOCENTRIC observer positions (should be ~0):")
    geocentric_pos = np.zeros((3, 3))  # Earth center in geocentric frame
    print(f"  obs_1: [{geocentric_pos[0,0]:15.6f}, {geocentric_pos[1,0]:15.6f}, {geocentric_pos[2,0]:15.6f}]")
    print(f"  obs_2: [{geocentric_pos[0,1]:15.6f}, {geocentric_pos[1,1]:15.6f}, {geocentric_pos[2,1]:15.6f}]")
    print(f"  obs_3: [{geocentric_pos[0,2]:15.6f}, {geocentric_pos[1,2]:15.6f}, {geocentric_pos[2,2]:15.6f}]")
    
    # Check if we can get Apophis geocentric position for comparison
    print(f"\nLet's also check Apophis GEOCENTRIC truth for comparison...")
    
    # Use the geocentric ephemeris data
    apophis_pos_geo = np.array([eph['delta'][0], eph2['delta'][1]])  # Distance in AU
    apophis_ra_geo = np.array([eph['RA'][0], eph2['RA'][1]]) * np.pi/180  # RA in radians
    apophis_dec_geo = np.array([eph['DEC'][0], eph2['DEC'][1]]) * np.pi/180  # DEC in radians
    
    # Convert geocentric spherical to Cartesian (central epoch)
    r_geo = apophis_pos_geo[1] * 149597870.7  # AU to km
    x_geo = r_geo * np.cos(apophis_dec_geo[1]) * np.cos(apophis_ra_geo[1])
    y_geo = r_geo * np.cos(apophis_dec_geo[1]) * np.sin(apophis_ra_geo[1])
    z_geo = r_geo * np.sin(apophis_dec_geo[1])
    
    print(f"Apophis GEOCENTRIC position (central epoch):")
    print(f"  Position: [{x_geo:15.6f}, {y_geo:15.6f}, {z_geo:15.6f}] km")
    print(f"  Range: {r_geo:15.6f} km")
    
    # The difference between heliocentric and geocentric should be ~Earth-Sun distance
    earth_sun_dist = np.linalg.norm(pos_obs[:, 1])
    print(f"\nEarth-Sun distance: {earth_sun_dist:15.6f} km")
    print(f"Expected difference between helio and geo positions: ~{earth_sun_dist/1e6:.1f} million km")
    
    helio_range = np.linalg.norm(truth_central[:3])
    print(f"Apophis heliocentric range: {helio_range:15.6f} km")
    print(f"Apophis geocentric range: {r_geo:15.6f} km")
    print(f"Difference: {abs(helio_range - r_geo):15.6f} km")

if __name__ == "__main__":
    compare_coordinate_frames()