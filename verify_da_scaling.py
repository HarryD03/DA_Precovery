#!/usr/bin/env python3
"""
Verify DA polynomial scaling issues based on user's analysis.
"""

import numpy as np

def verify_scaling_analysis():
    """Verify the user's scaling analysis"""
    
    print("="*70)
    print("VERIFYING DA POLYNOMIAL SCALING ISSUE")
    print("="*70)
    
    # User's observed values
    corner_eval = np.array([-159798272.20767927, 4696785.045161634, -2354769.628354633, 
                           4.990834839906142, -65.05404458428407, -24.103880592899777])
    
    nominal_eval = np.array([-1.59895558e+08, 4.96212453e+06, -2.26135539e+06, 
                            2.18052652e+00, -2.46849696e+01, -9.12761746e+00])
    
    residual = corner_eval - nominal_eval
    
    print("User's Observed Values:")
    print(f"  Corner evaluation: {corner_eval}")
    print(f"  Nominal evaluation: {nominal_eval}")
    print(f"  Residual: {residual}")
    
    # Position analysis
    pos_residual = residual[:3]
    pos_residual_mag = np.linalg.norm(pos_residual)
    nominal_range = np.linalg.norm(nominal_eval[:3])
    
    print(f"\nPosition Analysis:")
    print(f"  Position residual magnitude: {pos_residual_mag:.2e} km")
    print(f"  Nominal range: {nominal_range:.2e} km")
    print(f"  Relative position error: {pos_residual_mag/nominal_range:.2e}")
    
    # Expected vs actual perturbation
    angular_sigma = 3e-5  # 3σ in radians (from user's 1e-5 * 3)
    expected_linear_perturbation = nominal_range * angular_sigma
    
    print(f"\nExpected vs Actual Perturbation:")
    print(f"  Angular uncertainty (3σ): {angular_sigma:.2e} rad = {angular_sigma*180/np.pi*3600:.2f} arcsec")
    print(f"  Expected linear perturbation: {expected_linear_perturbation:.2e} km")
    print(f"  Actual DA perturbation: {pos_residual_mag:.2e} km")
    print(f"  Ratio (actual/expected): {pos_residual_mag/expected_linear_perturbation:.1f}")
    
    # User's trigonometry check
    user_distance = 800e6  # km
    user_angular_dev = 3e-5  # rad
    user_expected = user_distance * user_angular_dev
    
    print(f"\nUser's Trigonometry Check:")
    print(f"  Distance: {user_distance:.0e} km")
    print(f"  Angular deviation: {user_angular_dev:.0e} rad")
    print(f"  Expected perturbation: {user_expected:.0e} km = {user_expected/1000:.0f} km")
    print(f"  User's calculation matches our analysis: {abs(user_expected - expected_linear_perturbation) < 1000}")
    
    # Velocity analysis
    vel_residual = residual[3:]
    vel_residual_mag = np.linalg.norm(vel_residual)
    nominal_vel_mag = np.linalg.norm(nominal_eval[3:])
    
    print(f"\nVelocity Analysis:")
    print(f"  Velocity residual magnitude: {vel_residual_mag:.2e} km/s")
    print(f"  Nominal velocity magnitude: {nominal_vel_mag:.2e} km/s")
    print(f"  Relative velocity error: {vel_residual_mag/nominal_vel_mag:.2e}")
    
    # DA scaling diagnosis
    print(f"\n" + "="*50)
    print("DIAGNOSIS: DA SCALING ISSUE")
    print("="*50)
    
    if pos_residual_mag > 10 * expected_linear_perturbation:
        print("✗ CONFIRMED: DA polynomials have excessive scaling")
        print(f"  - DA perturbations are {pos_residual_mag/expected_linear_perturbation:.0f}× larger than expected")
        print(f"  - This explains the unrealistic range spans in correlation analysis")
        print(f"  - The box-like structure is obscured by over-scaled perturbations")
    else:
        print("✓ DA scaling appears reasonable")
    
    # Root cause analysis
    print(f"\nRoot Cause Analysis:")
    print(f"  1. DA domain [-1,+1] should map to ±3σ physical perturbations")
    print(f"  2. But actual DA evaluation gives {pos_residual_mag/expected_linear_perturbation:.0f}× larger perturbations")
    print(f"  3. Possible causes:")
    print(f"     - Double scaling in DAIOD_full (3σ applied twice)")
    print(f"     - Incorrect units in DA coefficient generation")
    print(f"     - Linearization assumptions breaking down at 6th order")
    print(f"     - Numerical conditioning issues in DA construction")
    
    # Recommendations
    print(f"\nRecommendations:")
    print(f"  1. Check DAIOD_full for double application of 3σ scaling")
    print(f"  2. Reduce DA order (try order=2 or 3) to improve conditioning")
    print(f"  3. Use smaller uncertainty values for DA construction")
    print(f"  4. Scale DA evaluation domain to match expected perturbations")

if __name__ == "__main__":
    verify_scaling_analysis()
