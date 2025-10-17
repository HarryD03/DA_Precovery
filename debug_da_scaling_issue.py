"""
Debug DA scaling issue in DAIOD algorithm.
Verify that DA polynomial construction is correctly scaled.
"""

import numpy as np
from daceypy import DA, array
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import utils.iod as iod
import utils.time_reference as time_ref

def test_da_scaling_construction():
    """
    Test if DA polynomial construction correctly incorporates 3σ scaling.
    """
    print("="*70)
    print("DA SCALING CONSTRUCTION TEST")
    print("="*70)
    
    # Test parameters (same as in DAIOD_full)
    RA_rad = np.array([0.759857367, 0.949802813, 1.12254676])
    DEC_rad = np.array([-0.15329974, -0.210726107, -0.263633267])
    RA_sigma = 1e-5  # ~2 arcsec
    DEC_sigma = 1e-5  # ~2 arcsec
    
    print(f"Nominal values:")
    print(f"  RA: {RA_rad}")
    print(f"  DEC: {DEC_rad}")
    print(f"  RA_sigma: {RA_sigma:.2e} rad ({RA_sigma * 3600 * 180/np.pi:.2f} arcsec)")
    print(f"  DEC_sigma: {DEC_sigma:.2e} rad ({DEC_sigma * 3600 * 180/np.pi:.2f} arcsec)")
    
    # Initialize DA with 6 variables (3 for RA, 3 for DEC)
    DA.init(6, 6)
    
    # Construct DA variables exactly as in DAIOD_full
    RA_DA = array([RA_rad[i] + 3*RA_sigma*DA(i + 1) for i in range(3)])
    DEC_DA = array([DEC_rad[i] + 3*DEC_sigma*DA(i + 4) for i in range(3)])
    
    print(f"\nDA Construction:")
    print(f"  RA_DA[0] = {RA_rad[0]} + 3*{RA_sigma}*DA(1)")
    print(f"  RA_DA[1] = {RA_rad[1]} + 3*{RA_sigma}*DA(2)")  
    print(f"  RA_DA[2] = {RA_rad[2]} + 3*{RA_sigma}*DA(3)")
    print(f"  DEC_DA[0] = {DEC_rad[0]} + 3*{DEC_sigma}*DA(4)")
    print(f"  DEC_DA[1] = {DEC_rad[1]} + 3*{DEC_sigma}*DA(5)")
    print(f"  DEC_DA[2] = {DEC_rad[2]} + 3*{DEC_sigma}*DA(6)")
    
    # Test evaluation at domain boundaries
    print(f"\nEvaluating at DA domain boundaries:")
    
    # Nominal evaluation (all DA vars = 0)
    nominal_eval = np.zeros(6)
    RA_nominal = [RA_DA[i].eval(nominal_eval) for i in range(3)]
    DEC_nominal = [DEC_DA[i].eval(nominal_eval) for i in range(3)]
    
    print(f"  Nominal (DA=0): RA={RA_nominal}, DEC={DEC_nominal}")
    
    # Maximum positive evaluation (all DA vars = +1)
    max_pos_eval = np.ones(6)
    RA_max_pos = [RA_DA[i].eval(max_pos_eval) for i in range(3)]
    DEC_max_pos = [DEC_DA[i].eval(max_pos_eval) for i in range(3)]
    
    print(f"  Max positive (DA=+1): RA={RA_max_pos}, DEC={DEC_max_pos}")
    
    # Maximum negative evaluation (all DA vars = -1) 
    max_neg_eval = -np.ones(6)
    RA_max_neg = [RA_DA[i].eval(max_neg_eval) for i in range(3)]
    DEC_max_neg = [DEC_DA[i].eval(max_neg_eval) for i in range(3)]
    
    print(f"  Max negative (DA=-1): RA={RA_max_neg}, DEC={DEC_max_neg}")
    
    # Calculate perturbations
    print(f"\nCalculated perturbations:")
    RA_pert_pos = np.array(RA_max_pos) - np.array(RA_nominal)
    RA_pert_neg = np.array(RA_nominal) - np.array(RA_max_neg)
    DEC_pert_pos = np.array(DEC_max_pos) - np.array(DEC_nominal)
    DEC_pert_neg = np.array(DEC_nominal) - np.array(DEC_max_neg)
    
    print(f"  RA perturbations (+): {RA_pert_pos}")
    print(f"  RA perturbations (-): {RA_pert_neg}")
    print(f"  DEC perturbations (+): {DEC_pert_pos}")
    print(f"  DEC perturbations (-): {DEC_pert_neg}")
    
    # Expected perturbations
    expected_RA_pert = 3 * RA_sigma
    expected_DEC_pert = 3 * DEC_sigma
    
    print(f"\nExpected 3σ perturbations:")
    print(f"  RA: {expected_RA_pert:.2e}")
    print(f"  DEC: {expected_DEC_pert:.2e}")
    
    # Verify scaling is correct
    print(f"\nScaling verification:")
    RA_scaling_correct = np.allclose(RA_pert_pos, expected_RA_pert) and np.allclose(RA_pert_neg, expected_RA_pert)
    DEC_scaling_correct = np.allclose(DEC_pert_pos, expected_DEC_pert) and np.allclose(DEC_pert_neg, expected_DEC_pert)
    
    print(f"  RA scaling correct: {RA_scaling_correct}")
    print(f"  DEC scaling correct: {DEC_scaling_correct}")
    
    if RA_scaling_correct and DEC_scaling_correct:
        print(f"\n✓ DA SCALING IS CORRECT")
        print(f"  The DA polynomials are properly scaled for 3σ perturbations")
        print(f"  Problem likely lies elsewhere in the DAIOD algorithm")
    else:
        print(f"\n✗ DA SCALING IS INCORRECT")
        print(f"  The DA polynomial construction has scaling issues")
    
    return RA_DA, DEC_DA

def test_line_of_sight_vectors():
    """
    Test how DA scaling affects line-of-sight vector calculations.
    This is where angular perturbations get converted to positional perturbations.
    """
    print(f"\n" + "="*70)
    print("LINE-OF-SIGHT VECTOR SCALING TEST")
    print("="*70)
    
    RA_DA, DEC_DA = test_da_scaling_construction()
    
    # Create line-of-sight vectors with DA
    i_rho_DA = array(time_ref.create_da_los_vectors(RA_DA, DEC_DA))
    
    print(f"\nLine-of-sight vector construction:")
    print(f"  i_rho_DA shape: {i_rho_DA.shape}")
    
    # Test evaluations at different DA values
    eval_points = [
        np.zeros(6),      # Nominal
        np.ones(6),       # +3σ
        -np.ones(6),      # -3σ
        np.array([1, 0, 0, 0, 0, 0]),  # Only RA[0] perturbed
        np.array([0, 0, 0, 1, 0, 0]),  # Only DEC[0] perturbed
    ]
    
    eval_names = ["Nominal", "+3σ all", "-3σ all", "RA[0] only", "DEC[0] only"]
    
    for i, (eval_point, name) in enumerate(zip(eval_points, eval_names)):
        i_rho_eval = np.array([[i_rho_DA[j,k].eval(eval_point) for k in range(3)] for j in range(3)])
        
        print(f"\n  {name} (DA={eval_point}):")
        print(f"    i_rho = \n{i_rho_eval}")
        
        # Check unit vector property
        norms = [np.linalg.norm(i_rho_eval[:, k]) for k in range(3)]
        print(f"    Vector norms: {norms} (should be ~1.0)")
        
        if i == 0:  # Store nominal for comparison
            i_rho_nominal = i_rho_eval.copy()
        else:
            # Calculate angular differences
            for k in range(3):
                dot_product = np.dot(i_rho_nominal[:, k], i_rho_eval[:, k])
                angle_diff = np.arccos(np.clip(dot_product, -1, 1))
                print(f"    Observation {k+1} angular change: {angle_diff:.2e} rad ({angle_diff * 3600 * 180/np.pi:.2f} arcsec)")

def test_range_impact():
    """
    Test how angular DA scaling impacts range calculations.
    This shows where the scaling amplification might occur.
    """
    print(f"\n" + "="*70)
    print("RANGE IMPACT SCALING TEST")
    print("="*70)
    
    # Typical range to an asteroid (AU scale)
    typical_range = 1.5e8  # 1 AU in km
    print(f"Typical asteroid range: {typical_range:.2e} km")
    
    # Angular uncertainties
    RA_sigma = 1e-5  # rad
    DEC_sigma = 1e-5  # rad
    
    # Expected positional uncertainty from angular uncertainty
    # For small angles: Δr ≈ range × Δθ
    expected_pos_uncertainty = typical_range * RA_sigma
    print(f"Expected positional uncertainty (linear): {expected_pos_uncertainty:.2e} km")
    print(f"Expected 3σ uncertainty: {3 * expected_pos_uncertainty:.2e} km")
    
    # This is the scale we should see in DA evaluations
    print(f"\nExpected DA evaluation scale:")
    print(f"  When DA variables = ±1, position changes should be ~{3 * expected_pos_uncertainty:.0f} km")
    print(f"  This matches your trigonometric analysis showing ~24,000 km expected")

if __name__ == "__main__":
    test_da_scaling_construction()
    test_line_of_sight_vectors()
    test_range_impact()
