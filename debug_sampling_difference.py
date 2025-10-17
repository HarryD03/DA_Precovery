"""
Diagnostic script to show why DA and pointwise evaluations sample different points.
"""

import numpy as np

def compare_sampling_strategies():
    """
    Compare the exact sampling points used by pointwise vs DA evaluation.
    """
    print("COMPARISON OF SAMPLING STRATEGIES")
    print("="*60)
    
    # Test parameters (same as in main script)
    ra_nom = np.array([1.93, 1.94, 1.95])    # 3 time epochs
    dec_nom = np.array([0.326, 0.327, 0.328])
    ra_sigma = 1e-5
    dec_sigma = 1e-5
    
    boundary_multipliers = [-3, 0, 3]
    
    print("Nominal observations:")
    print(f"  RA:  {ra_nom}")
    print(f"  DEC: {dec_nom}")
    print(f"  Uncertainties: ±{ra_sigma:.1e} rad")
    
    print(f"\n1. POINTWISE EVALUATION SAMPLING:")
    print("   Perturbs EACH observation independently")
    
    pointwise_points = []
    for i, (ra_mult, dec_mult) in enumerate([(ra, dec) for ra in boundary_multipliers for dec in boundary_multipliers if not (ra == 0 and dec == 0)]):
        print(f"\n   Point {i+1}: RA_mult={ra_mult:+2}, DEC_mult={dec_mult:+2}")
        
        # This is what pointwise does - perturb each epoch independently
        ra_pert = ra_nom + ra_mult * ra_sigma  # Each epoch gets same perturbation
        dec_pert = dec_nom + dec_mult * dec_sigma
        
        print(f"     RA_perturbed:  [{ra_pert[0]:.6f}, {ra_pert[1]:.6f}, {ra_pert[2]:.6f}]")
        print(f"     DEC_perturbed: [{dec_pert[0]:.6f}, {dec_pert[1]:.6f}, {dec_pert[2]:.6f}]")
        
        # Store the perturbations for comparison
        pointwise_points.append({
            'ra_mult': ra_mult, 'dec_mult': dec_mult,
            'ra_pert': ra_pert.copy(), 'dec_pert': dec_pert.copy()
        })
    
    print(f"\n2. DA EVALUATION SAMPLING:")
    print("   Uses normalized domain [-1, +1] for ALL epochs simultaneously")
    
    da_points = []
    for i, (ra_mult, dec_mult) in enumerate([(ra, dec) for ra in boundary_multipliers for dec in boundary_multipliers if not (ra == 0 and dec == 0)]):
        print(f"\n   Point {i+1}: RA_mult={ra_mult:+2}, DEC_mult={dec_mult:+2}")
        
        # This is what DA does - normalized domain
        da_ra = ra_mult / 3.0   # -1, 0, +1
        da_dec = dec_mult / 3.0
        
        # DA uses same perturbation for all epochs
        eval_point = np.array([da_ra, da_ra, da_ra, da_dec, da_dec, da_dec])
        
        print(f"     DA_eval_point: [{eval_point[0]:+.3f}, {eval_point[1]:+.3f}, {eval_point[2]:+.3f}, {eval_point[3]:+.3f}, {eval_point[4]:+.3f}, {eval_point[5]:+.3f}]")
        print(f"     (corresponds to all epochs with same ±{abs(da_ra):.3f}σ perturbation)")
        
        da_points.append({
            'ra_mult': ra_mult, 'dec_mult': dec_mult,
            'eval_point': eval_point.copy()
        })
    
    print(f"\n3. KEY DIFFERENCES:")
    print("="*60)
    print("POINTWISE:")
    print("  • Perturbs physical angles (radians)")
    print("  • Same perturbation applied to all 3 time epochs")
    print("  • Runs full IOD from perturbed observations")
    print("  • Each point requires expensive IOD computation")
    
    print("\nDA EVALUATION:")
    print("  • Uses normalized mathematical domain [-1, +1]")
    print("  • Same normalized perturbation for all 6 variables (3 RA + 3 DEC)")
    print("  • Evaluates pre-computed polynomial (fast)")
    print("  • Polynomial already encodes the IOD transformation")
    
    print(f"\n4. WHY THEY'RE DIFFERENT:")
    print("="*60)
    print("Even though both use boundary_multipliers [-3, 0, 3]:")
    print("  • Pointwise: Physical perturbations → IOD → State vectors")
    print("  • DA: Normalized perturbations → Polynomial evaluation → State vectors")
    print("  • The DA polynomial represents a DIFFERENT uncertainty mapping")
    print("  • DA domain [-1,+1] ≠ Physical domain [±3σ]")
    
    return pointwise_points, da_points

def demonstrate_coordinate_mapping():
    """Show how DA domain maps to physical domain."""
    print(f"\n5. COORDINATE MAPPING ISSUE:")
    print("="*60)
    
    ra_sigma = 1e-5
    
    print("Physical domain (what pointwise uses):")
    print(f"  -3σ = -3 × {ra_sigma:.1e} = {-3*ra_sigma:.1e} rad")
    print(f"   0σ = 0")
    print(f"  +3σ = +3 × {ra_sigma:.1e} = {+3*ra_sigma:.1e} rad")
    
    print("\nDA normalized domain (what DA uses):")
    print("  -1.0 (maps to some physical perturbation)")
    print("   0.0 (maps to nominal)")
    print("  +1.0 (maps to some physical perturbation)")
    
    print(f"\nThe mapping DA_domain → Physical_domain is:")
    print("  Physical_perturbation = DA_value × uncertainty_scaling")
    print("  Where uncertainty_scaling depends on how DA was initialized")
    print("  This scaling may NOT be exactly 3σ!")

if __name__ == "__main__":
    pointwise_points, da_points = compare_sampling_strategies()
    demonstrate_coordinate_mapping()
    
    print(f"\n6. CONCLUSION:")
    print("="*60)
    print("The points are different because:")
    print("1. Different coordinate systems (physical vs normalized)")
    print("2. Different computation methods (IOD vs polynomial)")
    print("3. DA uncertainty scaling may differ from ±3σ assumption")
    print("4. Both are valid but represent different uncertainty sampling approaches")
