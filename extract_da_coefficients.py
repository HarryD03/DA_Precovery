"""
Extract DA coefficients for specific independent variables.
Shows how to get coefficients of the 7th (or any) independent DA variable.
"""

import numpy as np
from daceypy import DA, array

def demonstrate_da_coefficient_extraction():
    """
    Demonstrate different ways to extract DA coefficients.
    """
    print("="*70)
    print("DA COEFFICIENT EXTRACTION METHODS")
    print("="*70)
    
    # Initialize DA with 7 variables, order 3
    DA.init(3, 7)
    
    # Create a sample DA polynomial with all 7 variables
    x = DA(1)  # 1st variable
    y = DA(2)  # 2nd variable
    z = DA(3)  # 3rd variable
    u = DA(4)  # 4th variable
    v = DA(5)  # 5th variable
    w = DA(6)  # 6th variable
    t = DA(7)  # 7th variable
    
    # Create a complex polynomial involving all variables
    polynomial = 5.0 + 2*x + 3*y + 4*z + 1.5*u + 2.5*v + 3.5*w + 4.5*t + \
                 0.1*x*y + 0.2*x*z + 0.3*y*z + 0.4*u*v + 0.5*w*t + \
                 0.01*x*y*z + 0.02*u*v*w + 0.03*t*x*y
    
    print(f"Sample polynomial created with 7 DA variables")
    print(f"Constant term: {polynomial.cons()}")
    
    # Method 1: Extract coefficient of 7th variable (linear term)
    print(f"\nMethod 1: Linear coefficient of 7th variable")
    coeff_7th_linear = polynomial.deriv(7)  # Get coefficient of DA(7)^1
    print(f"  Coefficient of 7th variable (linear): {coeff_7th_linear}")
    
    # Method 2: Extract coefficients for different orders of 7th variable
    print(f"\nMethod 2: Different order coefficients of 7th variable")
    try:
        coeff_7th_order1 = polynomial.deriv([7])  # Order 1: coefficient of t^1
        coeff_7th_order2 = polynomial.deriv([7, 7])  # Order 2: coefficient of t^2
        coeff_7th_order3 = polynomial.deriv([7, 7, 7])  # Order 3: coefficient of t^3
        
        print(f"  7th variable order 1: {coeff_7th_order1}")
        print(f"  7th variable order 2: {coeff_7th_order2}")
        print(f"  7th variable order 3: {coeff_7th_order3}")
    except Exception as e:
        print(f"  Error with deriv([7, 7, ...]): {e}")
    
    # Method 3: Extract mixed coefficients involving 7th variable
    print(f"\nMethod 3: Mixed coefficients involving 7th variable")
    try:
        # Coefficient of t*x (7th variable * 1st variable)
        coeff_7th_1st = polynomial.deriv([7, 1])
        print(f"  Coefficient of 7th*1st variables: {coeff_7th_1st}")
        
        # Coefficient of t*x*y (7th * 1st * 2nd variables)
        coeff_7th_1st_2nd = polynomial.deriv([7, 1, 2])
        print(f"  Coefficient of 7th*1st*2nd variables: {coeff_7th_1st_2nd}")
    except Exception as e:
        print(f"  Error with mixed derivatives: {e}")
    
    # Method 4: Extract all coefficients involving 7th variable
    print(f"\nMethod 4: All terms involving 7th variable")
    
    # Check all possible combinations up to order 3
    max_order = DA.getMaxOrder()
    all_7th_coeffs = []
    
    for order in range(1, max_order + 1):
        for combo in generate_derivative_combinations(7, order, 7):  # 7th variable, up to total variables 7
            try:
                coeff = polynomial.deriv(combo)
                if abs(coeff) > 1e-12:  # Only non-zero coefficients
                    all_7th_coeffs.append((combo, coeff))
                    print(f"  deriv{combo}: {coeff}")
            except:
                continue
    
    print(f"\nFound {len(all_7th_coeffs)} non-zero coefficients involving 7th variable")

def generate_derivative_combinations(target_var, max_order, max_var):
    """
    Generate all derivative combinations involving the target variable.
    """
    combinations = []
    
    # Single variable terms
    for order in range(1, max_order + 1):
        combinations.append([target_var] * order)
    
    # Mixed terms with other variables
    if max_order >= 2:
        for other_var in range(1, max_var + 1):
            if other_var != target_var:
                combinations.append([target_var, other_var])
                if max_order >= 3:
                    combinations.append([target_var, target_var, other_var])
                    combinations.append([target_var, other_var, other_var])
                    for third_var in range(1, max_var + 1):
                        if third_var != target_var and third_var != other_var:
                            combinations.append([target_var, other_var, third_var])
    
    return combinations

def extract_specific_da_variable_coefficients(da_polynomial, variable_index):
    """
    Extract all coefficients involving a specific DA variable.
    
    Args:
        da_polynomial: DA polynomial object
        variable_index: Index of the DA variable (1-based)
    
    Returns:
        dict: Dictionary with derivative combinations as keys and coefficients as values
    """
    print(f"\nExtracting all coefficients for DA variable {variable_index}:")
    
    coefficients = {}
    max_order = DA.getMaxOrder()
    max_vars = DA.getMaxVariables()
    
    # Linear term
    try:
        linear_coeff = da_polynomial.deriv(variable_index)
        if abs(linear_coeff) > 1e-12:
            coefficients[f"DA({variable_index})"] = linear_coeff
            print(f"  Linear term DA({variable_index}): {linear_coeff}")
    except:
        pass
    
    # Higher order and mixed terms
    for order in range(2, max_order + 1):
        # Pure higher order terms (e.g., DA(7)^2, DA(7)^3)
        try:
            pure_coeff = da_polynomial.deriv([variable_index] * order)
            if abs(pure_coeff) > 1e-12:
                coefficients[f"DA({variable_index})^{order}"] = pure_coeff
                print(f"  Pure order-{order} term DA({variable_index})^{order}: {pure_coeff}")
        except:
            pass
        
        # Mixed terms with other variables
        for other_var in range(1, max_vars + 1):
            if other_var != variable_index:
                try:
                    mixed_coeff = da_polynomial.deriv([variable_index, other_var])
                    if abs(mixed_coeff) > 1e-12:
                        coefficients[f"DA({variable_index})*DA({other_var})"] = mixed_coeff
                        print(f"  Mixed term DA({variable_index})*DA({other_var}): {mixed_coeff}")
                except:
                    pass
    
    return coefficients

def practical_example_from_daiod():
    """
    Practical example showing how to extract 7th variable coefficients from DAIOD results.
    """
    print(f"\n" + "="*70)
    print("PRACTICAL EXAMPLE: EXTRACTING 7th VARIABLE FROM DAIOD")  
    print("="*70)
    
    # Simulate a DAIOD result with 6 angular variables + 1 additional parameter
    DA.init(3, 7)  # Order 3, 7 variables
    
    # Variables 1-6 are RA/DEC for 3 observations, variable 7 is additional parameter
    RA1, RA2, RA3 = DA(1), DA(2), DA(3)
    DEC1, DEC2, DEC3 = DA(4), DA(5), DA(6)
    additional_param = DA(7)  # This is our 7th variable
    
    # Create a mock DAIOD result (position component)
    position_x = 1.5e8 + 1000*RA1 + 800*RA2 + 600*RA3 + \
                 500*DEC1 + 400*DEC2 + 300*DEC3 + \
                 2000*additional_param + \
                 50*RA1*DEC1 + 30*RA2*additional_param + \
                 10*additional_param*additional_param
    
    print(f"Mock DAIOD position_x created with 7 DA variables")
    print(f"Nominal position: {position_x.cons():.2e} km")
    
    # Extract coefficients for the 7th variable (additional parameter)
    coeffs_7th = extract_specific_da_variable_coefficients(position_x, 7)
    
    print(f"\nSummary of 7th variable coefficients:")
    for term, coeff in coeffs_7th.items():
        print(f"  {term}: {coeff}")
    
    # Show how these coefficients affect the result
    print(f"\nEffect of 7th variable on position:")
    
    # Evaluate at different values of 7th variable
    for val in [-1, 0, 1]:
        eval_point = [0, 0, 0, 0, 0, 0, val]  # Only 7th variable perturbed
        result = position_x.eval(eval_point)
        change = result - position_x.cons()
        print(f"  7th variable = {val:+2d}: position change = {change:+8.1f} km")

if __name__ == "__main__":
    demonstrate_da_coefficient_extraction()
    practical_example_from_daiod()
