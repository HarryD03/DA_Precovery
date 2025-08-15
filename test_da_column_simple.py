import numpy as np
from daceypy import DA, array

def test_da_column_dependency():
    """
    Direct test: Is the third column error due to the third DA variable?
    """
    print("=== TESTING DA COLUMN DEPENDENCY ===")
    
    # Simple function that depends on all three DA variables
    def test_function_da(x1, x2, x3):
        """Simple function for testing DA variable dependencies"""
        return array([
            x1 + x2 + x3,           # f1: depends on all variables
            x1 * x2 + x3,           # f2: nonlinear in x1,x2, linear in x3  
            x1 * x2 * x3            # f3: nonlinear in all variables
        ])
    
    def test_function_numeric(vars):
        """Numeric version for finite differences"""
        x1, x2, x3 = vars
        return np.array([
            x1 + x2 + x3,
            x1 * x2 + x3,
            x1 * x2 * x3
        ])
    
    # Test values
    x1_val, x2_val, x3_val = 2.0, 3.0, 4.0
    
    print(f"Testing at point: x1={x1_val}, x2={x2_val}, x3={x3_val}")
    
    # Initialize DA with 3 variables
    DA.init(4, 3)
    
    # Create DA variables
    x1_da = x1_val + DA(1)  # First DA variable
    x2_da = x2_val + DA(2)  # Second DA variable  
    x3_da = x3_val + DA(3)  # Third DA variable
    
    # Compute DA result
    result_da = test_function_da(x1_da, x2_da, x3_da)
    
    # Extract DA Jacobian
    da_jacobian = np.zeros((3, 3))
    for i in range(3):  # function components
        for j in range(3):  # variables
            da_jacobian[i, j] = result_da[i].deriv(j+1).cons()
    
    print("\nDA Jacobian:")
    print(da_jacobian)
    
    # Analytical Jacobian (we know the derivatives)
    analytical_jacobian = np.array([
        [1, 1, 1],                              # ∂f1/∂[x1,x2,x3] = [1,1,1]
        [x2_val, x1_val, 1],                   # ∂f2/∂[x1,x2,x3] = [x2,x1,1] = [3,2,1]
        [x2_val*x3_val, x1_val*x3_val, x1_val*x2_val]  # ∂f3/∂[x1,x2,x3] = [x2*x3,x1*x3,x1*x2] = [12,8,6]
    ])
    
    print("\nAnalytical Jacobian:")
    print(analytical_jacobian)
    
    # Compute errors
    error_matrix = np.abs(da_jacobian - analytical_jacobian)
    print(f"\nAbsolute Error Matrix:")
    print(error_matrix)
    
    # Check column-wise errors
    column_max_errors = np.max(error_matrix, axis=0)
    print(f"\nMax error by column:")
    for i, error in enumerate(column_max_errors):
        print(f"  Column {i+1} (DA({i+1})): {error}")
    
    # Test DA variable reordering
    print(f"\n=== TESTING DA VARIABLE REORDERING ===")
    
    # Test with different DA variable assignments
    orderings = [
        ("x1=DA(1), x2=DA(2), x3=DA(3)", [1, 2, 3]),
        ("x1=DA(3), x2=DA(2), x3=DA(1)", [3, 2, 1]),
        ("x1=DA(2), x2=DA(1), x3=DA(3)", [2, 1, 3]),
        ("x1=DA(1), x2=DA(3), x3=DA(2)", [1, 3, 2])
    ]
    
    for desc, ordering in orderings:
        print(f"\n--- {desc} ---")
        
        DA.init(4, 3)
        x1_da = x1_val + DA(ordering[0])
        x2_da = x2_val + DA(ordering[1]) 
        x3_da = x3_val + DA(ordering[2])
        
        result_da = test_function_da(x1_da, x2_da, x3_da)
        
        # Extract derivatives w.r.t. x3 (which uses DA(ordering[2]))
        f1_dx3 = result_da[0].deriv(ordering[2]).cons()
        f2_dx3 = result_da[1].deriv(ordering[2]).cons()
        f3_dx3 = result_da[2].deriv(ordering[2]).cons()
        
        # Expected derivatives w.r.t. x3
        expected_f1_dx3 = 1
        expected_f2_dx3 = 1  
        expected_f3_dx3 = x1_val * x2_val  # = 6
        
        errors = [
            abs(f1_dx3 - expected_f1_dx3),
            abs(f2_dx3 - expected_f2_dx3),
            abs(f3_dx3 - expected_f3_dx3)
        ]
        
        print(f"  ∂f/∂x3 using DA({ordering[2]}):")
        print(f"    f1: {f1_dx3} (expected: {expected_f1_dx3}, error: {errors[0]})")
        print(f"    f2: {f2_dx3} (expected: {expected_f2_dx3}, error: {errors[1]})")
        print(f"    f3: {f3_dx3} (expected: {expected_f3_dx3}, error: {errors[2]})")
        
        max_error = max(errors)
        if max_error > 1e-12:
            print(f"  ⚠️  DA({ordering[2]}) shows errors! Max error: {max_error}")
        else:
            print(f"  ✅ DA({ordering[2]}) is accurate")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    if column_max_errors[2] > column_max_errors[0] and column_max_errors[2] > column_max_errors[1]:
        print("🎯 CONFIRMED: Third column (DA(3)) has larger errors")
        print("This suggests DA variable index dependency in the error pattern")
    else:
        print("❓ Error pattern is not clearly related to DA variable index")
        
    return column_max_errors

if __name__ == "__main__":
    test_da_column_dependency()
