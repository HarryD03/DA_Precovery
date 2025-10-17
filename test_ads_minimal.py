#!/usr/bin/env python3
"""
Minimal test to debug ADS splitting issue
"""

import numpy as np
from daceypy import DA, array, ADS

def test_ads_minimal():
    """Test ADS with a simple function that should definitely split"""
    
    # Initialize DA with 2 variables, order 1
    DA.init(2, 1)
    
    # Create a simple quadratic function that varies significantly
    def simple_func(domain):
        x, y = domain.box[0], domain.box[1]
        # Simple function: f(x,y) = x^2 + y^2
        # This should have large variations across [-1,1] x [-1,1]
        result = x**2 + y**2
        # Return as DA array (this might be the issue!)
        return array([result])
    
    # Create initial domain as numpy array (NOT DA variables)
    domain_box = np.array([0.0, 0.0])  # Center point
    initial_domain = ADS(domain_box, [])
    
    # Test function evaluation
    print("=== Testing function evaluation ===")
    try:
        result = simple_func(initial_domain)
        print(f"Function evaluation successful: {result}")
        print(f"Result type: {type(result)}")
        if hasattr(result, 'manifold'):
            print(f"Result constant: {result.manifold.cons()}")
    except Exception as e:
        print(f"Function evaluation failed: {e}")
        return
    
    # Try ADS evaluation
    print("\n=== Testing ADS evaluation ===")
    tol = np.array([0.1])  # Very loose tolerance
    nsplits_limit = 5
    
    try:
        final_list = ADS.eval([initial_domain], tol, nsplits_limit, simple_func)
        print(f"ADS completed successfully!")
        print(f"Number of final domains: {len(final_list)}")
        for i, domain in enumerate(final_list):
            print(f"Domain {i}: nsplits = {len(domain.nsplit)}, box = {domain.box}")
    except Exception as e:
        print(f"ADS evaluation failed: {e}")

def test_ads_minimal_fixed():
    """Test ADS with correct return type"""
    
    # Initialize DA with 2 variables, order 1  
    DA.init(2, 1)
    
    def simple_func_fixed(domain):
        x, y = domain.box[0], domain.box[1]
        # Create DA variables from the domain
        x_da = DA(x, [1, 0])  # x variable
        y_da = DA(y, [0, 1])  # y variable
        
        # Compute function with DA arithmetic
        result_da = x_da**2 + y_da**2
        
        # Return as array of DA
        return array([result_da])
    
    # Create initial domain
    domain_box = array([0.0, 0.0])
    initial_domain = ADS(domain_box, [])
    
    print("=== Testing fixed function ===")
    try:
        result = simple_func_fixed(initial_domain)
        print(f"Function evaluation successful: {result}")
    except Exception as e:
        print(f"Function evaluation failed: {e}")
        return
    
    # Try ADS evaluation
    tol = np.array([0.1])
    nsplits_limit = 5
    
    try:
        final_list = ADS.eval([initial_domain], tol, nsplits_limit, simple_func_fixed)
        print(f"ADS completed successfully!")
        print(f"Number of final domains: {len(final_list)}")
    except Exception as e:
        print(f"ADS evaluation failed: {e}")

if __name__ == "__main__":
    print("Testing basic ADS functionality...")
    test_ads_minimal()
    print("\n" + "="*50 + "\n")
    test_ads_minimal_fixed()
