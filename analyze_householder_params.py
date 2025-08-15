#!/usr/bin/env python3

import pickle
import numpy as np
from daceypy import DA, array
import daceypy.op as op

def analyze_householder_params():
    """Analyze the saved Householder parameters and compare float vs DA"""
    
    try:
        # Load the parameters from the pickle file
        with open('householder_params.pkl', 'rb') as f:
            p = pickle.load(f)
        
        print("=== LOADED HOUSEHOLDER PARAMETERS ===")
        print(f"Parameters loaded: {len(p)} items")
        
        T_da, L_da, M = p
        
        print(f"T (time parameter):")
        print(f"  Type: {type(T_da)}")
        print(f"  Constant part: {T_da.cons()}")
        print(f"  Full DA object: {T_da}")
        
        print(f"\nL (lambda parameter):")
        print(f"  Type: {type(L_da)}")
        print(f"  Constant part: {L_da.cons()}")
        print(f"  Full DA object: {L_da}")
        
        print(f"\nM (revolution number): {M}")
        
        # Extract constant parts for float comparison
        T_float = T_da.cons()
        L_float = L_da.cons()
        
        print(f"\n=== EXTRACTED FLOAT VALUES ===")
        print(f"T_float = {T_float}")
        print(f"L_float = {L_float}")
        print(f"M = {M}")
        
        # Check if values are reasonable
        print(f"\n=== VALIDITY CHECKS ===")
        print(f"T > 0: {T_float > 0}")
        print(f"|L| < 1: {abs(L_float) < 1}")
        print(f"M >= 0: {M >= 0}")
        
        return T_float, L_float, M
        
    except FileNotFoundError:
        print("Error: householder_params.pkl not found. Run the DAIOD test first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None, None, None

def compare_with_float_householder(T_float, L_float, M, x0=None):
    """Compare the DA parameters with a float Householder iteration"""
    
    if T_float is None:
        print("Cannot run comparison - parameters not loaded")
        return
    
    # Use a reasonable initial guess if not provided
    if x0 is None:
        # Simple initial guess - you can adjust this
        x0 = 0.5
    
    print(f"\n=== FLOAT HOUSEHOLDER COMPARISON ===")
    print(f"Initial guess x0 = {x0}")
    print(f"Using T = {T_float}, L = {L_float}, M = {M}")
    
    # You would implement the float version here
    # For now, just show what parameters would be used
    print("Parameters ready for float Householder iteration")
    
    # TODO: Add actual float Householder implementation
    # This would call the householderfloat function with these parameters

if __name__ == "__main__":
    # Analyze the saved parameters
    T_float, L_float, M = analyze_householder_params()
    
    # Run comparison if parameters were loaded successfully
    if T_float is not None:
        compare_with_float_householder(T_float, L_float, M)
