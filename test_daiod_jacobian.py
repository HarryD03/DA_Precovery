import numpy as np
import pytest
from utils.iod import DAIOD_1
from scipy.optimize import approx_fprime
from daceypy import DA, array
import daceypy.op as op
from utils.lambert_izzo import lambert_izzo

def test_daiod_jacobian_comparison():
    """
    Compare the Jacobian from newton_nominal_DAVec in DAIOD context 
    with scipy finite difference Jacobian
    """
    # Use the same sample data from test_iod.py
    pos_obs = np.array([
        [3489.8, 3460.1, 3429.9],          
        [3430.2, 3460.1, 3490.1],          
        [4078.5, 4078.5, 4078.5]           
    ])

    obs_dir = np.array([            
        [0.71643, 0.56897, 0.41841],            
        [0.68074, 0.79531, 0.87007],            
        [-0.15270,-0.20917,-0.26059]            
    ])

    t = np.array([0.0, 118.10, 237.58])
    mu = 3.986e5
    
    # Get Gauss initial guess
    from utils.iod import Guass_8th_seed
    _, _, range_mag_guass, _ = Guass_8th_seed(pos_obs, obs_dir, t, mu=mu)
    
    print(f"Initial Gauss range_mag: {range_mag_guass}")
    
    # Define the DAIOD function f(range_mag) -> velocity difference
    def daiod_function(range_mag):
        """
        Reproduce the f function from DAIOD for standalone testing
        """
        # Convert obs_dir to line-of-sight vectors
        i_rho = obs_dir  # Assuming obs_dir is already normalized
        
        range_vec = np.zeros_like(i_rho)
        for i in range(len(range_mag)):
            range_vec[:,i] = range_mag[i] * i_rho[:,i]

        r_vec = np.zeros_like(range_vec)
        for i in range(len(range_vec[0])):                  
            r_vec[:,i] = range_vec[:,i] + pos_obs[:,i]

        vel = []
        for i in range(len(r_vec[0]) - 1):                  
            velocities = lambert_izzo(r_vec[:,i], r_vec[:,i+1], t[i+1] - t[i], mu, 0, prograde=True)
            solution = velocities[0]
            v1 = solution[:,0]
            v2 = solution[:,1]
            vel.append(v1)
            vel.append(v2)

        # Centre positions should have zero velocity difference
        v2_plus = vel[2]    # velocity at obs 2 from arc 2->3             
        v2_minus = vel[1]   # velocity at obs 2 from arc 1->2

        DV = (v2_plus - v2_minus)
        return DV.cons() if hasattr(DV, 'cons') else DV

    # Test function evaluation at initial guess
    dv_initial = daiod_function(range_mag_guass)
    print(f"Initial velocity difference: {dv_initial}")
    print(f"Initial |DV|: {np.linalg.norm(dv_initial)}")
    
    # Define DA version for Jacobian extraction
    def daiod_function_da(range_mag_da):
        """DA version of DAIOD function"""
        i_rho = obs_dir
        
        range_vec = range_mag_da * i_rho

        r_vec = range_vec + pos_obs

        vel = []
        for i in range(2):                  
            velocities = lambert_izzo(r_vec[:,i], r_vec[:,i+1], t[i+1] - t[i], mu, 0, prograde=True)
            solution = velocities[0]
            v1 = solution[:,0]
            v2 = solution[:,1]
            vel.append(v1)
            vel.append(v2)

        v2_plus = vel[2]                
        v2_minus = vel[1]
        DV = (v2_plus - v2_minus)
        return DV

    # Initialize DA for Jacobian computation
    n_vars = len(range_mag_guass)  # 3 range magnitudes
    DA.init(4, n_vars)
    
    # Create DA variables for range magnitudes
    range_mag_da = array([range_mag_guass[i] + DA(i+1) for i in range(n_vars)])
    
    # Compute DA version
    dv_da = daiod_function_da(range_mag_da)
    
    # Extract DA Jacobian
    da_jacobian = np.zeros((3, n_vars))  # 3 velocity components, n_vars range components
    for i in range(3):  # velocity components
        for j in range(n_vars):  # range magnitude variables
            da_jacobian[i, j] = dv_da[i].deriv(j+1).cons()
    
    print("\n=== DA JACOBIAN ===")
    print("∂[dvx, dvy, dvz]/∂[range1, range2, range3] =")
    print(da_jacobian)
    dv_da.linear()
    
    # Compute finite difference Jacobian for reference
    def dv_func_component(range_mag, component_idx):
        """Return specific component of velocity difference"""
        dv = daiod_function(range_mag)
        return dv[component_idx]
    
    fd_jacobian = np.zeros((3, n_vars))
    h = 1e-15  # Step size
    
    for i in range(3):  # velocity components
        def func_i(range_mag):
            return dv_func_component(range_mag, i)
        fd_jacobian[i, :] = approx_fprime(range_mag_guass, func_i, h)
    
    print("\n=== FINITE DIFFERENCE JACOBIAN ===")
    print("∂[dvx, dvy, dvz]/∂[range1, range2, range3] =")
    print(fd_jacobian)
    
    # Compare Jacobians
    jac_error = np.abs(da_jacobian - fd_jacobian)
    max_error = np.max(jac_error)
    norm_error = np.linalg.norm(jac_error)
    
    print(f"\n=== JACOBIAN COMPARISON ===")
    print(f"Max absolute error: {max_error}")
    print(f"Norm error: {norm_error}")
    print(f"Relative error matrix:")
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = jac_error / np.abs(fd_jacobian)
        rel_error[np.isnan(rel_error)] = 0  # Handle division by zero
        print(rel_error)
    
    # Check convergence properties
    tolerance = h*10            # tolerance is 10 times greater than step size
    jacobian_matches = max_error < tolerance
    
    print(f"\nJacobian matches (tolerance={tolerance}): {jacobian_matches}")
    
    if not jacobian_matches:
        print("❌ DA Jacobian doesn't match finite differences!")
        print("This suggests issues with DA automatic differentiation in DAIOD context")
        
        # Check if Jacobian is singular
        det_da = np.linalg.det(da_jacobian)
        det_fd = np.linalg.det(fd_jacobian)
        print(f"DA Jacobian determinant: {det_da}")
        print(f"FD Jacobian determinant: {det_fd}")
        
        if abs(det_da) < 1e-12:
            print("⚠️  DA Jacobian is nearly singular!")
        if abs(det_fd) < 1e-12:
            print("⚠️  FD Jacobian is nearly singular!")
            
    else:
        print("✅ DA Jacobian matches finite differences")
        
    return da_jacobian, fd_jacobian, jac_error

if __name__ == "__main__":
    test_daiod_jacobian_comparison()
