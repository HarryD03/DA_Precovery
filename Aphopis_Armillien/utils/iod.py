import numpy as np
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray

def f_g_series(r0, dt, f_order, g_order, mu=1.32712440018e11, v0=None):
    """
    Generate the f and g series for the Kepler problem to a maximum order of 4.
    :param r0: Initial position vector (1D column vector)
    :param dt: Time of flight in seconds
    :param f_order: Order of the f series expansion
    :param g_order: Order of the g series expansion
    :param v0: Initial velocity vector (optional, if not provided, only second order series is generated)
    :param mu: Standard gravitational parameter (default is the Sun's)
    :return: f_series, g_series - numpy arrays of the f and g series coefficients
    """
    assert r0.ndim == 1 and r0.shape[0] == 3, "r0 must be a 1D column vector with 3 components"
    assert v0 is None or (v0.ndim == 1 and v0.shape[0] == 3), "v0 must be a 1D column vector with 3 components or None"
    assert isinstance(dt, (int, float)), "dt must be a number"
    assert isinstance(f_order, int) and f_order >= 0, "f_order must be a non-negative integer"
    assert isinstance(g_order, int) and g_order >= 0, "g_order must be a non-negative integer"


    # Calculate the norm of the initial position vector
    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0) if v0 is not None else 0.0
    assert r0_norm > 0, "Initial position vector r0 must not be zero"
    
    # Precompute the An, Bn and M - do later for general case
    if v0 is None:
        v0 = np.zeros(3).T  # If no initial velocity is provided, assume zero velocity
    
    # expand series for f and g
    f_series = np.array([1.0,                                           #0th order
                        0.0,                                           #1st order
                        -(mu/(2*r0_norm**3))*dt**2,                     #2nd order
                        (mu/2*(np.dot(r0,v0))/r0_norm**5) * dt**3,     #3rd order                                      #3rd order
                        (mu/24 * (-2*mu/r0_norm**6) + 3*v0_norm**2/r0_norm**6 - 15*(np.dot(r0,v0))/r0_norm**7) * dt**4])  #4th order                                         #4th order

    g_series = np.array([0.0,                                           #0th order
                         dt,                                            #1st order 
                         0.0,                                           #2nd order
                         -1/6 * (mu/r0_norm**3) * dt**3,                 #3rd order
                         mu/4 * (np.dot(r0,v0)/ r0_norm**5) * dt**4])   #4th order

    # Ensure the series are of the correct order
    if f_order < len(f_series):
        f_series = f_series[:f_order+1]

    f_series = f_series.sum()


    if g_order < len(g_series):
        g_series = g_series[:g_order+1]
    
    g_series = g_series.sum()

    return f_series, g_series

def position_feasibility(r1: NDArray[np.double], r2: NDArray[np.double], r3: NDArray[np.double], v2: NDArray[np.double], mu, Re: float = 6378.137) -> bool:
    """
    Check if the Guass IOD positions are physically feasible.

    Paramters:
    r: column vectors of positions [x, y, z].T in kilometers
        Position vectors for the three observations. rows correspond to the r_2 roots, columns are the positions values.
    Re : float
        Earth radius in kilometers

    Returns:
    bool
        true if position are feasible, False otherwise
    Rasies:
    ValueError
        If positions fail feasibility check.
    """
    # Pre-condiition checks to ensure inputs are valid
    assert r1.shape == (3,), "r1 must be a 1D column vector"
    assert r2.shape == (3,), "r2 must be a 1D column vector"
    assert isinstance(Re, (int, float)), "Re must be a number"


    d_earth_sun = 1.496e+8  # Earth-Sun distance in kilometers, used for feasibility checks ( 1AU)
    # Test 1a: norm r < Earth Sun distance
    if mu == 1.32712440018e11:
        if np.linalg.norm(r1) <= d_earth_sun or np.linalg.norm(r2) <= d_earth_sun or np.linalg.norm(r3) <= d_earth_sun:
            r1[:] = np.nan
            r2[:] = np.nan
            r3[:] = np.nan
            v2[:] = np.nan
            raise ValueError("Position vectors r1 and r2 must be greater than distance of Earth as night time viewing).")
    elif mu == 3.986e5:
        if np.linalg.norm(r1) <= Re or np.linalg.norm(r2) <= Re or np.linalg.norm(r3) <= Re:
            r1[:] = np.nan
            r2[:] = np.nan
            r3[:] = np.nan
            v2[:] = np.nan
            raise ValueError("Position vectors r1 and r2 must be greater than Radius of earth).")
    
    # Test 3: Specific energy is negative - asteroid elliptic orbit
    specific_energy = (np.linalg.norm(v2)**2 / 2) - mu / (np.linalg.norm(r2))
    if specific_energy >= 0:
        r1[:] = np.nan
        r2[:] = np.nan
        r3[:] = np.nan
        raise ValueError(f"Specific energy is not negative (non-elliptic): {specific_energy}")

    # Test 4: Angular Momentum is positive
    angular_momentum = np.dot(r1, np.cross(r2, r3))
    if angular_momentum <= 0:
        r1[:] = np.nan
        r2[:] = np.nan
        r3[:] = np.nan
        raise ValueError(f"Angular momentum is not positive: {angular_momentum}")

    return r1, r2, r3, v2 # Return the positions if all checks pass

def Guass_8th_seed(pos_obs: Union[NDArray, array], obs_dir: Union[NDArray, array], t: NDArray, mu=1.32712440018e11) -> NDArray[np.double]:
    #Taken from Orbital Mechanics for Engineering Students (4th ed.) by Curtis. p.242 Algorithm 5.5. and Armillien Aphopis
    #The Seed takes real numbers not DA numbers.
    #Inputs:
    #   Pos_obs: 2D array of Observer positions in Heliocentric frame of reference from angle rotation matrix
    #           Rows are compoents, columns are observations instances
    #   t: 1D array of times in seconds
    #   obs_dir: 2D array of unit vectors pointing from observer to the point of interest, every column is an observation instance the rows are components x y z
    # 2-BP assumption:
    #   Assume the observations lie on the same plane
   
    dt_1 = t[0] - t[1]
    dt_3 = t[2] - t[1]
    dt = dt_3 - dt_1

    #Re-write the cross and dot products in numpy structure

    p1 = np.cross(obs_dir[:,1], obs_dir[:,2])  #cross product of the second and third observation direction vectors
    p2 = np.cross(obs_dir[:,0], obs_dir[:,2])  #cross product of the first and third observation direction vectors
    p3 = np.cross(obs_dir[:,0], obs_dir[:,1])  #cross product of the first and second observation direction vectors

    D0 = np.dot(obs_dir[:,0], p1)  #Dot product of the first observation direction vector and the cross product of the second and third observation direction vectors

    D11 = np.dot(pos_obs[:,0], p1)
    D12 = np.dot(pos_obs[:,0], p2)
    D13 = np.dot(pos_obs[:,0], p3)
    D21 = np.dot(pos_obs[:,1], p1)
    D22 = np.dot(pos_obs[:,1], p2)
    D23 = np.dot(pos_obs[:,1], p3)
    D31 = np.dot(pos_obs[:,2], p1)
    D32 = np.dot(pos_obs[:,2], p2)
    D33 = np.dot(pos_obs[:,2], p3)

    A = 1 / D0 * ((-D12 * dt_3/ dt) + D22 + (D32 * dt_1/dt))
    B = 1 / (6*D0) * (D12*(dt_3**2 - dt**2)*dt_3/dt + (D32*(dt**2 - dt_1**2) * dt_1/dt))

    E = np.dot(pos_obs[:,1], obs_dir[:,1])
    R_2_squared = np.dot(pos_obs[:,1], pos_obs[:,1])

    #obtain coefficient values for the 8th degree position polynomial
    a =  -1*(A**2 + 2*A*E + (R_2_squared))
    b = -2*mu*B*(A+E)
    c = -1 * (mu**2 * B**2)

    #set up coefficients for the 8th degree position polynomial for evaluation
    coeffs = [1.0,          #r^8
              0.0,          #r^7
              a,            #r^6
              0.0,          #r^5
              0.0,          #r^4
              b,            #r^3
              0.0,          #r^2
              0.0,          #r^1
              c             #r^0
              ]

    #Check if 8th degree polynomial coefficients are floats and can be evaluated using numpy
    # Assert all coefficients are floats
    for i, coeff in enumerate(coeffs):
        assert isinstance(coeff, float), f"Coefficient at position {i} is not a float: {type(coeff)}"
        assert not np.isnan(coeff), f"Coefficient at position {i} is NaN"
        assert not np.isinf(coeff), f"Coefficient at position {i} is infinite"

    roots = np.roots(coeffs)                                                    #obtain the roots of the 8th polynomial - np.roots using eignevalue evaluation of coeffs
    real_roots = roots[np.logical_and(np.imag(roots) == 0, np.real(roots) > 0)] #Need physical roots
    assert len(real_roots) > 0, "No positive real roots found"                  #If no real roots print

    #Inform user of real roots values, and conduct pruning if more than 1 real root
    r_2_mag = real_roots                                                #Store possible solutions of r_2
    if len(real_roots) > 0:
        print("There are more than 1 positive real roots.")
        range_1_mag = np.zeros((len(real_roots)))   # columns are different root value, rows are [x,y,z] components            #topocentric ranges
        range_2_mag = np.zeros((len(real_roots)))               #topocentric ranges
        range_3_mag = np.zeros((len(real_roots)))               #topocentric ranges

        r_1 = np.zeros((3,len(real_roots)))
        r_2 = np.zeros((3,len(real_roots)))                   #helocentric positions
        r_3 = np.zeros((3,len(real_roots)))

        print(f"There are {len(real_roots)} positive real roots:\n")

        for i in range(len(real_roots)):
            print(f"Root {i}: {r_2_mag[i]}\n")
            # build range1, range2, range3 and conduct feasibility tests.
            # Ranges are defined through truncated Lagrange coefficients so they're not accurate and refinement required
            
            # Calculate the range_1 and r1
            num = (6*(D31 * (dt_1/dt_3) + D21 * (dt/dt_3))*(r_2_mag[i]**3)) + (mu * D31 * (dt**2 - dt_1**2)* (dt_1/dt_3))
            den = (6 * r_2_mag[i]**3) + (mu * (dt**2 - dt_3**2))
            range_1_mag[i] = 1 / D0 * ( (num / den) - D11)          #slant range magnitude for the first observation

            # Calculate the range_3 and r3
            num = (6*(D13 * (dt_3/dt_1) - D23 * (dt/dt_1))*(r_2_mag[i]**3)) + (mu * D13 * (dt**2 - dt_3**2)* (dt_3/dt_1))
            den = (6 * r_2_mag[i]**3) + (mu * (dt**2 - dt_3**2))
            range_3_mag[i] = 1 / D0 * ((num / den) - D33)           #slant range magnitude for the third observation

            # Calculate the range_2 and r2
            range_2_mag[i] = A + (mu * B) / (r_2_mag[i] ** 3)
    

            # calculate r1,r2, r3
            r_1[:,i] = pos_obs[:,0] + range_1_mag[i]*obs_dir[:,0]
            r_2[:,i] = pos_obs[:,1] + range_2_mag[i]*obs_dir[:,1]
            r_3[:,i] = pos_obs[:,2] + range_3_mag[i]*obs_dir[:,2]

            #obtain v2 velcoity vector at r2 for feasibility check
            
            f_3, g_3 = f_g_series(r_2[:,i], dt_3, 2, 3, mu=mu)  #get f and g series for the third observation
            f_1, g_1 = f_g_series(r_2[:,i], dt_1, 2, 3, mu=mu)  #get f and g series for the first observation
            
            
            
            v2 = 1/((f_1*g_3) - (f_3*g_1)) * (-f_3*r_1[:,i] + f_1*r_3[:,i]) 

            #Assess the 3 positions for feasibility -
            r_1, r_2, r_3, v_2 = position_feasibility(r_1[:,i], r_2[:,i], r_3[:,i], v2, mu)

            if not np.any(np.isnan([r_1, r_2, r_3, v_2])):
                print(f"Feasibility passed for root {i}:")
                print(f"  r_1: {r_1}")
                print(f"  r_2: {r_2}")
                print(f"  r_3: {r_3}")

                range_1 = range_1_mag[i] * obs_dir[:,0]  #convert range to vector
                range_2 = range_2_mag[i] * obs_dir[:,1]  #convert range to vector
                range_3 = range_3_mag[i] * obs_dir[:,2]  #convert range to vector
                break

    else:
        print(f"There is no positive real root: {r_2}\n")
        r_1 = np.full((3, 1), np.nan)  # If no real roots, set the position to NaN
        r_2 = np.full((3, 1), np.nan)  # If no real roots, set the position to NaN
        r_3 = np.full((3, 1), np.nan)  # If no real roots, set the position to NaN
        range_1 = np.full((3, 1), np.nan)  # If no real roots, set the range to NaN
        range_2 = np.full((3, 1), np.nan)  # If no real roots, set the range to NaN
        range_3 = np.full((3, 1), np.nan)  # If no real roots, set the range to NaN


    #If no real roots, set the position and range to zero
    #obtain the [r1,r2,r3] [range1,range2,range3]. Rows are the real root, columns are positions
    position = np.array([r_1, r_2, r_3]).T
    ranges = np.array([range_1, range_2, range_3]).T

    return position, ranges


#
def lambert_battin_DA(r1: Union[array,NDArray], r2: Union[array, NDArray], dt: float, max_iter: int,  mu: float = 3.2712440018e11, order: int = None):
    """
    :parameter:
    :param r1: Float or DA array (x,y,z compoents) of initial poisition
    :param r2: Float or DA array (x,y,z compoents) of final poisition
    :param dt : time of flight
    :param mu : standard gravitional parameter - default is sun
    :param max_iter: maximum number of iterations for Newton iteration
    :param order : Maximum algebraic order of DA variables - default global order

    Returns:
    :param v1 : Float or DA array (x,y,z compoents) of velocity departure
    :param v2 : Float or DA array (x,y,z compoents) of velocity destination
    """
    # -----------------------------------------
    # 1) promote to DA if neccessary & set a sensible algebra
    # -----------------------------------------
    if not isinstance(r1[0], DA) and not isinstance(r2[0], DA):
        print(f"Real")        #nothing happens carry r1/r2 DA settings over - real numbers
    else:
        if order is None:
            #inherit global settings
            order = DA.getMaxOrder(r1[0] if isinstance(r1[0], DA) else r2[0])
        else:
            # make sure current algebra matches requested order
            nvar = DA.getMaxVariables()
            DA.init(order,nvar + 1)     # 3 variables for r1, 3 variables for r2, 1 variable for x

    # --------------------------
    # 2) Geometry calculations - DA aware
    # ---------------------------
    c_vec = r2 - r1
    c = op.vnorm(c_vec)
    r1_mag = op.vnorm(r1)
    r2_mag = op.vnorm(r2)
    s = 0.5 * (r1_mag + r2_mag + c)
    cosTA = ((r1.dot(r2)) / (r1_mag * r2_mag))
    TA = op.acos(cosTA)

    # -----------------------------
    # 3) nomial solution for x - scalar Newton
    # -----------------------------
    def battin_F(x):  # scalar (or DA) residual
        g = s / (2 * (1 - x ** 2))
        alpha = 2 * op.asin(op.sqrt(s / (2 * g)))
        beta = 2 * op.asin(op.sqrt((s - c) / (2 * g)))
        A = g ** 1.5 * (alpha - op.sin(alpha) - beta + op.sin(beta))
        return op.log10(A) - op.log10(dt)  # f(x) = 0

    def x_initial():
        dt_p = np.sqrt(2/mu)*(s.cons()**(3/2) - c.cons()**(3/2))
        T = dt/dt_p
        x0 = (1-T) / (1+T)
        return x0
    
    k = r1[0].getMaxVariables()
    
    x = 1e-3 + DA(k)                    #guess at 0.1

    def fbattin(x, p):
        #unpack p
        r1 = p[0,:]
        r2 = p[1,:]
        

        A = battin_A_DA(x, r1, r2)      #x is too large again - need to review battin lambert paper for first guess of x
        return op.log10(A) - np.log10(dt)
    
    x0 = 0.0
    p = array([r1, r2])
    
    x_nom = newton_nomial_DA(x0, p, fbattin, tol=1e-9, MaxIter=1000)
    
    # -------------------------------
    # 4) Get x_map through newton of battin scalar equation
    # ---------------------------------
    #Get battin's x in terms of DA variables from position perturbations
        #Newton iteartor - can replace with .invert() but need to see iteration number
    if isinstance(r1[0], DA) and isinstance(r2[0], DA):
        x = battin_x_DA(r1, r2, dt, order, x)           #generate DA map if DA exists
    
    #-------------------------------------------
    # 5) Prepare for lagrange coefficients/ velocity calculations
    #-------------------------------------------
    a = s / (2 *  (1 - x**2))
    g = s / (2 * (1 - x ** 2))
    alpha = 2 * op.asin(op.sqrt(s / (2 * g)))
    beta = 2 * op.asin(op.sqrt((s - c) / (2 * g)))
    
    if TA > np.pi:
        beta = -beta

    a_min = s / 2
    t_min = np.sqrt(a_min ** 3 / mu) * (np.pi - beta + op.sin(beta))

    if dt > t_min:
        alpha = 2 * np.pi - alpha

    dE = alpha - beta
    #Calculate the velcotiy of the points
    v2, v1 = battin_vel_DA(r1, r2, a, dE, dt, mu)

    return v2, v1

def battin_A_DA(x_da, r1_da, r2_da):
    """
    Battin scalar function A (see Battin 10.4 or Armellian et al 2012) for battin_x_da function


    where
        g = s/ 2(1-x
    :param x_da: DA or float
        battin parameter
    :param r1_da:  array-like
        initial poisition vector
    :param r2_da: array-like
        final position vector
    :param s:
    :param c:
    :param mu:
    :return:
    """

    #Generate geomtric properties
    c_vec = r2_da - r1_da
    c =     op.vnorm(c_vec)
    r1mag = op.vnorm(r1_da)
    r2mag = op.vnorm(r2_da)
    s = 0.5 * (r1mag + r2mag + c)

    # Battin auxillary g(x)
    g = s / (2 * (1 - x_da**2)) #Issue is g is negative as x_da**2 ^^^^
    al_frac = (s/(2*g)).sqrt()
    be_frac = ((s-c)/(2*g)).sqrt()
    print(al_frac.cons())
    print(be_frac.cons())
    EPS = 1e-12
    
    if al_frac.cons() >= 1.0:
        al_frac = al_frac - EPS
    if be_frac.cons() >= 1.0:
        be_frac = be_frac - EPS
    
    alpha = 2 * (al_frac.arcsin())
    beta = 2 * (be_frac.arcsin())

    #construct A
    A = g**(3/2) * (alpha - alpha.sin() - beta + beta.sin())
    if A.cons() < 0:
        print(f"A is negative")
    return A

def battin_x_DA(r1_da: Union[array,NDArray], r2_da: Union[array,NDArray], dt: float, order: int, x0: float) -> DA:
    """
    :param r1_da: (1,3) ndarray of DA variables
        DA series of initial position vector
    :param r2_da: (1,3) ndarray of DA variables
        DA series of final position vector
    :param tof_nom:
        Nominal time of flight between the two points (no DA)
    :param order:
        Max order in expansion series of r1 etc
    :return:
    x_da:
        DA expansion of Battin's parameter x(d) wrt to r1_da and r2_da
    """

    # 2) get DA battin variable - expand around nominal solution
    x_da = x0 + DA(7)
    # 3) obtain f(x,DA) = ln A - ln dt
    A = battin_A_DA(x_da, r1_da, r2_da)

    def f(x_da):
        return battin_A_DA(x_da, r1_da, r2_da).log() - np.log(dt)
    
    # 4) Newton iteration to obtain DA x. This is the Implicit solver
    p = array([r1_da, r2_da])
    x_da = Implicit_solver_DA(x_da, p, f)

    return x_da

def Nf(x, f, df):
    k = DA.getMaxVariables()

    f = f.plug(k,0)
    df = df.plug(k,0)

    if np.allclose(df, 0):
        raise ZeroDivisionError("Derivative is zero in Newton iteration")
    x1 = x - f/df    
    return x1

def Implicit_solver_DA(x_da: Union[float,DA], p_da: Union[DA,array], f: callable):
        """
            X needs to be the last DA variable
                x_da: Dependant Variable (x_nom + DA(x)) - need to be initalised prior
                p_da: Independant DA Variables (p_nom + DA(p_da))
                f: Function for Newtons method
            Return
                x_da as a function of DA(p_da). x_da = x_nom + DA(p_da)
        """
        i = 1
        k = DA.getMaxVariables()
        x_da = x_da + DA(k)
        
        def df(fx):
            return fx.deriv(k)
    
        while i <= (DA.getMaxOrder()):
          x_da = Nf(x_da, f(x_da), df(f(x_da)))
          i *= 2

        x_da = x_da.plug(k,0)
        return x_da

def battin_vel_DA(r1: Union[array,NDArray], r2: Union[array,NDArray], a: Union[DA,float], dE: Union[DA,float], dt: float, mu: float) -> DA:
    """
    :param r1: 3-vector initial position vector (DA series or real array)
    :param r2: 3-vector final position vector (DA series or real array)
    :param a: Semi major axis DA series
    :param dE: dE = alpha - beta
    :param dt: tof (float)
    :param mu: Standard gravitional parameter (float)
    :return:
    """

    f = 1 - (a / op.vnorm(r1)) * (1 - op.cos(dE))
    g = dt - op.sqrt(a**3 / mu) * (dE - op.sin(dE))
    g_dot = 1 - (a / op.vnorm(r2)) * (1 - op.cos(dE))
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v2, v1

def lagrange_coefficients(a: Union[float, DA], dE: Union[float, DA], r1: Union[array, NDArray], v1: Union[array, NDArray], sigma_1, mu: float) -> Union[array, NDArray]:
        """
        Calculate the lagrange coefficients for position and velocity vectors.
        :param a: Semi-major axis
        :param dE: Eccentric anomaly
        :param r1: Initial position vector
        :param mu: Standard gravitational parameter
        :return: Position and velocity vectors at t0 + dt
        Source: 16.346 Astrodynamics Fall 2008 MIT OpenCourseWare
        """
        f = 1 - (a / op.vnorm(r1)) * (1 - op.cos(dE))
        g = a*sigma_1/op.sqrt(mu) * (1 - op.cos(dE)) + (op.vnorm(r1) * op.sqrt(a/mu) * (op.sin(dE)))
        r2 = f*r1 + g*v1

        ft = - op.sqrt((mu*a)) / ((op.vnorm(r1)*op.vnorm(r2))) * op.sin(dE)
        gt = 1 - (a/(op.vnorm(r2)) * (1 - op.cos(dE)))
        v2 = ft*r1 + gt*v1

        #conservation of angular momentum

        assert np.isclose(f*gt - ft*g, 1, atol=1e-9), f"Lagrange Coefficients incorrect"

        return r2, v2

def newton_nomial_DA(x0, p: Union[DA, array, float, NDArray], f: callable, tol: float , MaxIter: float) -> float:
    """
    Newtons Method applied to DA to obtain Nomial Solution for dependant variable
    DA must have been initialised Prior

    :param x0: Initial guess 
    :param p: Everything other variable in f
    :parma f: Callable function f(x; p) = 0 which will be evaluated at every newton iteration
    :return: Nomial solution for x 
    """

    Max_variable = DA.getMaxVariables()
    
    x = x0 + DA(Max_variable)  
    flag = True
    iter = 1

    while flag:
        F = f(x)

        dF = F.deriv(Max_variable)
        if dF.cons() == 0:
            print(f"Iteration Number: {iter}\n")
            raise ValueError("Derivative became zero during iteration") 
        x -= F.cons()/dF.cons()
        iter += 1

        print(iter)
        if abs(F.cons()) < tol:
            flag = False
        if iter > MaxIter:
            flag = False
            raise(f"Maximum Number of Iterations reached")

    return x.cons()

def kepler_F(a: Union[float, DA], sigma: Union[float, DA], r1_norm: Union[float, DA], dM: Union[float, DA]) -> Union[float, DA]:
    """
    Kepler's equation F(dE) = dE + (sigma/sqrt(a)) * (1 - cos(dE) - (1- r1/a)*sin(dE))
    :param a: Semi-major axis
    :param sigma: Specific angular momentum
    :param dM: Change in mean anomaly
    :return: Value of Kepler's equation at E
    """

    Max_Variable = dM.getMaxVariables()
    dE = dM.cons() + DA(Max_Variable)

    for _ in range(1000):
        F = (dE + (sigma / op.sqrt(a)) * (1 - op.cos(dE)) - (1 - r1_norm / a) * op.sin(dE)) - dM
        if abs(F.cons()) < 10e-12:  # Convergence criterion
            break
        dF = F.deriv(Max_Variable)
        # dF = 1.0 + (sigma / op.sqrt(a)) * (op.sin(dE) - (1 - r1_norm / a) * op.cos(dE)) -> analytically derived dF/dM
        if dF.cons() == 0:
            raise ValueError("Derivative became zero during iteration")
        dE -= F.cons() / dF.cons()
        print(f"{F.cons() / dF.cons()}")
        print(f"dE:\n{dE}\n")
        
        
    return dE.cons()

def Kepler_DA(r1: Union[array, NDArray], v1: Union[array, NDArray], dt: float, mu: float = 3.2712440018e11,order: int = None):
    """
        High-order Kepler solver
          r0, v0 : 3-component ndarray of float *or* DA (if DA components are carried through calculation)
          dt     : propagation time [same units as μ]
          mu     : GM of central body
          order  : DA truncation order
        Returns (r, v) at t0+dt
    """
    if not DA.initialized:
        DA.init(order, r1.getMaxVariables()+1) # Initialize DA with order and number of variables
        if order is None:
            order = DA.getMaxOrder(r1[0]) + 1 if isinstance(r1[0], DA) else 0
    
    #Prepare variables
    sigma = r1.dot(v1) / op.sqrt(mu)

    energy = v1.dot(v1) / 2 - (mu / op.vnorm(r1))  #Specific orbital energy
    
    a = - mu / (2 * energy)  #Semi-major axis
    if a.cons() <= 0.0:
        raise ValueError("ERROR: hyperbolic or parabolic semi major axis")

    n = op.sqrt(mu/a**3)
    #Obtain time of passage through perigee 
    print(dt)
    dM = n*dt
    # -------------------------------------
    #Calculate the Nominal change in Eccentric anomaly - via Newton iteration
    # -------------------------------------
    a_nom = a.cons() if isinstance(a, DA) else a
    dM_nom = dM.cons() if isinstance(dM, DA) else dM
    sigma_nom = sigma.cons() if isinstance(sigma, DA) else sigma
    r1_nom = r1.cons() if isinstance(r1, DA) else r1
    dt_nom = dt if isinstance(dt, DA) else dt


    dE_nom = kepler_F(a, sigma, op.vnorm(r1), dM)
    # -------------------------------------
    # raise dE to DA variable and create taylor map
    # -------------------------------------

    dE_da = dE_nom + DA(dM.getMaxVariables())  # Create DA variable for dE
    p_da = array([r1, v1])          # Shape (2,3)
    
    def F_da(dE_da, p_da):
        #Unpack p array
        #Generate Variables for Function

        r1 = p_da[0,:]
        v1 = p_da[1,:]

        sigma = r1.dot(v1) / op.sqrt(mu)
        energy = v1.dot(v1) / 2 - mu / op.vnorm(r1)  #Specific orbital energy
        a = - mu / (2 * energy)  #Semi-major axis
        dM = op.sqrt(mu / a**3) * dt    # change in mean anomaly

        #Construct F.
        return dE_da + (sigma / op.sqrt(a)) * (1 - op.cos(dE_da)) - (1 - op.vnorm(r1) / a) * op.sin(dE_da) - dM
    
    #problem is here :(
    dE_da = Implicit_solver_DA(dE_da, p_da, F_da)

    #Recompute variables with final dE_da Map 


    # -------------------------------------
    # Calcualte the final position and velocity vectors through lagrange coefficients
    # -------------------------------------

    r2, v2 = lagrange_coefficients(a, dE_da, r1, v1, sigma, mu)

    return r2, v2