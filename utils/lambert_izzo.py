import numpy as np
from typing import Union
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray

"""
    This File Comtains Everything required for Izzo's solution to Lambert's Problem, defined in "revisiting Lamberts
    Problem" - 2017.

    Validation Status: 
        todo via v1.cons()/v2.cons() evaluation and comparison of v1, v2 = poliastro.iod.izzo.lambert() - Incomplete
    
    Unit Test Status:
        lambert_izzo() before findxy - Incomplete
        findxy()                     - complete
        x2tof()                      - complete
        x2tof2()                     - complete
        hypergeometricF()            - Complete
        householder_iter_DA_nom      - complete
        householder_iter_DA_Map      - complete
"""

def lambert_izzo(r1: Union[array, NDArray], r2: Union[array,NDArray], dt: float, mu: float, multi_revs, prograde=False) -> array:
    """
    Dario Izzo Solution to Lamberts problem (Revisiting Lambert's Problem) as used in Pirovano's Paper.
    Algorithm 1
    DA compatiable 
    :param :r1 Initial Position (Vector: 3x1, [i j k]')
    :param :r2 Final Position (Vector: 3x1, [i j k]')
    :param :dt Time of flight between two points in seconds
    :param :mu Standard Gravitional Parameter

    :returns :velocities. An array of [(v1, v2), (v1,v2)] depending on the number of revolutions
    """

    #Assertion statements for Inputs
    assert dt > 0, f"Time of Flight must be position"
    assert mu > 0, f"Standard Gravitional Parameter must be positive"
    assert len(r1) == 3, f"Require Cartesian coordinates in form [i, j, k]'"
    assert len(r2) == 3, f"Require Cartesian coordinates in form [i, j, k]'"

    #Pre-compute Lambert Variables
    c_vec = r2 - r1
    c_norm = op.vnorm(c_vec)
    r1_norm = op.vnorm(r1)
    r2_norm = op.vnorm(r2)
    s = 0.5*(r1_norm + r2_norm + c_norm)
    L_squared = 1 - c_norm/s                 #Lambda term. lambda = 1 => chord = 0 => points 1 and 2 next to eachother
    L = op.sqrt(L_squared)

    #Develop Unit Vectors
    r1_radial_dir = r1 / r1_norm       #Unit vector of posiiton 1 
    r2_radial_dir = r2 / r2_norm       #Unit vector of position 2

    #Numpy Branch 
    if isinstance(r1_radial_dir[0], float):
        print("ENTERING NUMPY BRANCH: Position 1 is entered as a NumPy array type")
        assert isinstance(r2_radial_dir, np.ndarray), f"Position 2 must have both NumPy array types"
        
        h_dir  = np.cross(r1_radial_dir, r2_radial_dir)
        h_dir = h_dir / op.vnorm(h_dir)  #Angular Momentum Vector
        
        if ((r1[0]*r2[1]) - (r1[1]*r2[0])) < 0:
            L = -L
            r1_tangent_dir = np.cross(r1_radial_dir, h_dir)
            r2_tangent_dir = np.cross(r2_radial_dir,h_dir)
        else:
            r1_tangent_dir = np.cross(h_dir,r1_radial_dir)
            r2_tangent_dir = np.cross(h_dir, r2_radial_dir)
    
    #DaceyPy Branch
    if isinstance(r1_radial_dir[0], DA):
        print("ENTERING DACEYPY BRANCH: Position 1 is entered as a DA array type")
        assert isinstance(r2_radial_dir[0], DA), f"Position 2 must have both DaceyPy array types"
        h_dir = r1_radial_dir.cross(r2_radial_dir)
        h_dir = h_dir / op.vnorm(h_dir)
        assert h_dir[2].cons() != 0, f"The Angular Momentum Vector has no z component, impossible to define clock or counterclockwise direction"
        
        if h_dir[2].cons() < 0:         #Transfer angle is larger than 180  degrees as seen from above the z axis
            L = -L  
            r1_tangent_dir = r1_radial_dir.cross(h_dir)
            r2_tangent_dir = r2_radial_dir.cross(h_dir)
        else:
            r1_tangent_dir = h_dir.cross(r1_radial_dir)
            r2_tangent_dir = h_dir.cross(r2_radial_dir)

    L, r1_tangent_dir, r2_tangent_dir = (L, r1_tangent_dir, r2_tangent_dir) if prograde else (-L, -r1_tangent_dir, -r2_tangent_dir)

    T = op.sqrt((2*mu)/(s**3))*dt                   #Non-dimensional Time Parameter defined through Izzo

    xy = findxy(L, T, multi_revs)                   #Function to find Lambert variables x and y 
    if multi_revs == 0:
        assert len(xy) == 1, f"The x and y variables should be the same in number"
    if multi_revs > 0:
        assert len(xy) == 2

    #For eaxh found x and y, construct the terminal velocities
    gamma = op.sqrt(mu*s / 2.0)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = op.sqrt(1 - rho**2)

    # Get Velocities
    velocities = []
    for idx, sublist in enumerate(xy): # iterate across embedded list xy = [[sublist], [sublist]]
       
     #Calculate v1 and v2 for different x, y. sublist = [x,y]
            x = sublist[0]
            y = sublist[1]

            Vr1 = gamma * ((L * y - x) - rho * (L * y + x)) / r1_norm
            Vr2 = -gamma * ((L * y -x) + rho * (L * y + x)) / r2_norm
            Vt1 = gamma * sigma * (y + L * x) / r1_norm
            Vt2 = gamma * sigma * (y + L * x) / r2_norm

            v1 = Vr1 * (r1 / r1_norm) + Vt1 * r1_tangent_dir
            v2 = Vr2 * (r2 / r2_norm) + Vt2 * r2_tangent_dir

            v1 = v1.reshape((3,1))
            v2 = v2.reshape((3,1))
            v1v2 = np.hstack([v1,v2])
            if isinstance(x, DA):
                v1v2 = array(v1v2)
            velocities.append(v1v2)        #Append and reshape to libray standard 

    return velocities   #velcoity[0] is (v1,v2) of first direction, velcoity[1] is (v1,v2) of second solution


def findxy(L: Union[DA, float], T: Union[DA,float], M) -> Union[array,NDArray]:
    """
    Algorithm 2 Defined by Dario Izzo solution to Lamberts problem. "Revisiting Lamberts problem".
    DA compatible

    :params :L Lambda Parameter in Lambert (1 - chord/semi-perimeter) [scalar]
    :params :T Non dimensional time parameter defined in line 65 of lambert_PyKep [scalar]
    :returns :x_list List of x solutions. [vector: Nx1] N x's depend on number of revolutions in transfer   
    :returns :y_list List of y solutions.  [vector: Nx1] N x's depend on number of revolutions in transfer 
    """
    if isinstance(T, float):
        print("Algorithm 2 ENTERING NUMPY BRANCH")
        assert isinstance(L, float), "Must be Numpy"
        assert abs(L) < 1, "Lambda Variable must be less than 1"
        assert T > 0, "Standard time parameter must be posiitve"        #Mistake on original paper
        T_cons = T

    if isinstance(T, DA):
        print("Algorithm 2 ENTERING DA BRANCH")
        assert isinstance(L, DA), "Must be DA"
        assert abs(L.cons()) < 1, "Lambda Variable must be less than 1"
        assert T.cons() > 0, "Standard time parameter must be positive"
        T_cons = T.cons()  #Convert to float for further calculations

    M_max = int(np.floor(T_cons/np.pi))                   #Maximum number of revolutions for time period
    T00 = op.acos(L) + (L*op.sqrt(1-L**2))
    T0 = T00 + M_max  * np.pi
    T1 = 2/3 * (1 - L**3)         #Standard Time parameter if parabolic orbit (Max Energy transfer)

    if isinstance(T00, DA):
        T00_cons = T00.cons()
        T0_cons = T0.cons()
    if isinstance(T00, float):
        T00_cons = T00
        T0_cons = T0
    
    T1 = 2/3 * (1 - L**3)         #Standard Time parameter if parabolic orbit (Max Energy transfer)

    #Require Time in terms of full revolutions
    if T_cons < (T00_cons) + M_max* np.pi and M_max > 0:
       _, T_min = compute_T_min(L, M_max, 20, 1e-9) # Min Time to complete a revolution

       if isinstance(T_min, DA):
           T_min_cons = T_min.cons()  #Convert to float for further calculations
       if isinstance(T_min, float):
           T_min_cons = T_min

       if T_cons < T_min_cons:        #if Time is less than min time to complete a full revolution it is not a full revolution so subtract 1
            M_max -= 1


    if M > M_max: #if the number of stated revolutions is more than max full revolutions within the time period
        raise ValueError(f"No feasible solution, try M <={M_max}")
    
    xy = []
    for x_0 in initial_guess(T, L, M):          #This only generates the single revolution case (M == 0) x0 or stated revolution case (M) [x0l,x0r]
        # FLOAT Branch
        if isinstance(T, float):
            x = householderfloat(x_0, T, L, M, tol=1e-9, maxiter=20)
            y = _compute_y(x, L)
            tmp = [x, y]
            xy.append(tmp)

        # DA BRANCH
        if isinstance(T, DA):
            def f(x,p):
                """
                Function to obtain f(x) = T(x) - T* for Householder Iteration scheme
                """
                T, L, M = p
                return x2tof(x, M, L) - T

            if isinstance(T, DA) and isinstance(L, DA):
                p0 = [T.cons(), L.cons(), M]  #Parameters for Householder Iteration Scheme
            else:
                p0 = [T, L, M]  #Parameters for Householder Iteration Scheme
            
            p = [T, L, M]  #Parameters for Householder Iteration Scheme in DA
            x_nom = householder_iter_DA_nom(x_0, p0, f, tol=1e-12, MaxIter=20)
            x_DA = householder_iter_DA_Map(x_nom, p, DA.getMaxVariables(), f, tol=1e-12, MaxIter=20)
            y_DA = op.sqrt(1 + L**2*(x_DA**2-1))

            tmp = [x_DA, y_DA]
            xy.append(tmp)
    
    return xy


def x2tof(x: Union[DA,float], M: float, L: Union[float, DA]):
    """
    Generate T(x) function is General form for a given x
    :params x: Lancaster/Battin Variablef for Lamberts problem. MUST BE DA
    :params tof: Parameterised Time of Flight Defined it 'Revisiting Lamberts problem'.
    :params M: Revolution Number
    :returns T: New Parameterised time of flight value based on x. T = T(x) in DA
    """

    battin = 0.01
    lagrange = 0.2
    
    K = L**2
    E = x**2 - 1
    y = op.sqrt(1 + K * E)
    
    if isinstance(x, DA):
        dist = np.abs(x.cons() - 1)
        rho = np.abs(E.cons())
        xx = E.cons()
    
    if isinstance(x, float):
        dist = np.abs(x - 1)
        rho = abs(E)
        xx  = E

    if dist < lagrange and dist > battin: #Use Lagrange Tof Expression for Low energy transfers (x ->)
        T = x2tof2(x, M, L)
        return T
    
    if dist < battin:                      #If x -> 1 use Battin
        eta = y - L * x
        S1 = 0.5 * (1.0 - L - x * eta)
        Q = hypergeometricF(S1, 1e-11)      
        Q = 4/3 * Q
        T = (eta**3 * Q + 4*L*eta) / 2 + M*np.pi/(rho*(3/2))
        return T
    
    else:                                  #Lancaster formulation performs best in general case
        g = (x*y) - (L*E)
        if xx < 0:                         #xx is a variable E.cons() dependant on the DA or float nature on E 
            l = op.acos(g)
            d = M * np.pi + l
        else:
            f = op.sqrt(rho) * (y - L * x)
            log_arg = f + g
            
            # Safety check for DA objects
            if isinstance(log_arg, DA):
                if log_arg.cons() <= 0:
                    raise ValueError(f"Logarithm argument is non-positive: {log_arg.cons()}")
            elif log_arg <= 0:
                raise ValueError(f"Logarithm argument is non-positive: {log_arg}")
            
        T = (x - L * y - d/op.sqrt(rho)) / E
        return T
    return

def x2tof2(x: DA, M, L):
    """
    Obtain the Lagrange Equation for Lamberts problem. Use if large transfer angles
    :params :x Lancaster/Battin's variable for Lamberts Problem [DA].
    :params :M Nnumber of revolutions
    :params :L Lambda defined in "Revisiting Lamberts Problem". L = 1 - chord / semi-perimeter
    :params :T Parametrised Time of Flight as a function of x. T(x) [DA type]
    """

    a = 1 / (1 - x**2)      #Expresssion for SMA
    if a.cons() > 0:
        alpha = 2.0 * op.acos(x)
        beta = 2.0 * (op.asin(op.sqrt(L**2/a)))
        if L.cons() < 0.0:         #Chord > semi-perimeter. 
            beta = -beta
        T = a**(3/2)*((alpha - op.sin(alpha)) - (beta - op.sin(beta)) + 2*M*np.pi) / 2
    else:   #For the Hyperbolic / parabolic case
        alpha = 2.0 * op.cosh(x)
        beta = 2.0 * op.sinh(op.sqrt(-L * L / a))
        if L.cons() < 0.0:
            beta = -beta
        T = (-a* op.sqrt(-a) * ((op.sinh(alpha) - alpha) - (op.sinh(beta) - beta))/2)

    return T

def hypergeometricF(S1, tol=1e-11):
    """
    Completes the Hypergeometric Function of S (A type of Infinite Series) till convergences
    achieved within successive steps. Look at Battin for more information
    DA compatible 

    :params S1: Guass' variable for Lambert's Problem 
    :params tol: Desired Convergence tolerance
    """
    #Initialise variables
    Cj = 1.0
    Sj = 1.0
    err = 1.0
    j = 0           #iterations = j + 1
    while err > tol:
        Cj1 = Cj *(3.0 + j) * (1.0 + j) / (2.5 + j) * S1 / (j+1)
        Sj1 = Sj + Cj1
        err = abs(Cj1)

        #Update Variables for next iteration
        Sj = Sj1
        Cj = Cj1
        j = j + 1

    return Sj 

def householder_iter_DA_nom(x0, p0, f: callable, tol=1e-12, MaxIter=1000):
    """
    Complete Household Iteration Scheme for root finding of implicit equation
    DA Reliant
    General form based on DA.deriv(Max_variable) of passed function
    
    :params x0: Initial Guess for implicit solution
    :params f(x): Function for Root. E.G. find x for f(x)= T(x) - T* = 0.
    :return x: NOMIAL root of function. NO DA MAPPING
    """
    MaxVar = DA.getMaxVariables()  #Get the maximum number of variables in the DA system
    x = x0 + DA(MaxVar)     #The dummy variable will always be the largest
    flag = True
    iter = 1
    DA.pushTO(4)
    while flag:
        F = f(x,p0)
        dFdx = F.deriv(MaxVar)
        assert dFdx.cons() != 0, "Derivative cannot be zero"
        
        dFFdxx = dFdx.deriv(MaxVar)
        dFFFdxxx = dFFdxx.deriv(MaxVar)

        num = dFdx.cons()**2 - (F.cons()*dFFdxx.cons()/2)
        denom = (dFdx.cons() * (dFdx.cons()**2 - F.cons()*dFFdxx.cons()) + (dFFFdxxx.cons() * F.cons()**2 / 6))

        x -= F.cons()*(num/denom)

        # CRITICAL: Constrain x to elliptical region for elliptical orbits
        if x.cons() >= 0.999999:  # Prevent x from exceeding ~1 (hyperbolic)
            raise ValueError(f"Convergence failed: x_new = {x.cons()} exceeds 0.999999, indicating a hyperbolic orbit.")
        elif x.cons() <= -0.999999:  # Prevent extreme negative values
            raise ValueError(f"Convergence failed: x_new = {x.cons()} exceeds -0.999999, indicating a hyperbolic orbit.")

        print(f"Iteration Number: {iter}\n")
        if abs(F.cons()) < tol or iter > MaxIter:
            flag = False
            print(f"Final Iteration: {iter}\n")
        iter += 1
           
    DA.popTO()
    return x.cons()

def householder_iter_DA_Map(x_nom: DA, p: DA, MaxVar: int, f: callable, tol=1e-12, MaxIter=30):
    """
    Implicit Equation solver for Household Iteration Algorithm. Given a Nomial (constant) value of x, x_nom, express the DA part in terms of the independant
    DA variables. i.e. [x] = x_nom + DA_MAP(del_p) for f(x;p) = x**2 - p. This gives a locally explicit equation (x) for various values of p around the nomial value p  
    (used to calculate x_nom). See DA_IOD for more
    
    :params x_nom: Nomial solution for DA variable [x] = x_nom + DA(x). Must be obtained via householder_iter_DA_nom.
    :params MaxVar: The Dependant DA variable, DA(x), will always be the last variable. Technically doesn't exisit but required for the automatic differentiation
    :params f(x): Callable function that defines you root finding equaiton.
    :params tol: Convergence Tolerance
    :params MaxIter: Maximum Iterations
    :returns [x]: System of Taylor Series Expansion Equations, defined by independant DA variables, around nomial solution x_nom.
    """
    iter = 1
    x_Var = MaxVar
    xp = x_nom + DA(x_Var)       #temp variable for Automatic Derivative
    flag=True
    while flag:
        #Collect Derivatives (can refactor this and put it in both schemes)
        F = f(xp,p)
    
        dFdx = F.deriv(x_Var)
        dFFdxx = F.deriv(x_Var).deriv(x_Var)
        dFFFdxxx = F.deriv(x_Var).deriv(x_Var).deriv(x_Var)

       
        #Remove DA(x) as an independant DA variable once automatic derivation has taken place (When refractoring above block; include subsquent block and 'nom' and 'DA' option to define if .cons() or .plug() used
        F = F.plug(x_Var, 0)
        dFdx = dFdx.plug(x_Var, 0)
        dFFdxx = dFFdxx.plug(x_Var, 0)
        dFFFdxxx = dFFFdxxx.plug(x_Var,0)

        #Householder iteration Equation
        num = dFdx**2 - F * dFFdxx/2
        denom = (dFdx * (dFdx**2 - F*dFFdxx)) + (dFFFdxxx* F**2)/6

        assert denom != 0, "Cannot divide by 0!"
        x = xp - F*(num/denom)
        
        print(f"Iteration Number: {iter}\n")    #For de-bugging purposes

        if iter > MaxIter:
            print("Convergence Achieved")
            flag=False
            break
        iter += 1
        xp = x

    
    x = x.plug(x_Var, 0)        #Assert the DA(x) are eliminated in final expansion
    return x

def compute_T_min(L, M, maxiter=20, tol=1e-9):
    """
    Compute Minimum T for revolution
    """
    if L == 1:
        x_Tmin = 0.0
        T_min = x2tof(x_Tmin, M, L)
    else:
        if M == 0: 
            x_Tmin = np.inf
            T_min = 0.0
        else:
            # Set x_i > 0 to avoid Lambda = -1
            x_i = 0.1
            
            def f(x):
                return x2tof(x, M, L)
            
            x_Tmin = halley(x_i, f, tol, maxiter)
            T_min = x2tof(x_Tmin, M, L)

    return [x_Tmin, T_min]

def halley(x0, f, tol=1e-9, maxiter=20):
    """
        Find the Minimium flight time using the Halley method
    """
    MaxVar = DA.getMaxVariables()
    x0 = x0 + DA(MaxVar)
    for i in range(maxiter):
        # Generate Derivatives
        #Get Derivatives for iteration
        Ti = f(x0)
        dTdx = Ti.deriv(MaxVar).cons()
        dTTdxx = Ti.deriv(MaxVar).deriv(MaxVar).cons()
        dTTTdxxx = Ti.deriv(MaxVar).deriv(MaxVar).deriv(MaxVar).cons()
        if dTTdxx == 0:
            raise RuntimeError("Deriative was zero")
        
        #Hally Iterator for NOMINAL X
        x = x0 - (2*dTdx * dTTdxx) / (2 * dTTdxx**2 - dTdx * dTTTdxxx)
        
        err = abs(x.cons() - x0.cons())

        if err < tol:
            return x.cons()
        x0 = x

    raise RuntimeError("Failed to converge. Perhaps increase Iteration count")

def initial_guess(T, L, M):
    """
        Generate Initial guess for Single and multi revolution case
    """
    if isinstance(T, DA):
        T_cons = T.cons()
    else:
        T_cons = T

    if M == 0:  #single revolution case
        T0 = op.acos(L) + (L*op.sqrt(1-L**2)) + M*np.pi
        T1 = 2/3 * (1 - L**3)         #Standard Time parameter if parabolic orbit (Max Energy transfer)

        if isinstance(T0, DA):
            T0_cons = T0.cons()
            T1_cons = T1.cons()
        else:
            T0_cons = T0
            T1_cons = T1

        if T_cons >= T0_cons:
            x0 = (T0/T)**(2/3) - 1
        elif T_cons < T1_cons:
            x0 = (5/2) * (T1*(T1-T))/(T*(1-L**5)) + 1
        elif T1_cons < T_cons < T0_cons:
            # Original (wrong Version) = x0 = (T0/T)**(op.log2(T1/T0)) - 1
            x0 = op.exp(op.log(2) * op.log(T/ T0) / op.log(T1/T0)) - 1
        else:
            raise ValueError("Parameterised Time of Flight Parameter does not fall into any of the possible solutiions")

        if x0 is isinstance(x0, DA):
            x0_cons = x0.cons()
        if isinstance(x0, float):
            x0_cons = x0

        return [x0_cons]

    else: #Multiple revolution case Maximum Revolution
        tmp = ((M * np.pi + np.pi) / (8.0 * T)) ** (2.0 / 3.0)
        x_0l = (tmp - 1) / (tmp + 1 )
        tmp = ((8.0 * T)/ (M * np.pi)) ** (2/3)
        x_0r = (tmp - 1) / (tmp + 1)

        return [x_0l.cons(), x_0r.cons()]


    
def householderfloat(p0, T0, ll, M, tol, maxiter):
    """Find a zero of time of flight equation using the Householder method.

    Notes
    -----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.

    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder = _tof_equation_p(p0, y, T, ll)
        fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)

        # Householder step (quartic)
        p = p0 - fval * (
            (fder**2 - fval * fder2 / 2)
            / (fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6)
        )

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")

def _compute_y(x, ll):
    """Computes y."""
    return np.sqrt(1 - ll**2 * (1 - x**2))

def _tof_equation_p(x, y, T, ll):
    # TODO: What about derivatives when x approaches 1?
    return (3 * T * x - 2 + 2 * ll**3 * x / y) / (1 - x**2)

def _tof_equation_p2(x, y, T, dT, ll):
    return (3 * T + 5 * x * dT + 2 * (1 - ll**2) * ll**3 / y**3) / (
        1 - x**2
    )

def _tof_equation_p3(x, y, _, dT, ddT, ll):
    return (
        7 * x * ddT + 8 * dT - 6 * (1 - ll**2) * ll**5 * x / y**5
    ) / (1 - x**2)

def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computated y."""
    if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4 / 3 * hyp2f1b(S_1)
        T_ = (eta**3 * Q + 4 * ll * eta) * 0.5
    else:
        psi = _compute_psi(x, y, ll)
        T_ = np.divide(
            np.divide(psi + M * np.pi, np.sqrt(np.abs(1 - x**2))) - x + ll * y,
            (1 - x**2),
        )

    return T_ - T0

def hyp2f1b(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin].

    .. todo::
        Add more information about this function

    Notes
    -----
    More information about hypergeometric function can be checked at
    https://en.wikipedia.org/wiki/Hypergeometric_function

    """
    if x >= 1.0:
        return np.inf
    else:
        res = 1.0
        term = 1.0
        ii = 0
        while True:
            term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
            res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1

def _compute_psi(x, y, ll):
    """Computes psi.

    "The auxiliary angle psi is computed using Eq.(17) by the appropriate
    inverse function"

    """
    if -1 <= x < 1:
        # Elliptic motion
        # Use arc cosine to avoid numerical errors
        return np.arccos(x * y + ll * (1 - x**2))
    elif x > 1:
        # Hyperbolic motion
        # The hyperbolic sine is bijective
        return np.arcsinh((y - x * ll) * np.sqrt(x**2 - 1))
    else:
        # Parabolic motion
        return 0.0