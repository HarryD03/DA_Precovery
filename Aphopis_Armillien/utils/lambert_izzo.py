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
        findxy()                     - Incomplete
        x2tof()                      - Incomplete
        x2tof2()                     - Incomplete
        hypergeometricF()            - Complete
        householder_iter_DA_nom      - Incomplete
        householder_iter_DA_Map      - Incomplete
"""

def lambert_izzo(r1: Union[array, NDArray], r2: Union[array,NDArray], dt: float, mu: float, multi_revs, cw=False) -> array:
    """
    Dario Izzo Solution to Lamberts problem (Revisiting Lambert's Problem) as used in Pirovano's Paper.
    Algorithm 1
    DA compatiable 
    :params :r1 Initial Position (Vector: 3x1, [i j k]')
    :params :r2 Final Position (Vector: 3x1, [i j k]')
    :params :dt Time of flight between two points 
    :params :mu Standard Gravitional Parameter

    :returns :v1 Velocity of point 1 
    :return :v2 Velocity of point 2
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
    if isinstance(r1_radial_dir, NDArray):
        print("ENTERING NUMPY BRANCH: Position 1 is entered as a NumPy array type")
        assert isinstance(r2_radial_dir, NDArray), f"Position 2 must have both NumPy array types"
        
        h_dir  = np.cross(r1_radial_dir, r2_radial_dir)
        
        if ((r1[0]*r2[1]) - (r1[1]*r2[0])) < 0:
            L = -L
            r1_tangent_dir = np.cross(r1_radial_dir, h_dir)
            r2_tangent_dir = np.cross(r2_radial_dir,h_dir)
        else:
            r1_tangent_dir = np.cross(h_dir,r1_radial_dir)
            r2_tangent_dir = np.cross(h_dir, h_dir)
    
    #DaceyPy Branch
    if isinstance(r1_radial_dir, array):
        print("ENTERING DACEYPY BRANCH: Position 1 is entered as a DA array type")
        assert isinstance(r2_radial_dir, NDArray), f"Position 2 must have both DaceyPy array types"
        h_dir = r1_radial_dir.cross(r2_radial_dir)
        assert h_dir[2] != 0, f"The Angular Momentum Vector has no z component, impossible to define clock or counterclockwise direction"
        
        if ((r1[0]*r2[1]) - (r1[1]*r2[0])) < 0:         #Transfer angle is larger than 180  degrees as seen from above the z axis
            L = -L
            r1_tangent_dir = r1_radial_dir.cross(h_dir)
            r2_tangent_dir = r2_radial_dir.cross(h_dir)
        else:
            r1_tangent_dir = h_dir.cross(r1_radial_dir)
            r2_tangent_dir = h_dir.cross(h_dir)
    
    if cw: #Retrograde motion
        r1_tangent_dir = -r1_tangent_dir
        r2_tangent_dir = -r2_tangent_dir

    T = op.sqrt((2*mu)/(s**3))*dt                   #Non-dimensional Time Parameter defined through Izzo

    x_list, y_list = findxy(L, T, multi_revs)                   #Function to find Lambert variables x and y 
    assert len(x_list) == len(y_list), f"The x and y variables should be the same in number"

    #For eaxh found x and y, construct the terminal velocities
    gamma = op.sqrt(mu*s / 2.0)
    rho = r1_norm - r2_norm
    sigma = op.sqrt(1 - rho**2)

    #Initialise array of solutions
    v1 = np.empty((3,len(x_list)), dtype=object)       #shape (3, length of x solutions). rows correspond to components, columns x solution 
    v2 = np.empty((3,len(x_list)), dtype=object)


    for i in range(0, len(x_list)):
        vr1 = gamma * ((L * y_list[i] - x_list[i]) - rho * (L * y_list[i] + x_list[i])) / r1_norm
        vr2 = -gamma * ((L * y_list[i] - x_list[i]) + rho * (L * y_list[i] + x_list[i])) / r2_norm
        vt = gamma * sigma * (y_list[i] + L * x_list[i])
        vt1 = vt / r1_norm
        vt2 = vt / r2_norm
        for j in range(0, 2):
            v1[i][j] = vr1 * r1_radial_dir[j] + vt1 * r1_tangent_dir[j]
            v2[i][j] = vr2 * r2_radial_dir[j] + vt2 * r2_tangent_dir[j]
    
    return v1, v2

def findxy(L: Union[DA, float], T: Union[DA,float], multi_revs) -> Union[array,NDArray]:
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
        assert T < 0, "Standard time parameter must be negative"

    if isinstance(T, DA):
        print("Algorithm 2 ENTERING DA BRANCH")
        assert isinstance(L, DA), "Must be DA"
        assert abs(L.cons()) < 1, "Lambda Variable must be less than 1"
        assert T.cons() < 0, "Standard time parameter must be negative"

    M_max = op.floor(T/np.pi)                   #Number of revolutions in which a solution exisits
    T00 = op.acos(L) + (L*op.sqrt(1-L**2))      
    T0 = T00 + M_max  * np.pi
    T1 = 2/3 * (1 - L**3)         #Standard Time parameter if parabolic orbit (Max Energy transfer)
    
    #Detect if the multi-revoultion time
    if T.cons() < (T0.cons()) and M_max > 0:
        #Halley iterations from x = 0, T = T0 and find T_min

        iter = 1
        T_min = T0
        flag = True 
        x0 = 0.0 + DA(MaxVar)

        while flag:

            #Get Derivatives for iteration
            dTdx = T_min.deriv(MaxVar).cons()
            dTTdxx = T_min.deriv(MaxVar).deriv(MaxVar).cons()
            dTTTdxxx = T_min.deriv(MaxVar).deriv(MaxVar).deriv(MaxVar).cons()

            assert dTdx != 0.0, f"Derivative cannot be equal to 0.0"
            
            #Hally Iterator for NOMINAL X
            x = x0 - (dTdx * dTTdxx) / (dTTdxx**2 - dTdx * dTTTdxxx / 2.0)
            err = abs(x.cons() - x0.cons())
            
            if err < 1e-12 or iter > 12:
                flag = False
                break
            
            #Get T(x)
            T_min = x2tof(x, M_max, L)

        if T_min > T:
            M_max -= 1
    

    M_max = np.min(M_max, multi_revs)

    ## Find all solutions in x,y
    x_list = np.empty(int(2*M_max + 1), dtype=object)
    
    #x seed generation based on zeta-T plane relationships transfered to x-T plane
    if T.cons() >= T0.cons():
        x0 = (T0/T)**(2/3) - 1
    elif T.cons() < T1.cons():
        x0 = (5/2) * (T1*(T1-T))/(T*(1-L**5)) + 1
    elif T1.cons() < T.cons() and T.cons() < T0.cons():
        x0 = (T0/T)**(op.log2(T1/T0)) - 1
    else:
        raise ValueError("Parameterised Time of Flight Parameter does not fall into any of the possible solutiions")
    
    x0 = x0.cons()      #x0 must be defined as an independant DA variable within the iteraiton loop 

    #HouseHolder iterator function
    def f(x):
        """
        Function to obtain f(x) = T(x) - T* for Householder Iteration scheme
        """
        return x2tof(x, M_max, L) - T
    
    #Single Revolution Case
    MaxVar = T.getMaxVariables()
    x_nom = householder_iter_DA_nom(x0, f, MaxVar, tol=1e-9, MaxIter=1000)       #Obtain Nomial Solution of x within DA framework
    x_list[0] = householder_iter_DA_Map(x_nom, f, MaxVar, tol=1e-9, MaxIter=1000)                                                #Obtain full DA expansion of x. x = x_nom + map(f(r1,r2))

    #Multi Revolution Case
    for i in range(1, M_max + 1): 
        #Compute x0l and find x0r, yr
        tmp = ((i * np.pi + np.pi) / (8.0 * T)) ** (2.0 / 3.0)
        x_list[2 * i - 1] = (tmp - 1) / (tmp + 1)
        #left Householder iter
        x_nom = householder_iter_DA_nom(x_list[2*i-1], f, MaxVar, tol=1e-9, MaxIter=1000)       #Obtain Nomial Solution of x within DA framework
        x_list[2*i-1] =  householder_iter_DA_Map(x_nom,)
        #right Householder iter
        tmp = ((8.0 * T)/ (i * np.pi)) ** (2/3)
        x_list[2*i] = (tmp-1)/(tmp+1)
        x_nom = householder_iter_DA_nom(x_list[2*i], f, MaxVar, tol=1e-9, MaxIter=1000)       #Obtain Nomial Solution of x within DA framework
        x_list[2*i] =  householder_iter_DA_Map(x_nom)
    
    #Generate Y list
    y_list = op.sqrt(1.0 - L**2 + L**2*x_list**2)

    return x_list, y_list

def x2tof(x: DA, M: float, L: Union[float, DA]):
    """
    Generate T(x) function is General form for a given x
    :params x: Lancaster/Battin Variablef for Lamberts problem. MUST BE DA
    :params tof: Parameterised Time of Flight Defined it 'Revisiting Lamberts problem'.
    :params M: Revolution Number
    :returns T: New Parameterised time of flight value based on x. T = T(x) in DA
    """

    battin = 0.01
    lagrange = 0.2
    dist = np.abs(x.cons() - 1)

    if dist < lagrange and dist > battin: #Use Lagrange Tof Expression for Low energy transfers (x ->)
        T = x2tof2(x, M, L)
        return T
    
    K = L**2
    E = x**2 - 1
    rho = abs(E.cons())             #abs for DA objects seems to fail. Work around is E.cons(). DA(x) will be neglected in root finding iteration
    y = op.sqrt(1 + K * E)

    if dist < battin:                      #If x -> 1 use Battin
        eta = y - L * x
        S1 = 0.5 * (1.0 - L - x * eta)
        Q = hypergeometricF(S1, 1e-11)      
        Q = 4/3 * Q
        T = (eta**3 * Q + 4*L*eta) / 2 + M*np.pi/(rho*(3/2))
        return T
    
    else:                                  #Lancaster formulation performs best in general case
        g = x * y - L*E
        if E.cons() < 0:
            l = op.acos(g)
            d = M * np.pi + l
        else:
            f = op.sqrt(rho) * (y - L * x)
            d = op.log(f + g)
        
        T = (x - L * y - d/op.sqrt(rho)) /E
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
        if L < 0.0:
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

def householder_iter_DA_nom(x0, f: callable, MaxVar, tol=1e-12, MaxIter=1000):
    """
    Complete Household Iteration Scheme for root finding of implicit equation
    DA Reliant
    General form based on DA.deriv(Max_variable) of passed function
    
    :params x0: Initial Guess for implicit solution
    :params f(x): Function for Root. E.G. find x for f(x)= T(x) - T* = 0.
    :return x: NOMIAL root of function. NO DA MAPPING
    """

    x = x0 + DA(MaxVar)     #The solution variable will always be the largest
    flag = True
    iter = 1

    while flag:
        F = f(x)
        dFdx = F.deriv(MaxVar)
        assert dFdx.cons() != 0, "Derivative cannot be zero"
        
        dFFdxx = dFdx.deriv(MaxVar)
        dFFFdxxx = dFFdxx.deriv(MaxVar)

        num = dFdx.cons()**2 - F.cons()*dFFdxx.cons()/2
        denom = (dFdx.cons() * (dFdx.cons()**2 - F.cons()*dFFdxx.cons())) + (dFFFdxxx.cons() * F.cons()**2)/6

        x -= F.cons()*(num/denom)
        print(f"Iteration Number: {iter}\n")
        if F.cons() < tol or iter > MaxIter:
            flag = False
            print(f"Final Iteration: {iter}\n")
            raise("Function has converged or Max iterations reached")

    return x

def householder_iter_DA_Map(x_nom: DA, MaxVar: int, f: callable, tol=1e-12, MaxIter=1000):
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
    x = x_nom + DA(x_Var)

    while iter <= (DA.getMaxOrder()):
        #Collect Derivatives (can refactor this and put it in both schemes)
        F = f(x)
        dFdx = F.deriv(x_Var)
        dFFdxx = F.deriv(x_Var).deriv(x_Var)
        dFFFdxxx = F.deriv(x_Var).deriv(x_Var)

        assert dFdx.cons() != 0, "Cannot divide by 0 in iteration scheme"

        #Remove DA(x) as an independant DA variable once automatic derivation has taken place (When refractoring above block; include subsquent block and 'nom' and 'DA' option to define if .cons() or .plug() used
        F = F.plug(x_Var, 0)
        dFdx = dFdx.plug(x_Var, 0)
        dFFdxx = dFFdxx.plug(x_Var, 0)
        dFFFdxxx = dFFFdxxx.plug(x_Var,0)

        #Householder iteration Equation
        num = dFdx**2 - F * dFFdxx/2
        denom = (dFdx * (dFdx**2 - F*dFFdxx)) + (dFFFdxxx* F**2)/6

        x -= F*(num/denom)
        
        print(f"Iteration Number: {iter}\n")    #For de-bugging purposes
        iter *= 2
    
    x = x.plug(x_Var, 0)        #Assert the DA(x) are eliminated in final expansion
    return x