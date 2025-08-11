import numpy as np
from daceypy import DA, array
import daceypy.op as op
from numpy.typing import NDArray



#It is assumed that v2+, v2- are known via lambert_izzo
#DeltaV = v2+ - v2- = DeltaV(drho1, drho2, drho3)

def Iterative_improvement(x, f, x0, tol, MaxIter):
    """
        Newton Iterative Improvement Algorithm derived from Pirovano's "INITIAL ORBIT DETERMINATIONBASEDONPROPAGATIONOFORBITSETSWITH
        DIFFERENTIALALGEBRA" and "High-order expansion of the solution of preliminary orbit
        determination problems"
       
        DA MUST BE INITIALISED PRIOR
        NOTE: p is assumed be to embedded into the function f in the form f(x;p)

        :params :x The variable to be iteratively improved
        :params :f The function to be iteratively improved. f(x;p)
        :params :x0 The initial guess
        :params :tol The convergence tolerance
        :params :MaxIter The maximum number of iterations

        :returns :x The converged solution in full Taylor Polynomial Expansion [x] = x + M(p)

    
    """

    #Nomial Newton Solver
    x_zeroth = newton_nomial_DA(x0, x, f, tol, MaxIter)

    #Implicit Equation Solver (Partial Inverson Method)
    x_DA_map = Implicit_solver_DA(x_zeroth, f)

    return x_DA_map

#Nomial Newton Solver Definition 
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


#Partial Inversion Method via Implicit Equation solver
def Implicit_solver_DA(x_da: Union[float,DA], f: callable):
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

def Nf(x, f, df):
    k = DA.getMaxVariables()        

    f = f.plug(k,0)
    df = df.plug(k,0)

    if np.allclose(df, 0):
        raise ZeroDivisionError("Derivative is zero in Newton iteration")
    x1 = x - f/df    
    return x1