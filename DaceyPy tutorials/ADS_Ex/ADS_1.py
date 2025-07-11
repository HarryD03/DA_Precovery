

from daceypy import DA, array, ADS
import numpy as np

import matplotlib.pyplot as plt


def somb(x: array) -> DA:
    """
    Evaluate sombrero function
    """
    norm_x = x.vnorm()
    return norm_x.sin() / norm_x


def eFun(domain_0: ADS) -> ADS:
    """
    Evaluate transformation function for ADS
    """
    x0 = domain_0.box                               #extract the domain of the ADS variable
    xf = somb(x0)                                   #evaluate the function
    return ADS(domain_0.box, domain_0.nsplit, xf)   #Generate next ADS: domain, nsplit, manifold (solution of function)


def main() -> None:

    # initialize DACE for 10th-order computations in 2 variables
    DA.init(10, 2)
    Ns: int = 30

    # define ADS parameters
    tol = 1e-3          # splitting tolerance
    Nmax = 100          # Maximum sub domains?

    # define initial domain:
    xb = 5
    yb = 5
    x = array([2 + xb * DA(1), 3 + yb * DA(2)])

    # assemble border of domain:

    #We're only interested in the border of the domain - all other solutions are encorprated within it
    xgrid = np.linspace(-1, 1, Ns)      
    ygrid = np.linspace(-1, 1, Ns)

    lb = np.ones((Ns, 2))       #left boundary [x,y]
    lb[:, 0] = lb[:, 0] * (-xb)
    lb[:, 1] = yb * ygrid

    rb = np.ones((Ns,2))        #right boundary [x,y]
    rb[:, 0] = rb[:, 0] * (+xb)
    rb[:, 1] = yb * ygrid

    bb = np.ones((Ns, 2))       #bottom boundary [x,y]
    bb[:, 0] = xb * xgrid
    bb[:, 1] = bb[:, 1] * (-yb)

    tb = np.ones((Ns, 2))       #top boundary [x,y]
    tb[:, 0] = xb*xgrid
    tb[:, 1] = tb[:, 1] * (+yb)

    perimeter = np.concatenate((tb, rb, bb, lb))        #concatenate all border points into a perimeter
    perimeter_norm = np.concatenate((tb, rb, bb, lb))  
    perimeter_norm[:, 0] = perimeter[:, 0] / xb
    perimeter_norm[:, 1] = perimeter[:, 1] / yb

    # evaluate ground truth of sombrero function along border of domain:
    
    # Finding exact solution for the sombrero function along the border through direct evaluation of perimeter points in sombero() func
    XF = np.zeros((perimeter.shape[0]))
    for j in range(perimeter.shape[0]):
        x0 = x.copy()
        x0[0:2] += perimeter[j, :]
        XF[j] = somb(array(x0.cons())).cons()

    # evaluate expanded sombrero function along border of domain:
    z = somb(x)                                     #Generate nomial sombero solution + variations in the form of DA variables
    XF_single = np.zeros((perimeter.shape[0]))      #Pre-populate solution array
    for j in range(perimeter.shape[0]):         
        XF_single[j] = z.eval(perimeter_norm[j, :]) #Evaluate points on sombero function by evaluating the DA variable at each point along the perimeter

    # evaluate sombrero function with ADS:
    init_domain = ADS(x)                                #Initialise ADS domain is box=x, nsplit=1
    init_list = [init_domain]                           #Define the domain as a list type, so it can appended/ split into sub domains.
    final_list = ADS.eval(init_list, tol, Nmax, eFun)   #Evaluate the ADS domain with the eFun function. Obtain a list of subdomains and manifolds to evaluate over.

    
    final_manifold = np.zeros((perimeter_norm.shape[0], len(final_list)))
    final_domain = np.zeros((2, perimeter_norm.shape[0], len(final_list)))
   
   # for every point along the perimeter, evaluate the manifolds, constrained by the boxes, of the subdomains.

   #for every sub-domain
    for j in range(len(final_list)):
        #for every point along the perimeter
        for k in range(perimeter_norm.shape[0]):
            #Evaluate the perimeter point as a DA variable/variation in each manifold (XF) within a subdomain.
            final_manifold[k, j] = final_list[j].manifold.eval(perimeter_norm[k, :])

            #Evaluate the perimeter point as a DA variable/variation in subdomain (perimeter)
            final_domain[:, k, j] = final_list[j].box.eval(perimeter_norm[k, :])

    # plot figure
    ax2 = plt.axes(projection='3d')
    for j in range(final_domain.shape[2]):
        for k in range(4):
            ax2.plot3D(
                final_domain[0, Ns * k : Ns * (k + 1), j],
                final_domain[1, Ns * k : Ns * (k + 1), j],
                final_manifold[Ns * k : Ns * (k + 1), j],
                color='black')
            ax2.plot3D(
                perimeter[Ns * k : Ns * (k + 1), 0] + x[0].cons(),
                perimeter[Ns * k : Ns * (k + 1), 1] + x[1].cons(),
                XF[Ns * k : Ns * (k + 1)],
                color='green')
            ax2.plot3D(
                perimeter[Ns * k : Ns * (k + 1), 0] + x[0].cons(),
                perimeter[Ns * k : Ns * (k + 1), 1] + x[1].cons(),
                XF_single[Ns * k : Ns * (k + 1)],
                color='red')

    ax2.set_xlabel('x', fontsize = 16)
    ax2.set_ylabel('y', fontsize = 16)
    ax2.set_zlabel('z', fontsize = 16)
    ax2.legend(
        ("ADS", "Exact", "10th order"), shadow=True, loc="best", handlelength=1.5, fontsize=16)
    ax2.grid('minor', linestyle = ':')

    plt.show()

    print('End')


if __name__ == "__main__":
    main()