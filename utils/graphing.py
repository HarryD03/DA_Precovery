import numpy as np

def gen_grid3D(threesigma_error,Ns):
    """
        Generate the boundary of the Orbital Set
        
        :param threesigma_error: The 3 sigma measurement precision
        :param Ns: The number of steps for the xgrid,ygrid and zgrid. 
    """
    # Define grids for each axis
    xgrid = np.linspace(-1, 1, Ns)
    ygrid = np.linspace(-1, 1, Ns)
    zgrid = np.linspace(-1, 1, Ns)

    # Uncertainties in each direction
    xb = threesigma_error
    yb = threesigma_error
    zb = threesigma_error  # example value for z uncertainty

# Each face is a (Ns, Ns, 3) array, then reshape to (Ns*Ns, 3)
# x = -xb face
    face1 = np.stack((
        np.full((Ns, Ns), -xb),         # x = -xb
        yb * ygrid[None, :],            # y varies
        zb * zgrid[:, None]             # z varies
    ), axis=-1).reshape(-1, 3)

    # x = +xb face
    face2 = np.stack((
        np.full((Ns, Ns), xb),          # x = +xb
        yb * ygrid[None, :],
        zb * zgrid[:, None]
    ), axis=-1).reshape(-1, 3)

    # y = -yb face
    face3 = np.stack((
        xb * xgrid[None, :],
        np.full((Ns, Ns), -yb),         # y = -yb
        zb * zgrid[:, None]
    ), axis=-1).reshape(-1, 3)

    # y = +yb face
    face4 = np.stack((
        xb * xgrid[None, :],
        np.full((Ns, Ns), yb),          # y = +yb
        zb * zgrid[:, None]
    ), axis=-1).reshape(-1, 3)

    # z = -zb face
    face5 = np.stack((
        xb * xgrid[:, None],
        yb * ygrid[None, :],
        np.full((Ns, Ns), -zb)          # z = -zb
    ), axis=-1).reshape(-1, 3)

    # z = +zb face
    face6 = np.stack((
        xb * xgrid[:, None],
        yb * ygrid[None, :],
        np.full((Ns, Ns), zb)           # z = +zb
    ), axis=-1).reshape(-1, 3)

    # Concatenate all faces to get the perimeter (surface) points
    perimeter = np.concatenate((face1, face2, face3, face4, face5, face6), axis=0)
    perimeter_norm = np.zeros_like(perimeter)
    perimeter_norm[:, 0] = perimeter[:, 0] / xb
    perimeter_norm[:, 1] = perimeter[:, 1] / yb
    perimeter_norm[:, 2] = perimeter[:, 2] / zb

    return perimeter_norm

def gen_grid6D(threesigma_error, Ns):
    """
    Generate the boundary (perimeter) of a 6D uncertainty box.

    :param threesigma_error: The 3 sigma measurement precision (applied to all 6 axes)
    :param Ns: The number of steps for each axis
    :return: perimeter_norm, shape (number_of_perimeter_points, 6)
    """
    # Create grids for each axis
    grids = [np.linspace(-1, 1, Ns) for _ in range(6)]
    bounds = [threesigma_error] * 6

    # For each axis, create the two "faces" at -bound and +bound, varying all other axes
    faces = []
    for dim in range(6):
        # Prepare meshgrid for all axes except the fixed one
        mesh_axes = [grids[d] if d != dim else None for d in range(6)]
        mesh_shape = [Ns if d != dim else 1 for d in range(6)]
        for sign in [-1, 1]:
            # Create a mesh for all axes except the fixed one
            mesh = np.meshgrid(*[g if g is not None else [0] for g in mesh_axes], indexing='ij')
            face = np.zeros(mesh[0].shape + (6,))
            for d in range(6):
                if d == dim:
                    face[..., d] = sign * bounds[d]
                else:
                    face[..., d] = bounds[d] * mesh[d]
            faces.append(face.reshape(-1, 6))

    # Concatenate all faces
    perimeter = np.concatenate(faces, axis=0)
    perimeter_norm = np.zeros_like(perimeter)
    for d in range(6):
        perimeter_norm[:, d] = perimeter[:, d] / bounds[d]
    
    return perimeter_norm #(Number of perimeter points, 6) -> Each row is from -1,1. Each collum presents the compoent variation
                        # i.e. if extracted one row, each element is the normalsed z position of a 6D perimeter, with values ranging from [-1,1].



