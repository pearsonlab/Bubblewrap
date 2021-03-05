import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay


def center_mass(points):
    # Compute center of mass of points, assuming equal masses here
    # points is a list of coords arrays [array((dim,))]*N
    # TODO: use average(..., weight=mass_array) in future;
    # can weight by e.g. similarity or flow vectors

    return np.mean(points, axis=0)


def check_bounded(points, x):
    # From https://stackoverflow.com/a/43564754.
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def dumb_bounding(points, ref, num=8):
    dist = np.linalg.norm(points - ref, axis=1)
    bounding = np.argsort(dist)[:num]
    return dist, bounding


def bounding(points, ref, num=8, lim=20):
    """Choose num nearest bounding points on the mesh (e.g. 4, assuming roughly square 2D grid)
    of ref given set of points, that is the new observation.

    If cannot bound `ref` after considering `lim` nearest points, returns `num`-th nearest points.

    Args:
        points (np.ndarray): (n × d) Array of all points in mesh.
        ref (np.ndarray): (n) Point of interest.
        num (int, optional): Number of points to send back. Defaults to 8.
        lim (int, optional): Number of nearest points to check for bounding before giving up. Defaults to 20.

    Returns:
        Distance matrix, `num` closest points.
    """
    dirs = points - ref
    zero = np.zeros(ref.shape)

    # TODO: Use some tree data structure, r-tree, etc.
    dist = np.linalg.norm(dirs, axis=1)
    closest = np.argsort(dist)

    if not check_bounded(dirs[closest[:num]], zero):
        # Add closest points one by one until enclosed.
        k = 1
        hull = Delaunay(dirs[closest[: num + k]])

        while (simp := hull.find_simplex(zero)) == -1:  # not bounded
            hull = Delaunay(dirs[closest[: num + k]])
            # hull.add_points(dirs[np.newaxis, closest[idx], :])  # TODO: Need to deal with non-uniqueness.
            k += 1
            if k > lim:  # Cannot bound, give up return num nearest point.
                bounding = closest[:num]
                break

        else:
            # Add simplex vertices, then next closest points.
            important = closest[hull.simplices[simp]]
            bounding = np.zeros(num, dtype=int)
            bounding[: important.size] = important
            k = 0
            for i in range(important.size, num):
                while closest[k] in important:
                    k += 1
                bounding[i] = closest[k]
                k += 1

    else:
        bounding = closest[:num]

    return dist, bounding
