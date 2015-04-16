import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


# ---------------------------------------
def hamiltonian(potential, xmin, xmax, N):
    """
    Return a matrix representing the the action of the operator
    H * u = (-0.5 * d^2/dx^2 + V(x))u

    Parameters:
    ==========
    potential : a function of one variable representing the potential energy
    xmin      : left endpoint of the interval
    xmax      : right endpoint of the interval
    N         : number of cells in which to discretize the interval

    Returns:
    =======
    H : a tridiagonal matrix
    """
    dx = (xmax - xmin) / (N + 1.0)
    x = np.linspace(xmin, xmax, N + 1)
    diagonal = np.ones(N + 1) / dx**2
    super_diag = -0.5 * np.ones(N) / dx**2
    for k in range(N):
        x = xmin + dx * k
        diagonal[k] += potential(x)

    H = diags([super_diag, diagonal, super_diag], [-1, 0, 1], (N + 1, N + 1))
    return H


# --------------------------------------------------------
def energy_levels(potential, xmin, xmax, N, k, tol = 1e-8):
    """
    Compute the lowest few energy levels of a Hamiltonian with the given
    potential

    Parameters:
    ==========
    potential, xmin, xmax, N : same as in `hamiltonian`
    k : number of energy levels desired
    tol : numerical precision of the eigensolve

    Returns:
    =======
    E : an array of length k containing the energies
    """
    H = hamiltonian(potential, xmin, xmax, N)
    E, _ = eigsh(H, k, which = 'SM', tol = tol)
    return E


# ------------------------
if __name__ == "__main__":

    def potential(x):
        return 0.5 * x**2

    E = energy_levels(potential, -10.0, 10.0, 100, 99, tol = 1e-12)

    plt.figure()
    plt.xlabel("k")
    plt.ylabel("energy level k")
    plt.title("Spectrum of Hamiltonian operator")
    plt.plot(E)
    plt.show()
