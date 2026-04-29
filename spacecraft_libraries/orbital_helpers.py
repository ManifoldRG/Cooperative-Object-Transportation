#NOTE TO SELF: This does not need to be converted to using the data_Structures class, its fine on its own

import numpy as np

def kepler_equation_solver(M, e, tolerance = 1e-12, max_iter = 100):
    """
    Solves Kepler equation for mean anomaly to return eccentric anomaly. Solution based on equation 1
    :param max_iter: maximum number of iterations
    :param tolerance: maximum tolerance
    :param M: mean anomaly
    :param e: orbital eccentricity
    :return E: eccentric anomaly
    """
    E = M #TODO: IMPROVE ON THIS <THE OPT CAN BE DONE BETTER>
    for _ in range(max_iter):
        E_next = E - (E-e*np.sin(E) - M) / (1-e*np.cos(E))
        if abs(E_next - E) < tolerance:
            return E_next
        E = E_next
    raise RuntimeError("No convergence for Eccentric anomaly in Kepler equation")



def true_anomaly_f(E, e):
    """
    Returns true anomaly f based on Eccentric Anomaly based on equation 2
    :param E: eccentric anomaly
    :param e: Eccentricity
    :return: True anomaly based on equation
    """
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

def f_dot(n, e, f):
    """
    Returns the derivative of true anomaly f
    :param n: mean motion = sqrt(mu/a^3)
    :param e: orbital eccentricity
    :param f: true anomaly
    :return: f_dot
    """
    return n * (1 + e * np.cos(f))**2 / (1 - e**2)**(3/2)

def f_double_dot(n, e, f):
    """
    Returns the second derivative of true anomaly f
    :param n:
    :param e: orbital eccentricity
    :param f: true anomaly
    :return: f_ddot
    """
    return -2 * n**2 * e * np.sin(f) * (1 + e * np.cos(f))**3 / (1 - e**2)**3

def orbital_params(mu, a, e, t, t0=0, f0=0):
    """
    Returns F, fdot, fddot for reference frame calculations
    :param mu: Gravitational Parameter
    :param a: Semi-major axis
    :param e: orbital eccentricity
    :param t: time
    :param t0: initial time
    :param f0: initial value
    :return: [f, fdot, fddot]
    """
    # Calculate the mean motion
    n = np.sqrt(mu / a**3)

    # Calculate the mean anomaly at the current time
    M = n * (t - t0)

    # Solve Kepler's equation for the eccentric anomaly E
    E = kepler_equation_solver(M, e)

    # Calculate the true anomaly f
    f = true_anomaly_f(E, e)

    # Calculate f_dot
    fdot = f_dot(n, e, f)

    # Calculate f_double_dot
    fddot = f_double_dot(n, e, f)

    return f, fdot, fddot


def chief_radius(a, e, f):
    """Instantaneous chief-orbit radius R(f) = a(1-e^2) / (1 + e cos f)."""
    return a * (1.0 - e * e) / (1.0 + e * np.cos(f))


def th_psi_matrix(mu, a, e, t, t0=0):
    """Tschauner-Hempel time-varying state-transition matrix Psi(t).

    State order [r(3), v(3)]^T. Mirrors ``cot_sdp.orbital.psi_matrix`` so the
    sibling repo's translational dynamics are accurate for eccentric chief
    orbits, not just the circular CW limit.

    Returns a 6x6 numpy array.
    """
    f, fdot, fddot = orbital_params(mu, a, e, t, t0=t0)
    R = chief_radius(a, e, f)
    mu_over_R3 = mu / (R ** 3)
    return np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [fdot ** 2 + 2.0 * mu_over_R3, fddot, 0.0, 0.0, 2.0 * fdot, 0.0],
        [-fddot, fdot ** 2 - mu_over_R3, 0.0, -2.0 * fdot, 0.0, 0.0],
        [0.0, 0.0, -mu_over_R3, 0.0, 0.0, 0.0],
    ])
