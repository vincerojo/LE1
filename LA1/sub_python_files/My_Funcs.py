import numpy as np
import math

def analytical_qho_wavefunctions(
    x,
    n,
    m,
    freq_Einstein,
    hbar,       
):
    # Coefficients of the Hermitian polynomial series
    coeff = np.zeros(shape=n+1)
    coeff[n] = 1

    # Pre-calculating xi
    xi = np.multiply(np.sqrt(m*freq_Einstein/hbar),x)

    # Calculating the first three qho wavefunctions analytically
    psi_n_qho_analytical_base = np.multiply((m*freq_Einstein/(np.pi*hbar))**(1/4),np.exp(np.multiply(-1/2,np.pow(xi,2))))

    psi_n_qho_analytical = np.multiply(psi_n_qho_analytical_base,np.multiply(((2**n)*math.factorial(n))**(-1/2),np.polynomial.hermite.hermval(xi,coeff)))

    return psi_n_qho_analytical


def analytical_debye_heat_capacities(
    T,
    N,
    freq_Debye,
    k_B,
    hbar,       
):
    return N*k_B*((12*np.pi**4)/5)*(((k_B*T)**3)/((hbar*freq_Debye)**3))


def electric_field_grid(
    electric_field_min,
    electric_field_max,
    N,
):
    return np.linspace(electric_field_min,electric_field_max,N,endpoint=True)

def relaxation_time_grid(
    relaxation_time_min,
    relaxation_time_max,
    N,
):
    return np.linspace(relaxation_time_min,relaxation_time_max,N,endpoint=True)