import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sub_python_files import qho1d as qho1d, einstein_solid as es, debye as db, diatomic_chain as dc, drude as dr, sommerfield_model as sf

from sub_python_files import My_Funcs as MyF

# No of particles/oscillators
N_atom = 1

# Various Grid Parameters
x_max = (5e-11)
x_min = (-5e-11)
freq_min = 0.1 
freq_max = 2.422e14     # Debye frequency of diamond
T_min = 0.1
T_max = 2000
electric_field_min = 0.01
electric_field_max = 1000
relaxation_time_min = 10
relaxation_time_max = 500
E_min = 0
E_max = 1e-17
N = 1000        # No of position grid points (including endpoints)
N_freq = 500    # No of frequency grid points (including endpoints)
N_T = 60        # No of temperature grid points (including enpoints)
N_k = 50        # No of wavevector grid points (including endpoints)
N_E = 2000      # No of energy grid points (including endpoints)
electric_field_no_of_points = 60    # No of electric field grid points (including endpoints)
relaxation_time_no_of_points = 60   # No of relaxation time grid points (including endpoints)

# Central Finite Difference Approximation parameter
dT = 0.001

# Matplotlib Parameters
N_eigs = 20 # No of eigenvalues to be plotted
eig_choice = 10 # n of the QHO wavefunction to be plotted

# Miscellaneous Constants
hbar = 1.055e-34
m = (12e-3)/(6.02e23)       # Mass of One Carbon Atom
freq_Einstein = 1.728e14    # Einstein frequency of Diamond
k_B = 1.381e-23
e = 1.602e-19               # Charge of One (1) Electron

# Diatomic Linear Chain parameters
M1 = m
M2 = 2*m
kappa = 1 # arbitrary
a = .3e-9 # arbitrary

# Drude Model parameters
carrier_density = 1e30
m_effective = 9.109e-31     # Mass of One (1) Electron
tau = 1                     # arbitrary

# Sommerfeld Model parameters
m_electron = 9.109e-31









# Creating an Einstein Solid
einstein_solid = es.EinsteinSolid(N_atom,N,hbar,freq_Einstein,k_B,T_min,T_max,N_T,dT)

# Getting the Einstein heat capacity analytically
C_Einstein_analytical = es.EinsteinSolid.einstein_heat_capacity(einstein_solid.T_spectrum,freq_Einstein,N_atom,hbar,k_B)










# Creating one 1D quantum harmonic oscillator (qho)
numerical_qho = qho1d.NumericalQHO1D(x_min,x_max,N,hbar,m,freq_Einstein)

# Extracting the eigenvalues and eigenvectors of the 1D qho, numerically and analytically
E_qho_numerical = numerical_qho.eigenvalues
E_qho_analytical = qho1d.AnalyticalQHO1D.energies(N-2,hbar,freq_Einstein)
psi_qho_numerical = numerical_qho.eigenvectors
psi_qho_analytical = np.empty(shape=[3,numerical_qho.x_operator.size])
for i in range(3):
    psi_qho_analytical[i,:] = MyF.analytical_qho_wavefunctions(numerical_qho.x_operator,i,m,freq_Einstein,hbar)

# Calculating the average oscillator energy
average_oscillator_energies = np.empty(einstein_solid.T_spectrum.size)
for i in range(einstein_solid.T_spectrum.size):
    average_oscillator_energies[i] = es.EinsteinSolid.average_oscillator_energy(einstein_solid.osc_spectrum,k_B,einstein_solid.T_spectrum[i])

# Getting the Einstein heat capacity of one oscillator numerically and analytically (Notice the 1/3)
one_oscillator = es.EinsteinSolid((1/3),N,hbar,freq_Einstein,k_B,T_min,T_max,N_T,dT)
C_oscillator_numerical = one_oscillator.heat_capacities
C_oscillator_analytical = es.EinsteinSolid.einstein_heat_capacity(einstein_solid.T_spectrum,freq_Einstein,(1/3),hbar,k_B)












# Creating a Debye Solid
debye_solid = db.DebyeSolid(N_atom,freq_min,freq_max,N_freq,hbar,k_B,T_min,T_max,N_T,dT)

# Calculating the Debye heat capacity analytically
C_Debye_analytical = MyF.analytical_debye_heat_capacities(debye_solid.T_spectrum,N_atom,freq_max,k_B,hbar)









# Calculating dispersion on the diatomic linear chain numerically
diatomic_chain = dc.DiatomicChain1D(M1=M1,M2=M2,kappa=kappa,a=a,N_k=N_k)








# Creating an electric field grid (Was thinking of doing this in the DrudeModel constructor, but I would have to pass additional arguments to that constructor.)
electric_field_grid = MyF.electric_field_grid(electric_field_min,electric_field_max,electric_field_no_of_points)

# Creating a relaxation time grid
relaxation_time_grid = MyF.relaxation_time_grid(relaxation_time_min,relaxation_time_max,relaxation_time_no_of_points)

# Creating a Drude metal
drude_model = dr.DrudeMetal(n=carrier_density,e=e,m=m_effective,tau=tau)

# Creating a 2nd Drude Metal with many relaxation time values
drude_model_2 = dr.DrudeMetal(n=carrier_density,e=e,m=m_effective,tau=relaxation_time_grid)

# Calculating current density
current_density = drude_model.current_density(electric_field_grid)

# Calculating resistivity
resistivity = drude_model_2.resistivity()

# Creating a Sommerfeld metal
sommerfeld_metal = sf.SommerfeldMetalCalculator.create_metal(
    electron_density=carrier_density,
    electron_mass=m_electron,
    hbar=hbar,
    k_B=k_B,
    energy_min=E_min,
    energy_max=E_max,
    num_energy_points=N_E)

# Calculating Sommerfeld heat capacity numerically and analytically
C_Sommerfeld_numerical = np.empty(debye_solid.T_spectrum.size)
for i in range(debye_solid.T_spectrum.size):
    C_Sommerfeld_numerical[i] = sf.SommerfeldMetalCalculator.heat_capacity(T=debye_solid.T_spectrum[i],dT=dT,metal=sommerfeld_metal)
C_Sommerfeld_analytical = sf.SommerfeldMetalCalculator.low_temperature_heat_capacity(T=debye_solid.T_spectrum,metal=sommerfeld_metal)












# Exporting the eigenergies into Excel files
df1 = pd.DataFrame(numerical_qho.eigenvalues[0:N_eigs])
df2 = pd.DataFrame(qho1d.AnalyticalQHO1D.energies(N-2,hbar,freq_Einstein)[0:N_eigs])
df1.to_excel("numerical_eigenenergies.xlsx",index=True)
df2.to_excel("analytical_eigenenergies.xlsx",index=True)


























# ------------------ PLOTTING EIGENENERGIES ------------------
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Analytic QHO eigenvalues
axs[0].plot(
    range(N_eigs), 
    qho1d.AnalyticalQHO1D.energies(N-2,hbar,freq_Einstein)[0:N_eigs], 
    marker='.',
    label=r'$E_\text{QHO,analytical}$')
# Numerical QHO eigenvalues
axs[0].plot(
    range(N_eigs), 
    numerical_qho.eigenvalues[0:N_eigs], 
    marker='o',fillstyle='none', 
    label=
        r'$E_\text{QHO,numerical}$' + f' using {N-2} internal points')
axs[0].set_title(r'$E_n \text{ vs } n$', fontsize=18)
axs[0].set_xlabel(r'Number (unitless), $1\leq n \leq$' + f'{N_eigs}')
axs[0].set_ylabel(r'Energy (J)')
axs[0].legend()

# ------------------ PLOTTING SOME QHO WAVEFUNCTION ------------------
# Plotting |psi_{eig_choice}(x)|^2 vs x
axs[1].plot(
    numerical_qho.x_operator, 
    qho1d.AnalyticalQHO1D.probability_density(numerical_qho.eigenvectors[:,eig_choice]),
    linestyle='-',
    label=r'$|\psi$' + f'$_{{{eig_choice}}}$' + r'$(x)|^2$')
axs[1].set_title(r'$P$' + f'$_{{{eig_choice}}}$' + r'$(x)$ vs $x$', fontsize=20)
axs[1].set_xlabel(r'x (m)')
axs[1].set_ylabel(r'$|\psi|^2 ( )$')
axs[1].legend()

# ----------- PLOTTING EINSTEIN AND DEBYE HEAT CAPACITIES ------------------
# Plotting Einstein analytical
axs[2].plot(
    einstein_solid.T_spectrum, 
    C_Einstein_analytical,
    marker='.',
    label=r'$C_\text{Einstein,analytical}$')
# Plotting Einstein numerical
axs[2].plot(
    einstein_solid.T_spectrum, 
    einstein_solid.heat_capacities,
    marker='o',fillstyle='none', 
    label=r'$C_\text{Einstein,numerical}$')
# Plotting Debye analytical
axs[2].plot(
    debye_solid.T_spectrum, 
    C_Debye_analytical,
    marker='.',fillstyle='none', 
    label=r'$C_\text{Debye,analytical}$')
# Plotting Debye numerical
axs[2].plot(
    debye_solid.T_spectrum, 
    debye_solid.heat_capacities,
    marker='o',fillstyle='none', 
    label=r'$C_\text{Debye,numerical}$')
axs[2].set_title(r'$C$ vs $T$', fontsize=20)
axs[2].set_xlabel(r'T (K)  ' + f'{T_min}K' + '$\leq T \leq$' + f'{T_max}K')
axs[2].set_ylabel(r'$C$ (J/K)')
axs[2].legend()




















# ------------------ PLOTTING DISPERSION BRANCHES ------------------
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Analytical Dispersion, Lower
axs[0].plot(
    diatomic_chain.grid, 
    diatomic_chain.dispersion_analytical[0], 
    marker='.',
    label=r'$\omega_\text{lower,analytical}$')
# Numerical Dispersion, Lower
axs[0].plot(
    diatomic_chain.grid, 
    diatomic_chain.dispersion_numerical[0], 
    marker='o',fillstyle='none', 
    label=
        r'$\omega_\text{lower,numerical}$')
# Analytical Dispersion, Upper
axs[0].plot(
    diatomic_chain.grid, 
    diatomic_chain.dispersion_analytical[1], 
    marker='.',
    label=r'$\omega_\text{upper,analytical}$')
# Numerical Dispersion, Upper
axs[0].plot(
    diatomic_chain.grid, 
    diatomic_chain.dispersion_numerical[1], 
    marker='o',fillstyle='none', 
    label=
        r'$\omega_\text{upper,numerical}$')
axs[0].set_title(r'$\omega(k)$ vs $k$', fontsize=20)
axs[0].set_xlabel(r'k (m$^{-1}$)')
axs[0].set_ylabel(r'$\omega(k)$  (s$^{-1}$)')
axs[0].legend()
# ------------------ PLOTTING HARMONIC OSCILLATOR POTENTIAL  ------------------
axs[1].plot(
    numerical_qho.x_operator, 
    numerical_qho.potential_values,
    marker='None',
    label=r'$V(x)$')
axs[1].set_title(r'$V(x)$ vs $x$', fontsize=20)
axs[1].set_xlabel(r'x (m)')
axs[1].set_ylabel(r'$V(x)$ (J/C)')
axs[1].legend()
# ------------------ PLOTTING SOMMERFELD HEAT CAPACITIES ------------------
# Plotting Sommerfeld analytical
axs[2].plot(
    debye_solid.T_spectrum, 
    C_Sommerfeld_analytical,
    marker='.',
    label=r'$C_\text{Sommerfeld,analytical}$')
# Plotting Sommerfeld numerical
axs[2].plot(
    debye_solid.T_spectrum, 
    C_Sommerfeld_numerical,
    marker='o',fillstyle='none', 
    label=r'$C_\text{Sommerfeld,numerical}$')
axs[2].set_title(r'$C_\text{Sommerfeld}$ vs $T$', fontsize=20)
axs[2].set_xlabel(r'T (K)  ' + f'{T_min}K' + '$\leq T \leq$' + f'{T_max}K')
axs[2].set_ylabel(r'$C$ (J/K)')
axs[2].legend()











# ------------------ PLOTTING 1ST THREE P(x)'s  ------------------
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Plotting P_0(x)
axs[0].plot(
    numerical_qho.x_operator, 
    qho1d.AnalyticalQHO1D.probability_density(numerical_qho.eigenvectors[:,0]),
    linestyle='-',
    label=r'$|\psi$' + f'$_{{{0}}}$' + r'$(x)|^2$')
axs[0].set_title(r'$P_0(x) \text{ vs } x$', fontsize=20)
axs[0].set_xlabel(r'x (m)')
axs[0].set_ylabel(r'$|\psi|^2$ (m$^{-1}$)')
axs[0].legend()

# Plotting P_1(x)
axs[1].plot(
    numerical_qho.x_operator, 
    qho1d.AnalyticalQHO1D.probability_density(numerical_qho.eigenvectors[:,1]),
    linestyle='-',
    label=r'$|\psi$' + f'$_{{{1}}}$' + r'$(x)|^2$')
axs[1].set_title(r'$P_1(x) \text{ vs } x$', fontsize=20)
axs[1].set_xlabel(r'x (m)')
axs[1].set_ylabel(r'$|\psi|^2$ (m$^{-1}$)')
axs[1].legend()

# Plotting P_2(x)
axs[2].plot(
    numerical_qho.x_operator, 
    qho1d.AnalyticalQHO1D.probability_density(numerical_qho.eigenvectors[:,2]),
    linestyle='-',
    label=r'$|\psi$' + f'$_{{{2}}}$' + r'$(x)|^2$')
axs[2].set_title(r'$P_2(x) \text{ vs } x$', fontsize=20)
axs[2].set_xlabel(r'x (m)')
axs[2].set_ylabel(r'$|\psi|^2$ (m$^{-1}$)')
axs[2].legend()










# --- PLOTTING THE FIRST THREE WAVEFUNCTIONS (NOT PROBABILITY DENSITIES)  ----
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Numerical psi_0
axs[0].plot(
    numerical_qho.x_operator, 
    numerical_qho.eigenvectors[:,0],
    linestyle='-',
    label=r'$\psi$' + f'$_{{{0}}}$' + r'$_\text{,numerical}$' + r'$(x)$')
# Analytical psi_0
axs[0].plot(
    numerical_qho.x_operator, 
    psi_qho_analytical[0,:],
    linestyle='dashed',
    label=r'$\psi$' + f'$_{{{0}}}$' + r'$_\text{,analytical}$' + r'$(x)$')
axs[0].set_title(r'$\psi_0(x) \text{ vs } x$', fontsize=20)
axs[0].set_xlabel(r'x (m)')
axs[0].set_ylabel(r'$\psi(x)$  (m$^{-1/2}$)')
axs[0].legend()

# Numerical psi_1
axs[1].plot(
    numerical_qho.x_operator, 
    numerical_qho.eigenvectors[:,1],
    linestyle='-',
    label=r'$\psi$' + f'$_{{{1}}}$' + r'$_\text{,numerical}$' + r'$(x)$')
# Analytical psi_1
axs[1].plot(
    numerical_qho.x_operator, 
    psi_qho_analytical[1,:],
    linestyle='dashed',
    label=r'$\psi$' + f'$_{{{1}}}$' + r'$_\text{,analytical}$' + r'$(x)$')
axs[1].set_title(r'$\psi_1(x) \text{ vs } x$', fontsize=20)
axs[1].set_xlabel(r'x (m)')
axs[1].set_ylabel(r'$\psi(x)$  (m$^{-1/2}$)')
axs[1].legend()

# Numerical psi_2
axs[2].plot(
    numerical_qho.x_operator, 
    numerical_qho.eigenvectors[:,2],
    linestyle='-',
    label=r'$\psi$' + f'$_{{{2}}}$' + r'$_\text{,numerical}$' + r'$(x)$')
# Analytical psi_2
axs[2].plot(
    numerical_qho.x_operator, 
    psi_qho_analytical[2,:],
    linestyle='dashed',
    label=r'$\psi$' + f'$_{{{2}}}$' + r'$_\text{,analytical}$' + r'$(x)$')
axs[2].set_title(r'$\psi_2(x) \text{ vs } x$', fontsize=20)
axs[2].set_xlabel(r'x (m)')
axs[2].set_ylabel(r'$\psi(x)$  (m$^{-1/2}$)')
axs[2].legend()











# ------ PLOTTING EINSTEIN AND DEBYE HEAT CAPACITIES SEPARATELY ----------
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Plotting Einstein analytical
axs[0].plot(
    einstein_solid.T_spectrum, 
    C_Einstein_analytical,
    marker='.',
    label=r'$C_\text{Einstein,analytical}$')
# Plotting Einstein numerical
axs[0].plot(
    einstein_solid.T_spectrum, 
    einstein_solid.heat_capacities,
    marker='o',fillstyle='none', 
    label=r'$C_\text{Einstein,numerical}$')
axs[0].set_title(r'$C_\text{Einstein}$ vs $T$', fontsize=20)
axs[0].set_xlabel(r'T (K)  ' + f'{T_min}K' + '$\leq T \leq$' + f'{T_max}K')
axs[0].set_ylabel(r'$C$ (J/K)')
axs[0].legend()
# Plotting Debye analytical
axs[1].plot(
    debye_solid.T_spectrum, 
    C_Debye_analytical,
    marker='.',fillstyle='none', 
    label=r'$C_\text{Debye,analytical}$')
# Plotting Debye numerical
axs[1].plot(
    debye_solid.T_spectrum, 
    debye_solid.heat_capacities,
    marker='o',fillstyle='none', 
    label=r'$C_\text{Debye,numerical}$')
axs[1].set_title(r'$C_\text{Debye}$ vs $T$', fontsize=20)
axs[1].set_xlabel(r'T (K)  ' + f'{T_min}K' + '$\leq T \leq$' + f'{T_max}K')
axs[1].set_ylabel(r'$C$ (J/K)')
axs[1].legend()







# ------- PLOTTING AVERAGE OSCILLATOR ENERGY AND HEAT CAPACITY --------------
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Plotting average oscillator energies
axs[0].plot(
    einstein_solid.T_spectrum, 
    average_oscillator_energies,
    marker='none',fillstyle='none', 
    label=r'$E_\text{oscillator}$')
axs[0].set_title(r'$E_\text{oscillator}$ vs $T$', fontsize=20)
axs[0].set_xlabel(r'T (K)  ' + f'{T_min}K' + '$\leq T \leq$' + f'{T_max}K')
axs[0].set_ylabel(r'$E_\text{oscillator}$ (J)')
axs[0].legend()
# Plotting oscillator heat capacity
axs[1].plot(
    einstein_solid.T_spectrum, 
    C_oscillator_numerical,
    marker='none',fillstyle='none', 
    label=r'$C_\text{oscillator}$')
axs[1].set_title(r'$C_\text{oscillator}$ vs $T$', fontsize=20)
axs[1].set_xlabel(r'T (K)  ' + f'{T_min}K' + '$\leq T \leq$' + f'{T_max}K')
axs[1].set_ylabel(r'$C_\text{oscillator}$ (J)')
axs[1].legend()















# --------------- PLOTTING CURRENT DENSITY AND RESISTIVITY ------------------
fig, axs = plt.subplots(ncols=3, nrows=1,
                        layout="constrained")
# Plotting current density
axs[0].plot(
    electric_field_grid, 
    current_density,
    marker='none',fillstyle='none', 
    label=r'Current Density')
axs[0].set_title(r'Absolute Current Density vs Electric Field Strength', fontsize=12)
axs[0].set_xlabel(r'$E$ (J/C)')
axs[0].set_ylabel(r'$J$ (A/m$^2$)')
axs[0].legend()
# Plotting resistivity
axs[1].plot(
    relaxation_time_grid, 
    resistivity,
    marker='none',fillstyle='none', 
    label=r'Resistivity')
axs[1].set_title(r'Resistivity vs Relaxation Time', fontsize=12)
axs[1].set_xlabel(r'$\tau$ (s)')
axs[1].set_ylabel(r'$\rho$ ($\Omega\cdot$m)')
axs[1].legend()





















plt.show(block=True) # block=True lets the computer know to continue after showing the graph

plt.close()