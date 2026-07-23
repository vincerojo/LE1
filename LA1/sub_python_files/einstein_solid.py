from sub_python_files import qho1d as qho1d
import numpy as np

class EinsteinSolid:

    def __init__(self, N_atom, n_max, hbar, freq, k_B,T_min,T_max,N_T,dT) -> None:
        self.N_atom = N_atom
        self.n_max = n_max
        self.hbar = hbar
        self.freq = freq
        self.k_B = k_B
        self.T_min = T_min
        self.T_max = T_max
        self.N_T = N_T
        self.dT = dT
        self.N_osc = 3*N_atom
        self.osc_spectrum = qho1d.AnalyticalQHO1D.energies(n_max,hbar,freq)
        self.T_spectrum = np.linspace(T_min,T_max,num=N_T,endpoint=True)


        self.heat_capacities = EinsteinSolid.heat_capacity(self.osc_spectrum,self.T_spectrum,self.dT,self.N_osc,self.k_B)
    
    @staticmethod
    def partition_function(
        energy_spectrum,
        k_B,
        T):
        Z = 0

        # Precalculating "beta"
        beta = 1/(k_B*T)
        for energy in energy_spectrum:
            Z = np.add(Z,np.exp(np.multiply(-beta,energy)))
    
        return Z

    @staticmethod
    def probabilities(
        energy_spectrum,
        k_B,
        T):
        
        Z = EinsteinSolid.partition_function(energy_spectrum,k_B,T)
        beta = 1/(k_B*T)

        probability_list = np.divide(np.exp(np.multiply(-beta,energy_spectrum)),Z)

        return probability_list

    @staticmethod
    def average_oscillator_energy(energy_spectrum,k_B,T):

        return np.sum(np.multiply(energy_spectrum,EinsteinSolid.probabilities(energy_spectrum,k_B,T)))
    
    @staticmethod
    def internal_energy(
        energy_spectrum,
        N_osc,
        k_B,
        T):

        average_energy_per_oscillator = EinsteinSolid.average_oscillator_energy(energy_spectrum,k_B,T)

        return N_osc*average_energy_per_oscillator

    @staticmethod
    def heat_capacity(
        energy_spectrum,
        T_spectrum,
        dT,
        N_osc,
        k_B,
    ):
    
        dU = np.empty(T_spectrum.size)
        for i in range(T_spectrum.size):
            dU[i] = EinsteinSolid.internal_energy(energy_spectrum,N_osc,k_B,T_spectrum[i]+dT) - EinsteinSolid.internal_energy(energy_spectrum,N_osc,k_B,T_spectrum[i]-dT)

        return np.divide(dU,2*dT)
    
    @staticmethod
    def einstein_heat_capacity(
    T,
    freq_Einstein,
    N,
    hbar,
    k_B,
    ):
        """
        Return the Einstein heat capacity for a solid with N atoms.

        T may be a float or NumPy array.
        """
    
        # Why doesn't this function accept an hbar value when it needs an hbar value?
        # hbar = 1.055e-34

        # Calculating beta*hbar*freq in advance
        bho = np.multiply(np.multiply(np.pow(np.multiply(k_B,T),-1),hbar),freq_Einstein)

        # Calculating the Einstein heat capacity at each given T
        return np.multiply(np.multiply(np.multiply(np.multiply(np.multiply(np.exp(bho),np.pow(np.subtract(np.exp(bho),1),-2)),np.pow(bho,2)),k_B),N),3)