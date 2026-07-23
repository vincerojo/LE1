import numpy as np


class DebyeSolid:

    def __init__(self,
        N_atom,
        freq_min,
        freq_max,
        N_freq,
        hbar,
        k_B,
        T_min,
        T_max,
        N_T,
        dT
    ):
        self.N_atom = N_atom
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_Debye = freq_max
        self.N_freq = N_freq
        self.hbar = hbar
        self.k_B = k_B
        self.T_min = T_min
        self.T_min = T_max
        self.N_T = N_T
        self.dT = dT

        self.N_osc = 3*N_atom
        self.T_spectrum = np.linspace(T_min,T_max,num=N_T,endpoint=True)

        self.freq_grid, self.domega = DebyeSolid.frequency_grid(self.freq_min,self.freq_max,self.N_freq)
        
        self.heat_capacities = DebyeSolid.heat_capacities(self.freq_grid,self.freq_Debye,self.domega,self.T_spectrum,self.dT,self.N_atom,self.hbar,self.k_B)


    @staticmethod
    def frequency_grid(
        freq_min: float,
        freq_max: float,
        N: int,
    ):
        # Constructing the grid
        freq_grid = np.linspace(start=freq_min,stop=freq_max,num=N,endpoint=True)

        # Taking domega
        domega = freq_grid[1] - freq_grid[0]

        return [freq_grid,domega]

    @staticmethod
    def density_of_states(
        freq,
        freq_Debye,
        N_atom,
    ):
        
        # Calculating g(freq), the density of states
        return 9*N_atom*(freq**2)/(freq_Debye**3)

    @staticmethod
    def mode_average_energy(
        freq,
        T,
        hbar,
        k_B
    ):
        
        # Pre-calculating "beta*hbar*freq"
        bho = hbar*freq/(k_B*T)

        # Calculating the average energy per mode of vibration
        return hbar*freq*( 0.5 + 1/(np.exp(bho)-1) )

    @staticmethod
    def internal_energy(
        freq_spectrum,
        freq_Debye,
        domega,
        T,
        N_atom,
        hbar,
        k_B
    ):
        U = 0
        for freq_i in freq_spectrum:
            U = U + DebyeSolid.density_of_states(freq_i,freq_Debye,N_atom)*DebyeSolid.mode_average_energy(freq_i,T,hbar,k_B)*domega
        return U
    

    def heat_capacities(
        freq_spectrum,
        freq_Debye,
        domega,
        T_spectrum,
        dT,
        N_atom,
        hbar,
        k_B
    ):
        dU = np.empty(T_spectrum.size)
        for i in range(T_spectrum.size):
            dU[i] = DebyeSolid.internal_energy(freq_spectrum,freq_Debye,domega,T_spectrum[i]+dT,N_atom,hbar,k_B) - DebyeSolid.internal_energy(freq_spectrum,freq_Debye,domega,T_spectrum[i]-dT,N_atom,hbar,k_B)
        
        return np.divide(dU,2*dT)