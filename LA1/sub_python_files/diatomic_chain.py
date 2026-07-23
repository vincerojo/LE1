import numpy as np

class DiatomicChain1D:

    def __init__(self, M1, M2, kappa, a, N_k) -> None:
        self.M1 = M1
        self.M2 = M2
        self.kappa = kappa
        self.a = a
        self.N_k = N_k
        
        self.grid = DiatomicChain1D.k_grid(self)

        self.dispersion_numerical = self.dispersion()
        self.dispersion_analytical = self.analytical_frequencies()
    
    @staticmethod
    def k_grid(chain):
        return np.linspace(
            start=-np.pi/chain.a,
            stop=np.pi/chain.a,
            num=chain.N_k,
            endpoint=True)
    
    @staticmethod
    def dynamical_matrix(k, chain):
        D = np.empty(shape=[2,2],dtype=np.complex64)
        D[0,0] = 2*chain.kappa/chain.M1
        D[0,1] = -(chain.kappa/np.sqrt(chain.M1*chain.M2))*(1+np.exp(np.multiply(-1j,k*chain.a)))
        D[1,0] = -(chain.kappa/np.sqrt(chain.M1*chain.M2))*(1+np.exp(np.multiply(1j,k*chain.a)))
        D[1,1] = 2*chain.kappa/chain.M2

        return D
    
    @staticmethod
    def frequencies_at_k(k,chain):
        D = DiatomicChain1D.dynamical_matrix(k,chain)
        eigenvals, eigenvecs = np.linalg.eig(D)

        # Note: eigenvalue = frequency^2
        # print(f"{eigenvals = }")
        freqs = np.sqrt(eigenvals) # Why in this order?
        freqs = np.real(freqs)
        freqs = np.sort(freqs)

        return freqs
    
    def dispersion(self):
        freq_lower = np.zeros(shape=self.grid.size)
        freq_upper = np.zeros(shape=self.grid.size)

        # Calculating freq_lower and freq_upper for all k
        for i in range(self.grid.size):
            D = DiatomicChain1D.dynamical_matrix(self.grid[i],self)
            freqs = self.frequencies_at_k(self.grid[i],self)
            freq_lower[i] = freqs[0]
            freq_upper[i] = freqs[1]

        return [freq_lower, freq_upper]
    
    def analytical_frequencies(self):
        freq_lower = np.zeros(shape=self.grid.size)
        freq_upper = np.zeros(shape=self.grid.size)

        # Precalculating (1/M1) + (1/M2)
        M12 = 1/self.M1 + 1/self.M2

        # Calculating freq_lower^2 and freq_upper^2 for all k
        for i in range(self.grid.size):
            D = DiatomicChain1D.dynamical_matrix(self.grid[i],self)
            freqs = DiatomicChain1D.frequencies_at_k(self.grid[i],self)
            freq_lower[i] = self.kappa*M12 - self.kappa*np.sqrt(M12**2 - (4/(self.M1*self.M2))*np.pow(np.sin(self.grid[i]*self.a/2),2))
            freq_upper[i] = self.kappa*M12 + self.kappa*np.sqrt(M12**2 - (4/(self.M1*self.M2))*np.pow(np.sin(self.grid[i]*self.a/2),2))
            
        freq_lower = np.sqrt(freq_lower)
        freq_upper = np.sqrt(freq_upper)

        return [freq_lower, freq_upper]
