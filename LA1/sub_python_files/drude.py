import numpy as np

class DrudeMetal:

    def __init__(self, n,e,m,tau) -> None:
        self.n = n
        self.e = e
        self.m = m
        self.tau = tau

    def drift_velocity(self,E):
        return -(self.e*self.tau/self.m)*E
    
    def current_density(self,E):
        return -(self.n*self.e)*self.drift_velocity(E)
    
    def conductivity(self):
        return self.n*(self.e**2)*self.tau/self.m
    
    def resistivity(self):
        return 1/self.conductivity()
    
    def mobility(self):
        return self.e*self.tau/self.m