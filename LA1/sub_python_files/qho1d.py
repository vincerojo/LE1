# qho1.py

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class AnalyticalQHO1D:
    """Analytical formulas for the one-dimensional quantum harmonic oscillator."""

    @staticmethod
    def potential(
        x: ArrayLike,
        m: float,
        omega: float,
    ) -> NDArray[np.float64]:
        """Return the harmonic oscillator potential V(x).

        V(x) = (1/2) m omega^2 x^2
        """
        # Obtaining potential values for all interior points of the grid
        return np.multiply(0.5*m*(omega**2),np.pow(x,2))

    @staticmethod
    def energy(
        n: int,
        hbar: float,
        omega: float,
    ) -> float:
        """Return the analytical energy eigenvalue E_n.

        E_n = hbar omega (n + 1/2)
        """
        return hbar*omega*(n+0.5)

    @staticmethod
    def energies(
        n_states: int,
        hbar: float,
        omega: float,
    ) -> NDArray[np.float64]:
        """Return the first n_states analytical energy eigenvalues.

        The returned array should contain

            E_0, E_1, ..., E_{n_states - 1}.
        """

        # Enumerating the 1st n states
        n = np.array(range(n_states))

        # Calculating the energy eigenvalue corresponding to each state in n
        E = np.array([AnalyticalQHO1D.energy(i,hbar,omega) for i in n])

        return E

    @staticmethod
    def probability_density(
        psi: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return the probability density |psi|^2."""

        return np.multiply(np.conjugate(psi),psi)

    @staticmethod
    def norm(
        psi: ArrayLike,
        dx: float,
    ) -> float:
        """Return the grid norm sqrt(sum_i |psi_i|^2 dx)."""
        
        return np.multiply(np.inner(psi,psi),dx)

    @staticmethod
    def normalize(
        psi: ArrayLike,
        dx: float,
    ) -> NDArray[np.complex128]:
        """Return a normalized wavefunction on a uniform grid."""
        
        return np.divide(psi,np.sqrt(AnalyticalQHO1D.norm(psi,dx)))

    @staticmethod
    def length_scale(
        m: float,
        omega: float,
        hbar: float,
    ) -> float:
        """Return the harmonic oscillator length scale.

        a = sqrt(hbar / (m omega))
        """
        
        return np.sqrt(hbar/(m*omega))


class NumericalQHO1D:
    """Finite-difference numerical model of the one-dimensional QHO.

    This class represents one numerical QHO problem.

    The constructor stores the physical parameters, constructs the grid,
    builds the operators, builds the Hamiltonian matrix, solves the
    eigenvalue problem, and stores the results.

    Students must decide how the Dirichlet boundary conditions affect the
    matrix problem.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int,
        hbar: float,
        m: float,
        omega: float,
    ) -> None:
        """Create and solve one numerical QHO problem."""
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.n_points: int = n_points

        self.hbar: float = hbar
        self.m: float = m
        self.omega: float = omega

        # Full position grid on [x_min, x_max].
        self.x_full: NDArray[np.float64]
        self.dx: float
        self.x_full, self.dx = self.grid(
            x_min=self.x_min,
            x_max=self.x_max,
            n_points=self.n_points,
        )

        # Grid points used as unknowns in the matrix problem.
        # This is where boundary conditions enter the numerical model.
        self.x_operator: NDArray[np.float64] = self.operator_grid(
            x_full=self.x_full,
        )

        # Number of degrees of freedom in the matrix problem.
        self.n_operator: int = self.x_operator.size

        # Finite-difference representation of d^2/dx^2.
        self.laplacian: NDArray[np.float64] = self.laplacian_1d(
            n_operator=self.n_operator,
            dx=self.dx,
        )

        # Matrix representation of the kinetic energy operator.
        self.kinetic_matrix: NDArray[np.float64] = self.kinetic_energy_matrix(
            laplacian=self.laplacian,
            hbar=self.hbar,
            m=self.m,
        )

        # Potential evaluated on the operator grid.
        self.potential_values: NDArray[np.float64] = self.potential(
            x=self.x_operator,
            m=self.m,
            omega=self.omega,
        )

        # Matrix representation of the potential energy operator.
        self.potential_matrix: NDArray[np.float64] = self.potential_energy_matrix(
            potential_values=self.potential_values,
        )

        # Matrix representation of the Hamiltonian operator.
        self.hamiltonian_matrix: NDArray[np.float64] = self.hamiltonian(
            kinetic_matrix=self.kinetic_matrix,
            potential_matrix=self.potential_matrix,
        )

        # Solve the matrix eigenvalue problem.
        eigenvalues, eigenvectors = self.eigenpairs(
            hamiltonian_matrix=self.hamiltonian_matrix,
        )

        # Store sorted eigenvalues and normalized eigenvectors.
        self.eigenvalues: NDArray[np.float64] = eigenvalues
        self.eigenvectors: NDArray[np.complex128] = self.normalize_eigenvectors(
            eigenvectors=eigenvectors,
            dx=self.dx,
        )

    @staticmethod
    def grid(
        x_min: float,
        x_max: float,
        n_points: int,
    ) -> tuple[NDArray[np.float64], float]:
        """Return a uniform grid and grid spacing.

        The full grid should include both endpoints:

            x_0 = x_min
            x_{N-1} = x_max

        Parameters
        ----------
        x_min:
            Left endpoint of the interval.

        x_max:
            Right endpoint of the interval.

        n_points:
            Number of grid points.

        Returns
        -------
        x:
            Full position grid.

        dx:
            Grid spacing.
        """
        # Constructing the grid
        x = np.linspace(start=x_min,stop=x_max,num=(n_points),endpoint=True)

        # Taking dx
        dx = x[1] - x[0]

        # Removing x_min and x_max (the endpoints)
        # x = x[1:-1]

        return [x,dx]

    @staticmethod
    def operator_grid(
        x_full: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the grid used as unknowns in the matrix problem.

        The full grid includes the endpoints x_min and x_max.

        For Dirichlet boundary conditions,

            psi(x_min) = 0,
            psi(x_max) = 0.

        Decide whether the endpoint values are unknowns or fixed values.
        This determines which grid points the Hamiltonian matrix acts on.
        """
        
        return x_full[1:-1]

    @staticmethod
    def laplacian_1d(
        n_operator: int,
        dx: float,
    ) -> NDArray[np.float64]:
        """Return the finite-difference second-derivative matrix.

        The centered finite-difference approximation is

            d^2 psi / dx^2
            approx
            (psi_{i+1} - 2 psi_i + psi_{i-1}) / dx^2.

        Parameters
        ----------
        n_operator:
            Number of unknown wavefunction values in the matrix problem.

        dx:
            Grid spacing.

        Returns
        -------
        RealArray
            Matrix representation of the one-dimensional Laplacian.
        """
        
         # Creating an N x N identity matrix
        D = np.eye(n_operator)

        # Setting -2 as coefficient for central wavefunction values
        D = np.multiply(-2,D)

        # Setting 1 as coefficient for left and right wavefunction values
        for i in range(n_operator-1):
            D[i,i+1] = 1
            D[i+1,i] = 1

        # Dividing by (dx)^2
        D = np.multiply(1/(dx**2),D)

        return D

    @staticmethod
    def kinetic_energy_matrix(
        laplacian: NDArray[np.float64],
        hbar: float,
        m: float,
    ) -> NDArray[np.float64]:
        """Return the kinetic energy matrix.

        T = -(hbar^2 / 2m) d^2/dx^2
        """
        return np.multiply(-(hbar**2)/(2*m),laplacian)

    @staticmethod
    def potential(
        x: ArrayLike,
        m: float,
        omega: float,
    ) -> NDArray[np.float64]:
        """Return the harmonic oscillator potential on a grid.

        V(x) = (1/2) m omega^2 x^2
        """

        # Obtaining potential values for all interior points of the grid
        return np.multiply(0.5*m*(omega**2),np.pow(x,2))

    @staticmethod
    def potential_energy_matrix(
        potential_values: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return the diagonal potential energy matrix.

        In the position basis, the potential energy operator is local.
        Therefore its matrix representation is diagonal.
        """
        
        # The potential value at any point x is multiplied with wavefunction value at the same point x
        return np.diag(v=potential_values,k=0)

    @staticmethod
    def hamiltonian(
        kinetic_matrix: NDArray[np.float64],
        potential_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the Hamiltonian matrix.

        H = T + V
        """
        
        return np.add(kinetic_matrix,potential_matrix)

    @staticmethod
    def sort_eigenpairs(
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sort eigenvalues and eigenvectors in ascending eigenvalue order.

        Eigenvectors are stored column-by-column.
        """
        
         # Using NumPy's "argsort"
        a = np.argsort(eigenvalues)

        # Creating empty matrices for receiving the sorted eigenvalues and eigenvectors
        sorted_eigenvalues = np.empty(eigenvalues.shape)
        sorted_eigenvectors = np.empty(eigenvectors.shape)

        # Sorting the eigenvalues and eigenvectors by ascending eigenvalue
        for i in range(eigenvalues.size):
            sorted_eigenvalues[i] = eigenvalues[a[i]]
            sorted_eigenvectors[:,i] = eigenvectors[:,a[i]]

        return [sorted_eigenvalues,sorted_eigenvectors]

    @staticmethod
    def eigenpairs(
        hamiltonian_matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the sorted eigenpairs of the Hamiltonian matrix."""
        
        # Calculating the eigenvalues and eigenvectors of H
        eigenvalues, eigenvectors = np.linalg.eig(hamiltonian_matrix)

        # Sorting the eigenvalues and eigenvectors by ascending eigenvalue
        sorted_eigenvalues, sorted_eigenvectors = NumericalQHO1D.sort_eigenpairs(eigenvalues,eigenvectors)

        return [sorted_eigenvalues,sorted_eigenvectors]

    @staticmethod
    def normalize_eigenvector(
        psi: ArrayLike,
        dx: float,
    ) -> NDArray[np.complex128]:
        """Return one normalized eigenvector."""
        
        return np.divide(psi,np.sqrt(np.multiply(np.inner(psi,psi),dx)))

    @staticmethod
    def normalize_eigenvectors(
        eigenvectors: ArrayLike,
        dx: float,
    ) -> NDArray[np.complex128]:
        """Normalize eigenvectors column-by-column."""
        
        normalized_eigenvectors = np.empty(eigenvectors.shape)

        for i in range(eigenvectors.shape[1]):
            normalized_eigenvectors[:,i] = NumericalQHO1D.normalize_eigenvector(eigenvectors[:,i],dx)

        return normalized_eigenvectors


    def analytical_energies(
        self,
        n_states: int,
    ) -> NDArray[np.float64]:
        """Return analytical QHO energies using this object's parameters."""
        return AnalyticalQHO1D.energies(
            n_states=n_states,
            hbar=self.hbar,
            omega=self.omega,
        )

    def energy_errors(
        self,
        n_states: int,
    ) -> NDArray[np.float64]:
        """Return numerical minus analytical energy errors."""
        exact: NDArray[np.float64] = self.analytical_energies(n_states=n_states)
        numerical: NDArray[np.float64] = self.eigenvalues[:n_states]

        return numerical - exact