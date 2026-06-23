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
    ) -> RealArray:
        """Return the harmonic oscillator potential V(x).

        V(x) = (1/2) m omega^2 x^2
        """
        raise NotImplementedError

    @staticmethod
    def energy(
        n: int,
        hbar: float,
        omega: float,
    ) -> float:
        """Return the analytical energy eigenvalue E_n.

        E_n = hbar omega (n + 1/2)
        """
        raise NotImplementedError

    @staticmethod
    def energies(
        n_states: int,
        hbar: float,
        omega: float,
    ) -> RealArray:
        """Return the first n_states analytical energy eigenvalues.

        The returned array should contain

            E_0, E_1, ..., E_{n_states - 1}.
        """
        raise NotImplementedError

    @staticmethod
    def probability_density(
        psi: ArrayLike,
    ) -> RealArray:
        """Return the probability density |psi|^2."""
        raise NotImplementedError

    @staticmethod
    def norm(
        psi: ArrayLike,
        dx: float,
    ) -> float:
        """Return the grid norm sqrt(sum_i |psi_i|^2 dx)."""
        raise NotImplementedError

    @staticmethod
    def normalize(
        psi: ArrayLike,
        dx: float,
    ) -> ComplexArray:
        """Return a normalized wavefunction on a uniform grid."""
        raise NotImplementedError

    @staticmethod
    def length_scale(
        m: float,
        omega: float,
        hbar: float,
    ) -> float:
        """Return the harmonic oscillator length scale.

        a = sqrt(hbar / (m omega))
        """
        raise NotImplementedError


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
        self.x_full: RealArray
        self.dx: float
        self.x_full, self.dx = self.grid(
            x_min=self.x_min,
            x_max=self.x_max,
            n_points=self.n_points,
        )

        # Grid points used as unknowns in the matrix problem.
        # This is where boundary conditions enter the numerical model.
        self.x_operator: RealArray = self.operator_grid(
            x_full=self.x_full,
        )

        # Number of degrees of freedom in the matrix problem.
        self.n_operator: int = self.x_operator.size

        # Finite-difference representation of d^2/dx^2.
        self.laplacian: RealArray = self.laplacian_1d(
            n_operator=self.n_operator,
            dx=self.dx,
        )

        # Matrix representation of the kinetic energy operator.
        self.kinetic_matrix: RealArray = self.kinetic_energy_matrix(
            laplacian=self.laplacian,
            hbar=self.hbar,
            m=self.m,
        )

        # Potential evaluated on the operator grid.
        self.potential_values: RealArray = self.potential(
            x=self.x_operator,
            m=self.m,
            omega=self.omega,
        )

        # Matrix representation of the potential energy operator.
        self.potential_matrix: RealArray = self.potential_energy_matrix(
            potential_values=self.potential_values,
        )

        # Matrix representation of the Hamiltonian operator.
        self.hamiltonian_matrix: RealArray = self.hamiltonian(
            kinetic_matrix=self.kinetic_matrix,
            potential_matrix=self.potential_matrix,
        )

        # Solve the matrix eigenvalue problem.
        eigenvalues, eigenvectors = self.eigenpairs(
            hamiltonian_matrix=self.hamiltonian_matrix,
        )

        # Store sorted eigenvalues and normalized eigenvectors.
        self.eigenvalues: RealArray = eigenvalues
        self.eigenvectors: ComplexArray = self.normalize_eigenvectors(
            eigenvectors=eigenvectors,
            dx=self.dx,
        )

    @staticmethod
    def grid(
        x_min: float,
        x_max: float,
        n_points: int,
    ) -> tuple[RealArray, float]:
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
        raise NotImplementedError

    @staticmethod
    def operator_grid(
        x_full: RealArray,
    ) -> RealArray:
        """Return the grid used as unknowns in the matrix problem.

        The full grid includes the endpoints x_min and x_max.

        For Dirichlet boundary conditions,

            psi(x_min) = 0,
            psi(x_max) = 0.

        Decide whether the endpoint values are unknowns or fixed values.
        This determines which grid points the Hamiltonian matrix acts on.
        """
        raise NotImplementedError

    @staticmethod
    def laplacian_1d(
        n_operator: int,
        dx: float,
    ) -> RealArray:
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
        raise NotImplementedError

    @staticmethod
    def kinetic_energy_matrix(
        laplacian: RealArray,
        hbar: float,
        m: float,
    ) -> RealArray:
        """Return the kinetic energy matrix.

        T = -(hbar^2 / 2m) d^2/dx^2
        """
        raise NotImplementedError

    @staticmethod
    def potential(
        x: ArrayLike,
        m: float,
        omega: float,
    ) -> RealArray:
        """Return the harmonic oscillator potential on a grid.

        V(x) = (1/2) m omega^2 x^2
        """
        raise NotImplementedError

    @staticmethod
    def potential_energy_matrix(
        potential_values: ArrayLike,
    ) -> RealArray:
        """Return the diagonal potential energy matrix.

        In the position basis, the potential energy operator is local.
        Therefore its matrix representation is diagonal.
        """
        raise NotImplementedError

    @staticmethod
    def hamiltonian(
        kinetic_matrix: RealArray,
        potential_matrix: RealArray,
    ) -> RealArray:
        """Return the Hamiltonian matrix.

        H = T + V
        """
        raise NotImplementedError

    @staticmethod
    def sort_eigenpairs(
        eigenvalues: RealArray,
        eigenvectors: RealArray,
    ) -> tuple[RealArray, RealArray]:
        """Sort eigenvalues and eigenvectors in ascending eigenvalue order.

        Eigenvectors are stored column-by-column.
        """
        raise NotImplementedError

    @staticmethod
    def eigenpairs(
        hamiltonian_matrix: RealArray,
    ) -> tuple[RealArray, RealArray]:
        """Return the sorted eigenpairs of the Hamiltonian matrix."""
        raise NotImplementedError

    @staticmethod
    def normalize_eigenvector(
        psi: ArrayLike,
        dx: float,
    ) -> ComplexArray:
        """Return one normalized eigenvector."""
        raise NotImplementedError

    @staticmethod
    def normalize_eigenvectors(
        eigenvectors: ArrayLike,
        dx: float,
    ) -> ComplexArray:
        """Normalize eigenvectors column-by-column."""
        raise NotImplementedError

    def analytical_energies(
        self,
        n_states: int,
    ) -> RealArray:
        """Return analytical QHO energies using this object's parameters."""
        return AnalyticalQHO1D.energies(
            n_states=n_states,
            hbar=self.hbar,
            omega=self.omega,
        )

    def energy_errors(
        self,
        n_states: int,
    ) -> RealArray:
        """Return numerical minus analytical energy errors."""
        exact: RealArray = self.analytical_energies(n_states=n_states)
        numerical: RealArray = self.eigenvalues[:n_states]

        return numerical - exact