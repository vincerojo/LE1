"""
Sommerfeld free-electron metal model.

Object split:

1. SommerfeldMetal
   - data object
   - stores physical parameters, derived Fermi quantities, and energy grid

2. SommerfeldMetalCalculator
   - action object
   - static methods only
   - computes quantities from explicit inputs or from a SommerfeldMetal object

All quantities are SI unless otherwise stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SommerfeldMetal:
    """
    Data object for a Sommerfeld free-electron metal.

    This object stores the state of the model. It does not perform numerical
    calculations beyond construction-time assignment.

    Parameters
    ----------
    electron_density:
        Electron density n in m^{-3}.
    electron_mass:
        Electron mass or effective mass m in kg.
    hbar:
        Reduced Planck constant in J s.
    k_B:
        Boltzmann constant in J K^{-1}.
    energy_grid:
        Numerical energy grid in J.
    k_F:
        Fermi wavevector in m^{-1}.
    E_F:
        Fermi energy in J.
    T_F:
        Fermi temperature in K.
    """

    electron_density: float
    electron_mass: float
    hbar: float
    k_B: float

    energy_grid: NDArray[np.float64]

    k_F: float
    E_F: float
    T_F: float

    @property
    def n(self) -> float:
        """Electron density in m^{-3}."""
        return self.electron_density

    @property
    def m(self) -> float:
        """Electron mass in kg."""
        return self.electron_mass

    def summary(self) -> dict[str, float]:
        """
        Return stored model quantities.
        """
        return {
            "electron_density_m^-3": self.electron_density,
            "electron_mass_kg": self.electron_mass,
            "fermi_wavevector_m^-1": self.k_F,
            "fermi_energy_J": self.E_F,
            "fermi_energy_eV": self.E_F / 1.602176634e-19,
            "fermi_temperature_K": self.T_F,
            "energy_min_J": float(self.energy_grid[0]),
            "energy_max_J": float(self.energy_grid[-1]),
            "num_energy_points": float(self.energy_grid.size),
        }


class SommerfeldMetalCalculator:
    """
    Static action object for Sommerfeld-metal calculations.

    This class owns no state.

    It implements the computational chain

        n -> E_F -> D(E) -> f(E,T) -> u(T) -> c_V^e(T)

    required by the assignment. :contentReference[oaicite:0]{index=0}
    """

    @staticmethod
    def create_metal(
        electron_density: float,
        electron_mass: float = 9.1093837015e-31,
        hbar: float = 1.054571817e-34,
        k_B: float = 1.380649e-23,
        energy_min: float = 0.0,
        energy_max: float | None = None,
        num_energy_points: int = 10_000,
    ) -> SommerfeldMetal:
        """
        Create a SommerfeldMetal data object from primitive parameters.

        This is a factory method. It computes construction-time derived data,
        then returns a passive data object.
        """
        SommerfeldMetalCalculator.validate_parameters(
            electron_density=electron_density,
            electron_mass=electron_mass,
            hbar=hbar,
            k_B=k_B,
            energy_min=energy_min,
            energy_max=energy_max,
            num_energy_points=num_energy_points,
        )

        k_F = SommerfeldMetalCalculator.fermi_wavevector(electron_density)
        E_F = SommerfeldMetalCalculator.fermi_energy(
            k_F=k_F,
            electron_mass=electron_mass,
            hbar=hbar,
        )
        T_F = SommerfeldMetalCalculator.fermi_temperature(
            E_F=E_F,
            k_B=k_B,
        )

        if energy_max is None:
            # For a known maximum temperature T_max, prefer:
            #
            #   energy_max >= E_F + 20 k_B T_max
            #
            # as specified in the assignment. :contentReference[oaicite:1]{index=1}
            energy_max = 3.0 * E_F

        energy_grid = SommerfeldMetalCalculator.energy_grid(
            energy_min=energy_min,
            energy_max=energy_max,
            num_energy_points=num_energy_points,
        )

        return SommerfeldMetal(
            electron_density=electron_density,
            electron_mass=electron_mass,
            hbar=hbar,
            k_B=k_B,
            energy_grid=energy_grid,
            k_F=k_F,
            E_F=E_F,
            T_F=T_F,
        )

    @staticmethod
    def validate_parameters(
        electron_density: float,
        electron_mass: float,
        hbar: float,
        k_B: float,
        energy_min: float,
        energy_max: float | None,
        num_energy_points: int,
    ) -> None:
        """
        Validate physical and numerical parameters.
        """
        if electron_density <= 0:
            raise ValueError("electron_density must be positive.")

        if electron_mass <= 0:
            raise ValueError("electron_mass must be positive.")

        if hbar <= 0:
            raise ValueError("hbar must be positive.")

        if k_B <= 0:
            raise ValueError("k_B must be positive.")

        if energy_min < 0:
            raise ValueError("energy_min must be nonnegative.")

        if energy_max is not None and energy_max <= energy_min:
            raise ValueError("energy_max must be greater than energy_min.")

        if num_energy_points < 2:
            raise ValueError("num_energy_points must be at least 2.")

    @staticmethod
    def energy_grid(
        energy_min: float,
        energy_max: float,
        num_energy_points: int,
    ) -> NDArray[np.float64]:
        """
        Create the numerical energy grid.

        Returns
        -------
        numpy.ndarray
            Energy grid in J.
        """
        return np.linspace(
            energy_min,
            energy_max,
            num_energy_points,
            dtype=np.float64,
        )

    @staticmethod
    def fermi_wavevector(electron_density: float) -> float:
        """
        Compute the Fermi wavevector.

        Formula
        -------
        k_F = (3 pi^2 n)^{1/3}
        """
        return float((3.0 * np.pi**2 * electron_density) ** (1.0 / 3.0))

    @staticmethod
    def fermi_energy(
        k_F: float,
        electron_mass: float,
        hbar: float,
    ) -> float:
        """
        Compute the Fermi energy.

        Formula
        -------
        E_F = hbar^2 k_F^2 / (2m)
        """
        return float(hbar**2 * k_F**2 / (2.0 * electron_mass))

    @staticmethod
    def fermi_temperature(E_F: float, k_B: float) -> float:
        """
        Compute the Fermi temperature.

        Formula
        -------
        T_F = E_F / k_B
        """
        return float(E_F / k_B)

    @staticmethod
    def density_of_states(
        E: ArrayLike,
        metal: SommerfeldMetal,
    ) -> NDArray[np.float64]:
        """
        Evaluate the 3D free-electron density of states per unit volume.

        Formula
        -------
        D(E) = (1 / 2 pi^2) (2m / hbar^2)^{3/2} E^{1/2}
        """
        E_array = np.asarray(E, dtype=np.float64)
        E_nonnegative = np.maximum(E_array, 0.0)

        prefactor = (1.0 / (2.0 * np.pi**2)) * (
            (2.0 * metal.m / metal.hbar**2) ** 1.5
        )

        return prefactor * np.sqrt(E_nonnegative)

    @staticmethod
    def fermi_dirac(
        E: ArrayLike,
        T: float,
        metal: SommerfeldMetal,
    ) -> NDArray[np.float64]:
        """
        Evaluate the Fermi-Dirac occupation using mu = E_F.

        At T = 0, this returns the limiting step function instead of
        evaluating the finite-temperature formula.
        """
        E_array = np.asarray(E, dtype=np.float64)

        if T < 0:
            raise ValueError("Temperature T must be nonnegative.")

        if T == 0:
            return np.where(E_array < metal.E_F, 1.0, 0.0)

        x = (E_array - metal.E_F) / (metal.k_B * T)

        # Prevent overflow in exp(x), as required by the numerical stability
        # section of the assignment. :contentReference[oaicite:2]{index=2}
        x_clipped = np.clip(x, -700.0, 700.0)

        return 1.0 / (np.exp(x_clipped) + 1.0)

    @staticmethod
    def occupied_density_of_states(
        E: ArrayLike,
        T: float,
        metal: SommerfeldMetal,
    ) -> NDArray[np.float64]:
        """
        Compute the occupied density of states.

        Formula
        -------
        D_occ(E,T) = D(E) f(E,T)
        """
        E_array = np.asarray(E, dtype=np.float64)

        return (
            SommerfeldMetalCalculator.density_of_states(E_array, metal)
            * SommerfeldMetalCalculator.fermi_dirac(E_array, T, metal)
        )

    @staticmethod
    def electron_density(
        T: float,
        metal: SommerfeldMetal,
    ) -> float:
        """
        Numerically approximate the finite-temperature electron density.

        Formula
        -------
        n_num(T) = integral_0^infty D(E) f(E,T) dE
        """
        E = metal.energy_grid

        integrand = SommerfeldMetalCalculator.occupied_density_of_states(
            E=E,
            T=T,
            metal=metal,
        )

        return float(np.trapezoid(integrand, E))

    @staticmethod
    def zero_temperature_density(metal: SommerfeldMetal) -> float:
        """
        Numerically check the zero-temperature density relation.

        Formula
        -------
        n = integral_0^{E_F} D(E) dE
        """
        E = metal.energy_grid[metal.energy_grid <= metal.E_F]

        if E.size < 2:
            raise ValueError("Energy grid has too few points below E_F.")

        integrand = SommerfeldMetalCalculator.density_of_states(E, metal)

        return float(np.trapezoid(integrand, E))

    @staticmethod
    def energy_density(
        T: float,
        metal: SommerfeldMetal,
    ) -> float:
        """
        Numerically approximate the electronic energy density.

        Formula
        -------
        u(T) = integral_0^infty E D(E) f(E,T) dE
        """
        E = metal.energy_grid

        integrand = E * SommerfeldMetalCalculator.occupied_density_of_states(
            E=E,
            T=T,
            metal=metal,
        )

        return float(np.trapezoid(integrand, E))

    @staticmethod
    def zero_temperature_energy_density(metal: SommerfeldMetal) -> float:
        """
        Numerically approximate the zero-temperature energy density.

        Formula
        -------
        u(0) = integral_0^{E_F} E D(E) dE
        """
        E = metal.energy_grid[metal.energy_grid <= metal.E_F]

        if E.size < 2:
            raise ValueError("Energy grid has too few points below E_F.")

        integrand = E * SommerfeldMetalCalculator.density_of_states(E, metal)

        return float(np.trapezoid(integrand, E))

    @staticmethod
    def thermal_excess_energy_density(
        T: float,
        metal: SommerfeldMetal,
    ) -> float:
        """
        Compute the thermal excess energy density.

        Formula
        -------
        Delta u(T) = u(T) - u(0)
        """
        return (
            SommerfeldMetalCalculator.energy_density(T, metal)
            - SommerfeldMetalCalculator.zero_temperature_energy_density(metal)
        )

    @staticmethod
    def heat_capacity(
        T: float,
        dT: float,
        metal: SommerfeldMetal,
    ) -> float:
        """
        Compute electronic heat capacity by centered finite difference.

        Formula
        -------
        c_V^e(T) approx [Delta u(T + dT) - Delta u(T - dT)] / (2 dT)
        """
        if T <= 0:
            raise ValueError("T must be positive for centered finite difference.")

        if dT <= 0:
            raise ValueError("dT must be positive.")

        if T - dT < 0:
            raise ValueError("Centered finite difference requires T - dT >= 0.")

        u_plus = SommerfeldMetalCalculator.thermal_excess_energy_density(
            T=T + dT,
            metal=metal,
        )
        u_minus = SommerfeldMetalCalculator.thermal_excess_energy_density(
            T=T - dT,
            metal=metal,
        )

        return float((u_plus - u_minus) / (2.0 * dT))

    @staticmethod
    def low_temperature_heat_capacity(
        T: ArrayLike,
        metal: SommerfeldMetal,
    ) -> NDArray[np.float64]:
        """
        Compute the low-temperature Sommerfeld heat-capacity approximation.

        Formula
        -------
        c_V^e(T) approx (pi^2 / 2) n k_B^2 T / E_F
        """
        T_array = np.asarray(T, dtype=np.float64)

        return (
            (np.pi**2 / 2.0)
            * metal.n
            * metal.k_B**2
            * T_array
            / metal.E_F
        )

    @staticmethod
    def relative_heat_capacity_error(
        T: float,
        dT: float,
        metal: SommerfeldMetal,
    ) -> float:
        """
        Compare numerical heat capacity against the low-temperature result.
        """
        c_num = SommerfeldMetalCalculator.heat_capacity(
            T=T,
            dT=dT,
            metal=metal,
        )
        c_low = float(
            SommerfeldMetalCalculator.low_temperature_heat_capacity(
                T=T,
                metal=metal,
            )
        )

        return float(abs(c_num - c_low) / abs(c_low))


def main() -> None:
    """
    Minimal executable example.
    """
    metal = SommerfeldMetalCalculator.create_metal(
        electron_density=8.47e28,
        num_energy_points=20_000,
    )

    T = 300.0
    dT = 1.0

    print("Sommerfeld metal data")
    for key, value in metal.summary().items():
        print(f"{key}: {value:.6e}")

    print()
    print(
        f"n_num({T} K): "
        f"{SommerfeldMetalCalculator.electron_density(T, metal):.6e} m^-3"
    )
    print(
        f"n_0,num: "
        f"{SommerfeldMetalCalculator.zero_temperature_density(metal):.6e} m^-3"
    )
    print(
        f"u(0): "
        f"{SommerfeldMetalCalculator.zero_temperature_energy_density(metal):.6e} "
        "J m^-3"
    )
    print(
        f"u({T} K): "
        f"{SommerfeldMetalCalculator.energy_density(T, metal):.6e} J m^-3"
    )
    print(
        f"Delta u({T} K): "
        f"{SommerfeldMetalCalculator.thermal_excess_energy_density(T, metal):.6e} "
        "J m^-3"
    )
    print(
        f"c_V^e,num({T} K): "
        f"{SommerfeldMetalCalculator.heat_capacity(T, dT, metal):.6e} "
        "J m^-3 K^-1"
    )
    print(
        f"c_V^e,lowT({T} K): "
        f"{float(SommerfeldMetalCalculator.low_temperature_heat_capacity(T, metal)):.6e} "
        "J m^-3 K^-1"
    )
    print(
        f"relative heat-capacity error: "
        f"{SommerfeldMetalCalculator.relative_heat_capacity_error(T, dT, metal):.6e}"
    )


if __name__ == "__main__":
    main()