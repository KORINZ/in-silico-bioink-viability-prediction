import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns

sns.set_style("ticks")
sns.set_context("talk", font_scale=1.21)

# Parameters
R = 0.1e-3  # Radius of the needle (0.2 mm diameter)
rho = 1000
eta_inf = 1e-3
eta0 = 203.20667
m = 0.981701
n = 0.666894
L = 20e-3  # Length of the needle (20 mm)
dp_dz = (
    -1465.807e3 / L
)  # !Pressure gradient from the simulation results to validate the model


def cross_power_law(gamma_dot, eta_inf, eta0, m, n):
    return eta_inf + (eta0 - eta_inf) / (1 + (m * np.abs(gamma_dot)) ** n)


def dvdr(v, r):
    gamma_dot = -dvdr.dv_dr
    eta = cross_power_law(gamma_dot, eta_inf, eta0, m, n)
    dvdr.dv_dr = dp_dz * r / (2 * eta)
    return dvdr.dv_dr


def main():
    # Load and preprocess data
    df = pd.read_csv(r"openfoam_alg_i1g_5.5wv_3uLs_25C.csv")

    plot_cols_shearStress = [col for col in df.columns if "shearStress" in col]
    df_shearStress = (df[plot_cols_shearStress] * rho) ** 2
    df_shearStress = np.array(df_shearStress.sum(axis=1) ** (1 / 2))[:-1] * 1e-3
    xx_shearStress = np.linspace(1e-3, R * 1e3, len(df_shearStress))

    plot_cols_U = [
        col for col in df.columns if "U:" in col
    ]  # put : to avoid selecting grad(U)
    df_U = (df[plot_cols_U]) ** 2  # m/s
    df_U = df_U.sum(axis=1) ** (1 / 2)
    xx_U = np.linspace(1e-3, R * 1e3, len(df_U))

    plot_cols_gradU = [col for col in df.columns if "grad(U):" in col]
    df_gradU = (df[plot_cols_gradU]) ** 2
    df_gradU = df_gradU.sum(axis=1) ** (1 / 2)
    xx_gradU = np.linspace(1e-3, R * 1e3, len(df_gradU))

    plot_cols_eta = [col for col in df.columns if "strainRateViscosityModel:nu" in col]
    df_eta = df[plot_cols_eta] * rho
    xx_eta = np.linspace(1e-3, R * 1e3, len(df_eta))

    # Downsample the simulation data
    downsample_factor = 27
    xx_shearStress_downsampled = xx_shearStress[::downsample_factor]
    df_shearStress_downsampled = df_shearStress[::downsample_factor]
    xx_U_downsampled = xx_U[::downsample_factor]
    df_U_downsampled = df_U[::downsample_factor]
    xx_gradU_downsampled = xx_gradU[::downsample_factor]
    df_gradU_downsampled = df_gradU[::downsample_factor]
    xx_eta_downsampled = xx_eta[::downsample_factor]
    df_eta_downsampled = df_eta[::downsample_factor]

    # Radial positions
    r = np.linspace(0, R, 2000)

    # Solve the ODE for velocity profile
    v = np.zeros_like(r)
    for i in range(1, len(r)):
        dvdr.dv_dr = (v[i - 1] - v[i - 2]) / (r[i - 1] - r[i - 2])
        v[i] = odeint(dvdr, v[i - 1], [r[i - 1], r[i]])[-1, 0]

    # Calculate shear stress
    gamma_dot = np.gradient(v) / np.gradient(r)
    eta = cross_power_law(gamma_dot, eta_inf, eta0, m, n)
    tau = -eta * gamma_dot * 1e-3

    # Calculate wall shear stress
    tau_wall = -eta[-1] * gamma_dot[-1]

    v = v + abs(v.min())
    gamma_dot = -gamma_dot

    print(f"Average velocity: {np.mean(v):.4f} m/s")
    print(f"Average flow rate: {np.pi * R ** 2 * np.mean(v) * 1e9:.4f} µL/s")
    print(f"Maximum wall shear stress at the wall: {tau_wall:.3f} kPa")
    print(f"Cell residence time: {L / np.mean(v) * 1e3:.3f} ms")

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    axs[0, 0].plot(
        r * 1e3, v, label="Analytical", alpha=0.9, color="tab:red", linewidth=3
    )
    axs[0, 0].scatter(
        xx_U_downsampled,
        df_U_downsampled,
        label="Simulation",
        alpha=0.9,
        color="tab:blue",
        s=30,
        zorder=10,
    )
    axs[0, 0].set_xlabel(r"$r$ (mm)", fontweight="bold")
    axs[0, 0].set_ylabel(r"$u$ (m/s)", fontweight="bold")
    axs[0, 0].set_title("(I)")
    axs[0, 0].grid()
    axs[0, 0].legend(
        loc="lower left",
        framealpha=1,
        edgecolor="black",
        fancybox=False,
    )
    axs[0, 1].plot(
        r * 1e3, tau, label="Analytical", alpha=0.9, color="tab:red", linewidth=3
    )
    axs[0, 1].scatter(
        xx_shearStress_downsampled,
        df_shearStress_downsampled,
        label="Simulation",
        alpha=0.9,
        color="tab:blue",
        s=30,
        zorder=10,
    )
    axs[0, 1].set_xlabel(r"$r$ (mm)", fontweight="bold")
    axs[0, 1].set_ylabel(r"$\tau$ (kPa)", fontweight="bold")
    axs[0, 1].set_title("(II)")
    axs[0, 1].yaxis.tick_right()
    axs[0, 1].yaxis.set_label_position("right")
    axs[0, 1].grid()

    axs[1, 0].plot(
        r * 1e3, gamma_dot, label="Analytical", alpha=0.9, color="tab:red", linewidth=3
    )
    axs[1, 0].scatter(
        xx_gradU_downsampled,
        df_gradU_downsampled,
        label="Simulation",
        alpha=0.9,
        color="tab:blue",
        s=30,
        zorder=10,
    )
    axs[1, 0].set_xlabel(r"$r$ (mm)", fontweight="bold")
    axs[1, 0].set_ylabel(r"$\dot{\gamma}$ (s$^\mathbf{-1}$)", fontweight="bold")
    axs[1, 0].set_title("(III)")
    axs[1, 0].set_yscale("log")
    axs[1, 0].grid()

    axs[1, 1].plot(
        r * 1e3, eta, label="Analytical", alpha=0.9, color="tab:red", linewidth=3
    )
    axs[1, 1].scatter(
        xx_eta_downsampled,
        df_eta_downsampled,
        label="Simulation",
        alpha=0.9,
        color="tab:blue",
        s=30,
        zorder=10,
    )
    axs[1, 1].set_xlabel(r"$r$ (mm)", fontweight="bold")
    axs[1, 1].set_ylabel(r"$\eta$ (Pa·s)", fontweight="bold")
    axs[1, 1].set_title("(IV)")
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].yaxis.set_label_position("right")
    # log scale
    axs[1, 1].set_yscale("log")
    axs[1, 1].grid()

    plt.tight_layout()
    plt.savefig("analytical_vs_simulation.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
