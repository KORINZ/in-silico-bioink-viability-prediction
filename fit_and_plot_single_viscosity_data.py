import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import seaborn as sns

sns.set_context("talk", font_scale=1.1)
sns.set_style("ticks")


# Bird-Carreau model, a = 2
def bird_carreau(gamma_dot, eta_inf, eta0, k, n) -> float:
    return eta_inf + (eta0 - eta_inf) * (1 + (k * gamma_dot) ** 2) ** ((n - 1) / 2)


# Cross-Power Law model
def cross_power_law(gamma_dot, eta_inf, eta0, m, n) -> float:
    return eta_inf + (eta0 - eta_inf) / (1 + (m * gamma_dot) ** n)


# Power Law model
def power_law(gamma_dot, k, n) -> float:
    return k * gamma_dot ** (n - 1)


# Shear stress from viscosity
def shear_stress_from_viscosity(gamma_dot, eta) -> float:
    return gamma_dot * eta


def weighted_root_mean_squared_error(y_obs, y_pred, weights) -> np.float64:
    return np.sqrt(np.sum(weights * (y_obs - y_pred) ** 2) / np.sum(weights))


def plot_single_data_file_with_models(df) -> None:
    # Convert viscosity from mPas to Pas
    df = df.copy()

    df = df[df["ɣ̇ in 1/s"] >= 0.01]

    df = df[df["ɣ̇ in 1/s"] <= 1000]

    df.loc[:, "η in Pas"] = df["η in mPas"] * 1e-3

    # Fit models
    popt_bc, _ = curve_fit(
        bird_carreau,
        df["ɣ̇ in 1/s"],
        df["η in Pas"],
        sigma=(1.0 / df["ɣ̇ in 1/s"]) ** 0.75,
        bounds=([1e-3, 0, 0, 0], [np.inf, np.inf, np.inf, 1]),
        maxfev=10000,
    )
    popt_bc = [round(val, 3) for val in popt_bc]
    popt_cross, _ = curve_fit(
        cross_power_law,
        df["ɣ̇ in 1/s"],
        df["η in Pas"],
        bounds=([1e-3, 0, 0, 0], [1.01e-3, np.inf, np.inf, 1]),
        sigma=(1.0 / df["ɣ̇ in 1/s"]) ** 0.5,
        maxfev=10000,
    )
    popt_power, _ = curve_fit(
        power_law,
        df["ɣ̇ in 1/s"],
        df["η in Pas"],
        sigma=1.0 / df["ɣ̇ in 1/s"],
        bounds=([0, 0], [np.inf, 1]),
        maxfev=10000,
    )

    # Calculate WRMSE for each model
    weights = (1.0 / df["ɣ̇ in 1/s"]) ** 0.5
    rmse_bc = weighted_root_mean_squared_error(
        df["η in Pas"], bird_carreau(df["ɣ̇ in 1/s"], *popt_bc), weights
    ).round(2)
    rmse_cross = weighted_root_mean_squared_error(
        df["η in Pas"], cross_power_law(df["ɣ̇ in 1/s"], *popt_cross), weights
    ).round(2)
    rmse_power = weighted_root_mean_squared_error(
        df["η in Pas"], power_law(df["ɣ̇ in 1/s"], *popt_power), weights
    ).round(2)

    # Print parameters
    print(
        f"Bird-Carreau: η∞={popt_bc[0]:.4f}, η₀={popt_bc[1]:.3f}, k={popt_bc[2]:.3f}, n={popt_bc[3]:.3f}, RMSE={rmse_bc}"
    )
    print(
        f"Cross Power Law: η∞={popt_cross[0]:.4f}, η₀={popt_cross[1]:.3f}, m={popt_cross[2]:.5f}, n={popt_cross[3]:.3f}, RMSE={rmse_cross}"
    )
    print(f"Power Law: k={popt_power[0]:.3f}, n={popt_power[1]:.3f}, RMSE={rmse_power}")

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    x = np.logspace(-3, np.log10(20000), 1000)
    # Viscosity plot
    sns.scatterplot(
        x=df["ɣ̇ in 1/s"],
        y=df["η in Pas"],
        ax=axs[0],
        marker="o",
        s=65,
        color="r",
        label="Experimental",
        alpha=1,
    )
    axs[0].plot(
        x,
        bird_carreau(x, *popt_bc),
        "k--",
        alpha=0.9,
        label=f"Bird-Carreau (RMSE: {rmse_bc} Pa·s)",
    )
    axs[0].plot(
        x,
        cross_power_law(x, *popt_cross),
        "b-",
        alpha=0.9,
        label=f"Cross Power Law (RMSE: {rmse_cross} Pa·s)",
    )
    axs[0].plot(
        x,
        power_law(x, *popt_power),
        "g-.",
        alpha=0.9,
        label=f"Power Law (RMSE: {rmse_power} Pa·s)",
    )
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_title("(I)")
    axs[0].set_xlabel(
        r"Shear Rate ($\dot{\gamma}$) in s$^\mathbf{-1}$", fontweight="bold"
    )
    axs[0].set_ylabel(r"Apparent Viscosity ($\eta$) in Pa·s", fontweight="bold")
    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4])
    axs[0].grid(True, which="both", ls="-", alpha=0.5)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(
        handles[::-1],
        labels[::-1],
        fontsize=14.5,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        loc="upper right",
        fancybox=False,
    )

    # Shear Stress plot

    rmse_bc_ss = weighted_root_mean_squared_error(
        df["τ in Pa"],
        shear_stress_from_viscosity(
            df["ɣ̇ in 1/s"], bird_carreau(df["ɣ̇ in 1/s"], *popt_bc)
        ),
        weights,
    ).round(2)
    rmse_cross_ss = weighted_root_mean_squared_error(
        df["τ in Pa"],
        shear_stress_from_viscosity(
            df["ɣ̇ in 1/s"], cross_power_law(df["ɣ̇ in 1/s"], *popt_cross)
        ),
        weights,
    ).round(2)
    rmse_power_ss = weighted_root_mean_squared_error(
        df["τ in Pa"],
        shear_stress_from_viscosity(
            df["ɣ̇ in 1/s"], power_law(df["ɣ̇ in 1/s"], *popt_power)
        ),
        weights,
    ).round(2)

    sns.scatterplot(
        x=df["ɣ̇ in 1/s"],
        y=df["τ in Pa"],
        ax=axs[1],
        s=65,
        marker="D",
        label="Experimental",
        color="r",
        alpha=1,
    )
    axs[1].plot(
        x,
        shear_stress_from_viscosity(x, bird_carreau(x, *popt_bc)),
        "k--",
        alpha=0.9,
        label=f"Bird-Carreau (RMSE: {rmse_bc_ss} Pa)",
    )
    axs[1].plot(
        x,
        shear_stress_from_viscosity(x, cross_power_law(x, *popt_cross)),
        "b-",
        alpha=0.9,
        label=f"Cross Power Law (RMSE: {rmse_cross_ss} Pa)",
    )
    axs[1].plot(
        x,
        # shear_stress_from_power_law(x, *popt_power),
        shear_stress_from_viscosity(x, power_law(x, *popt_power)),
        "g-.",
        alpha=0.9,
        label=f"Power Law (RMSE: {rmse_power_ss} Pa)",
    )
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_title("(II)")

    # move y-axis label to the right
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")

    axs[1].set_xlabel(r"$\dot{\gamma}$ (s$^\mathbf{-1}$)", fontweight="bold")
    axs[1].set_ylabel(r"Shear Stress ($\tau$) in Pa", fontweight="bold")
    axs[1].set_xlim(x.min(), x.max())
    axs[1].set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4])
    axs[1].grid(True, which="both", ls="-", alpha=0.5)

    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(
        handles[::-1],
        labels[::-1],
        fontsize=14.5,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        loc="lower right",
        fancybox=False,
    )

    plt.tight_layout()
    plt.savefig("fitted_data.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    file_path = r"alg_i1g_4.0_wv_25C.xlsx"
    df = pd.read_excel(file_path.replace("\\", "/"))
    print(f"\n{file_path}")
    plot_single_data_file_with_models(df)
