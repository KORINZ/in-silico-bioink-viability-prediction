import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
import os

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score


sns.set_context("talk", font_scale=1.2)
sns.set_style("ticks")

# Preprocessing
data = pd.read_csv("alg_i1g_cross_power_law_fittings.csv")

data_filtered = data[["concentration", "temperature", "eta_0", "m", "n"]].copy()

# Apply logarithmic transformation to eta_0, m, and n
data_filtered[["eta_0", "m", "n"]] = np.log(data_filtered[["eta_0", "m", "n"]])

# Scaling the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_filtered)

# Separating input and output features
X = scaled_data[:, :2]  # concentration and temperature
Y = scaled_data[:, 2:]  # log(eta_0), log(m), log(n)


def train_svr_cv(Y: np.ndarray) -> tuple[SVR, np.float64, dict]:
    """Train an SVR model using grid search with cross-validation."""

    # Define the grid of hyperparameters
    param_grid = {
        "C": [i for i in np.linspace(10, 1000, 16)],
        "gamma": [i for i in np.linspace(0.01, 0.075, 16)],
        "epsilon": [i for i in np.linspace(0.01, 0.075, 16)],
    }

    # Grid search of parameters
    svr_grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=20,
        verbose=2,
        n_jobs=-1,
    )

    # Fit the grid search model
    svr_grid.fit(X, Y)

    # Extract the best model and best parameters
    best_model = svr_grid.best_estimator_
    best_params = svr_grid.best_params_

    # Perform cross-validated MSE evaluation
    mse_scores = -cross_val_score(
        best_model, X, Y, scoring="neg_mean_squared_error", cv=20
    )
    mean_mse = np.mean(mse_scores).round(4)

    return best_model, mean_mse, best_params


# Train models for all target variables
svr_models = []
mse_scores = []
best_params_list = []
for i in range(3):
    svr_model, mse_score, best_params = train_svr_cv(Y[:, i])
    svr_models.append(svr_model)
    mse_scores.append(mse_score)
    best_params_list.append(best_params)

# Calculate MSE scores in the original space
mse_scores_original_space = []
if scaler.mean_ is not None and scaler.var_ is not None:
    for i in range(3):
        Y_pred_scaled = cross_val_predict(svr_models[i], X, Y[:, i], cv=20)
        Y_pred_log = Y_pred_scaled * np.sqrt(scaler.var_[i + 2]) + scaler.mean_[i + 2]
        Y_actual_log = Y[:, i] * np.sqrt(scaler.var_[i + 2]) + scaler.mean_[i + 2]

        Y_pred = np.exp(Y_pred_log)
        Y_actual = np.exp(Y_actual_log)

        mse_original = np.mean((Y_actual - Y_pred) ** 2)
        mse_scores_original_space.append(mse_original)

# Calculate RMSE scores in the original space
rmse_scores_original_space = np.sqrt(mse_scores_original_space).round(3)

print(f"Total data points: {len(data)}")
for i, name in enumerate(["eta_0", "m", "n"]):
    print(f"Best parameters for SVR model for {name}: {best_params_list[i]}")
    print(
        f"Cross-validated Root Mean Squared Error for {name} in original space: {rmse_scores_original_space[i]:.3f}"
    )


def plot_3d_surface(model, title, feature_index, scaler, data, save_fig=False) -> None:
    """Plot a 3D surface plot of the predicted values for a given feature."""

    # Extract mean and std for concentration and temperature from the scaler
    concentration_mean, temperature_mean = scaler.mean_[:2]
    concentration_std, temperature_std = np.sqrt(scaler.var_[:2])

    # Extract mean and std for the feature to be predicted (eta_0, m, or n)
    feature_mean = scaler.mean_[feature_index]
    feature_std = np.sqrt(scaler.var_[feature_index])

    # Create a grid of values
    concentration_range = np.linspace(
        data["concentration"].min(), data["concentration"].max(), 200
    )
    temperature_range = np.linspace(
        data["temperature"].min(), data["temperature"].max(), 200
    )
    concentration_grid, temperature_grid = np.meshgrid(
        concentration_range, temperature_range
    )

    # Manually scale the grid data
    concentration_scaled = (
        concentration_grid.ravel() - concentration_mean
    ) / concentration_std
    temperature_scaled = (temperature_grid.ravel() - temperature_mean) / temperature_std
    prediction_input_scaled = np.column_stack(
        (concentration_scaled, temperature_scaled)
    )

    # Predict using the model
    predictions_scaled = model.predict(prediction_input_scaled)

    # Manually inverse transform the predictions to the log scale
    predictions_log = predictions_scaled * feature_std + feature_mean

    # Inverse transform the predictions from log scale to the original scale
    predictions_original = np.exp(predictions_log)

    predictions_original = predictions_original.reshape(concentration_grid.shape)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d", computed_zorder=False)

    # Set the title and zlabel based on the value of 'title'
    if title == "eta_0":
        zlabel = r"$\eta_0$"
        rmse = rmse_scores_original_space[0]
        symbol = r"$\eta_0$"
        name = f"Zero-Shear-Rate Viscosity ({symbol})"
        unit = "Pa·s"
        cmap = "viridis"
        marker = "o"
    elif title == "m":
        zlabel = "$m$"
        rmse = rmse_scores_original_space[1]
        symbol = "$m$"
        name = f"Time Constant ({symbol})"
        unit = "s"
        cmap = "plasma"
        marker = "s"
    else:
        zlabel = "$n$"
        rmse = rmse_scores_original_space[2]
        symbol = "$n$"
        name = f"Shear-Thinning Index ({symbol})"
        unit = "-"
        cmap = "coolwarm"
        marker = "^"

    original_data_points = data[["concentration", "temperature", title]]

    # Apply logarithm to predictions_original if the title is "eta_0" or "m"
    if title in ["eta_0", "m"]:
        predictions_original = np.log10(predictions_original)
        original_data_points.loc[:, title] = np.log10(original_data_points[title])

    ax.scatter(
        original_data_points["concentration"],
        original_data_points["temperature"],
        original_data_points[title],
        marker=marker,
        edgecolors="black",
        color="red",
        s=55,
        label=f"Fitted Empirical Data",
        depthshade=False,
        zorder=100,
        alpha=0.85,
    )

    surf = ax.plot_surface(
        concentration_grid,
        temperature_grid,
        predictions_original,
        cmap=cmap,
        edgecolors="black",
        linewidth=0.3,
        alpha=0.75,
    )

    ax.set_xticks([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ax.set_xticklabels([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ax.set_yticks([4, 10, 20, 25, 30, 37, 45])

    # Set the z-ticks and labels for "eta_0" and "m" in the original values
    if title == "eta_0":
        ax.set_zticks(
            [np.log10(0.1), np.log10(1), np.log10(10), np.log10(100), np.log10(1000)]
        )
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    elif title == "m":
        ax.set_zticks([np.log10(0.01), np.log10(0.1), np.log10(1), np.log10(10)])
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    elif title == "n":
        ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    ax.set_xlabel("Concentration (% w/v)", fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontweight="bold")
    ax.set_zlabel(zlabel + f" ({unit})", fontweight="bold")
    ax.set_title(
        f"{name}",
    )

    # Add padding to labels
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 12

    # Add padding to tick labels
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=5)
    ax.tick_params(axis="z", pad=5)

    # Initial view angle
    ax.view_init(25, 130)

    plt.legend(loc="upper center", edgecolor="black", fancybox=False)

    if save_fig:
        if not os.path.exists("svm/images"):
            os.makedirs("svm/images")
        plt.savefig(
            f"svm/images/{title}_surface_plot_{title}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.4,
        )

    plt.show()


def log_tick_formatter(val, pos=None) -> str:
    """Format the log scale ticks for the z-axis."""
    return f"$10^{{{val:g}}}$"


def plot_actual_vs_predicted_cv(
    model, feature_index, scaler, X, Y, title, save_fig=False
) -> None:
    """Plot the actual vs predicted values for a given feature using cross-validation."""

    # Predict using cross-validation
    Y_pred_scaled = cross_val_predict(model, X, Y[:, feature_index - 2], cv=20)

    # Inverse transform the predictions and actual values to the log scale
    Y_pred_log = (
        Y_pred_scaled * np.sqrt(scaler.var_[feature_index])
        + scaler.mean_[feature_index]
    )
    Y_actual_log = (
        Y[:, feature_index - 2] * np.sqrt(scaler.var_[feature_index])
        + scaler.mean_[feature_index]
    )

    # Inverse transform the predictions and actual values from log scale to the original scale
    Y_pred = np.exp(Y_pred_log)
    Y_actual = np.exp(Y_actual_log)

    # Calculate R-squared
    r2 = r2_score(Y_actual, Y_pred)

    # Calculate MAE
    mae = np.mean(np.abs(Y_actual - Y_pred))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((Y_actual - Y_pred) ** 2))

    plt.figure(figsize=(8, 8))

    if title == "eta_0":
        unit = "Pa·s"
        marker = "o"
        plt.title(r"$\eta_0$")
    elif title == "m":
        unit = "s"
        marker = "s"
        plt.title("$m$")
    else:
        unit = "-"
        marker = "^"
        plt.title("$n$")

    # Plot
    plt.scatter(Y_actual, Y_pred, alpha=0.5, s=55, marker=marker, color="blue")
    plt.plot(
        [Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], "r--", lw=2
    )
    plt.xlabel(f"Actual ({unit})", fontweight="bold")
    plt.ylabel(f"Predicted ({unit})", fontweight="bold")

    if title == "n":
        unit = ""

    # Create a box with error evaluation results
    textstr = "\n".join(
        [f"R-squared: {r2:.3f}", f"MAE: {mae:.3f} {unit}", f"RMSE: {rmse:.3f} {unit}"]
    )
    props = dict(facecolor="white", alpha=1, edgecolor="black")
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=22,
        verticalalignment="top",
        bbox=props,
    )

    # log scale
    if title == "eta_0" or title == "m":
        plt.xscale("log")
        plt.yscale("log")

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    if save_fig:
        if not os.path.exists("svm/images"):
            os.makedirs("svm/images")
        plt.savefig(
            f"svm/images/{title}_actual_vs_predicted_cv.png",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


# Feature indices: 2 for eta_0, 3 for m, 4 for n
for i, name in enumerate(["eta_0", "m", "n"]):
    plot_3d_surface(svr_models[i], name, i + 2, scaler, data, save_fig=True)
    plot_actual_vs_predicted_cv(svr_models[i], i + 2, scaler, X, Y, name, save_fig=True)

if not os.path.exists("svm/model"):
    os.makedirs("svm/model")

for i, name in enumerate(["eta_0", "m", "n"]):
    with open(f"svm/model/svr_{name}_model_log.pkl", "wb") as file:
        pickle.dump(svr_models[i], file)

# Save the scaler
with open("svm/model/scaler_log.pkl", "wb") as file:
    pickle.dump(scaler, file)
