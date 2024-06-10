import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_context("talk", font_scale=1.1)
sns.set_style("ticks")


CELL_TYPE = "HUEhT-1"

HIDDEN_LAYER_SIZES = (
    [(i,) for i in [16, 32, 64, 128, 256]]
    + [(i, i) for i in [16, 32, 64, 128]]
    + [(i,) for i in [20, 50, 80]]
    + [(i, i) for i in [20, 50, 80]]
)

if CELL_TYPE == "HUEhT-1":
    marker = "o"
    color = "red"
    color_map = sns.dark_palette("red", as_cmap=True)

elif CELL_TYPE == "HeLa":
    marker = "s"
    color = "blue"
    color_map = sns.dark_palette("blue", as_cmap=True)

elif CELL_TYPE == "10T12":
    marker = "^"
    color = "green"
    color_map = sns.dark_palette("green", as_cmap=True)
else:
    raise ValueError("Invalid CELL_TYPE.")


def preprocess_data(file_path) -> pd.DataFrame:
    """Preprocess the data by filtering the columns and return the DataFrame."""
    data = pd.read_csv(file_path)
    data_filtered = data[
        ["wall_shear_stress_kPa", "residence_time_ms", "relative_cell_viability"]
    ]
    return data_filtered


def scale_data(data) -> tuple[np.ndarray, StandardScaler]:
    """Scale the data using StandardScaler and return the scaled data and the scaler."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def separate_features(scaled_data) -> tuple[np.ndarray, np.ndarray]:
    """Separate the features and target variable from the scaled data."""
    x = scaled_data[:, :2]
    Y = scaled_data[:, 2:]
    return x, Y


def create_param_grid() -> dict[str, list]:
    """Create a parameter grid for GridSearchCV."""
    if CELL_TYPE == "HUEhT-1":
        return {
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(0.575, 0.75, 15)],
            "learning_rate": ["constant"],
        }
    elif CELL_TYPE == "HeLa":
        return {
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(1.0, 1.5, 15)],
            "learning_rate": ["constant"],
        }
    elif CELL_TYPE == "10T12":
        return {
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(0.15, 1, 15)],
            "learning_rate": ["constant"],
        }
    else:
        raise ValueError("Invalid CELL_TYPE.")


def train_mlp_cv(x, Y) -> tuple[MLPRegressor, np.float64, dict]:
    """Train a MLP model using GridSearchCV and return the best model, mean MSE, and best parameters."""
    param_grid = create_param_grid()
    mlp_grid = GridSearchCV(
        estimator=MLPRegressor(random_state=42, max_iter=500),
        param_grid=param_grid,
        cv=20,
        verbose=2,
        n_jobs=-1,
    )
    mlp_grid.fit(x, Y)
    best_model = mlp_grid.best_estimator_
    best_params = mlp_grid.best_params_
    mse_scores = -cross_val_score(
        best_model, x, Y, scoring="neg_mean_squared_error", cv=20
    )
    mean_mse = np.mean(mse_scores).round(4)
    return best_model, mean_mse, best_params


def scale_mse_scores(mse_scores, scaler) -> np.ndarray:
    """Scale the MSE scores back to the original space and return the RMSE scores."""
    if scaler.var_ is not None:
        mse_scores_original_space = mse_scores * scaler.var_[2:]
        rmse_scores_original_space = np.sqrt(mse_scores_original_space)
        return rmse_scores_original_space
    else:
        raise ValueError(
            "scaler.var_ is None. Make sure scaler.fit() is called on valid data."
        )


def plot_3d_surface(model, title, scaler, data, rsme, save_fig=False) -> None:
    """Plot a 3D surface plot of the MLP model predictions."""
    wall_shear_stress_kPa_mean, residence_time_ms_mean = scaler.mean_[:2]
    wall_shear_stress_kPa_std, residence_time_ms_std = np.sqrt(scaler.var_[:2])
    feature_mean = scaler.mean_[2]
    feature_std = np.sqrt(scaler.var_[2])

    wall_shear_stress_kPa_range = np.linspace(1, 5, 200)
    residence_time_ms_range = np.linspace(100, 700, 200)
    wall_shear_stress_kPa_grid, residence_time_ms_grid = np.meshgrid(
        wall_shear_stress_kPa_range, residence_time_ms_range
    )

    wall_shear_stress_kPa_scaled = (
        wall_shear_stress_kPa_grid.ravel() - wall_shear_stress_kPa_mean
    ) / wall_shear_stress_kPa_std
    residence_time_ms_scaled = (
        residence_time_ms_grid.ravel() - residence_time_ms_mean
    ) / residence_time_ms_std
    prediction_input_scaled = np.column_stack(
        (wall_shear_stress_kPa_scaled, residence_time_ms_scaled)
    )

    predictions_scaled = model.predict(prediction_input_scaled)
    predictions_original = predictions_scaled * feature_std + feature_mean
    predictions_original = predictions_original.reshape(
        wall_shear_stress_kPa_grid.shape
    )
    predictions_original = np.clip(predictions_original, 0, 100)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d", computed_zorder=False)

    ax.scatter(
        data["wall_shear_stress_kPa"],
        data["residence_time_ms"],
        data["relative_cell_viability"],
        marker=marker,
        color="lightgray",
        s=85,
        label=f"Experimental Data Point",
        depthshade=False,
        edgecolor="black",
        zorder=100,
        alpha=0.85,
    )
    surf = ax.plot_surface(
        wall_shear_stress_kPa_grid,
        residence_time_ms_grid,
        predictions_original,
        cmap=color_map,
        edgecolors="black",
        alpha=0.75,
        linewidth=0.75,
    )

    ax.set_xlabel("Wall Shear Stress (kPa)", fontweight="bold")
    ax.set_ylabel("Exposure Time (ms)", fontweight="bold")
    ax.set_zlabel("Cell Viability (%)", fontweight="bold")

    # padding for labels
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10

    ax.set_yticks([100, 200, 300, 400, 500, 600, 700])
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    ax.set_zticks([i for i in range(0, 101, 10)])

    ax.tick_params(axis="both", which="major")

    ax.set_title(
        f'{title.replace("10T12", "10T1/2")}',
    )
    ax.set_zlim(0, 100)
    ax.view_init(25, 55)

    plt.legend(loc="upper center", edgecolor="black", fancybox=False)

    if save_fig:
        os.makedirs(f"mlp/images/{title}", exist_ok=True)
        plt.savefig(
            f"mlp/images/{title}/{title}_3d_surface_plot.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.35,
        )

    plt.show()


def save_mlp_model(mlp_model, scaler, cell_type) -> None:
    """Save the MLP model and the scaler to a pickle file."""
    os.makedirs(f"mlp/model/{cell_type}", exist_ok=True)
    with open(f"mlp/model/{cell_type}/{cell_type}_mlp.pkl", "wb") as file:
        pickle.dump(mlp_model, file)
    with open(f"mlp/model/{cell_type}/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)


def plot_actual_vs_predicted_cv(model, x, Y, scaler) -> None:
    """Plot the actual vs predicted values using cross-validation."""
    y_pred = cross_val_predict(model, x, Y, cv=20)

    # Scale back the predictions and actual values to original space
    y_pred_original = y_pred * np.sqrt(scaler.var_[2]) + scaler.mean_[2]
    Y_original = Y * np.sqrt(scaler.var_[2]) + scaler.mean_[2]

    y_pred_original = np.clip(y_pred_original, 0, 100)

    # Calculate metrics
    r2 = r2_score(Y_original, y_pred_original)
    mae = mean_absolute_error(Y_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(Y_original, y_pred_original))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        Y_original,
        y_pred_original,
        alpha=0.5,
        edgecolor="k",
        s=55,
        color=color,
        zorder=100,
        marker=marker,
    )

    min_val = np.round(
        min(min(original, pred) for original, pred in zip(Y_original, y_pred_original)),
        -1,
    )

    ax.plot([min_val, 100], [min_val, 100], "k--", lw=2)
    ax.set_xlabel("Actual (%)", fontweight="bold", fontsize=24)
    ax.set_ylabel("Predicted (%)", fontweight="bold", fontsize=24)
    ax.set_title(f"{CELL_TYPE}")

    # Create a box with error evaluation results
    textstr = "\n".join(
        [f"R-squared: {r2:.3f}", f"MAE: {mae:.3f}%", f"RMSE: {rmse:.3f}%"]
    )
    props = dict(facecolor="white", alpha=1, edgecolor="black")
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
        fontsize=22,
    )

    ax.grid(True)

    ax.set_ylim(min_val, 100)
    ax.set_xlim(min_val, 100)

    # Use a step size of 5 for the ticks
    ax.set_xticks(np.arange(min_val, 100 + 1, 5))
    ax.set_yticks(np.arange(min_val, 100 + 1, 5))

    plt.tight_layout()
    plt.savefig(
        f"mlp/images/{CELL_TYPE}/{CELL_TYPE}_actual_vs_predicted.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()


def main() -> None:
    """Train a MLP model using GridSearchCV and plot the results."""
    data = preprocess_data(f"{CELL_TYPE}_all.csv")
    scaled_data, scaler = scale_data(data)
    x, Y = separate_features(scaled_data)

    mlp_model, mse_score, best_params = train_mlp_cv(x, Y[:, 0])
    rmse_scores_original_space = scale_mse_scores([mse_score], scaler)

    name = "relative_cell_viability"
    print(f"Best parameters for MLP model for {name}: {best_params}")
    print(
        f"Cross-validated Root Mean Squared Error for {name}: {rmse_scores_original_space[0] :.3f}"
    )

    plot_3d_surface(
        mlp_model, CELL_TYPE, scaler, data, rmse_scores_original_space[0], save_fig=True
    )
    plot_actual_vs_predicted_cv(mlp_model, x, Y[:, 0], scaler)
    save_mlp_model(mlp_model, scaler, CELL_TYPE)


if __name__ == "__main__":
    main()
