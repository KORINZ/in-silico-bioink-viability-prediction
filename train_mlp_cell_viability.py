from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns

sns.set_context("talk", font_scale=1.5)
sns.set_style("ticks")


CELL_TYPE = "UE7T-13"

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

elif CELL_TYPE == "UE7T-13":
    marker = "D"
    color = "purple"
    color_map = sns.dark_palette("purple", as_cmap=True)
else:
    raise ValueError("Invalid CELL_TYPE.")


def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data_filtered = data[
        ["wall_shear_stress_kPa", "residence_time_ms", "relative_cell_viability"]
    ]
    return data_filtered


def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def separate_features(scaled_data):
    x = scaled_data[:, :2]
    Y = scaled_data[:, 2:]
    return x, Y


def create_param_grid():# -> dict[str, Any]:
    if CELL_TYPE == "HUEhT-1":
        return {
            "hidden_layer_sizes": [(64, 64), (80, 80)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(0.575, 0.75, 10)],
            "learning_rate": ["constant"],
        }
    elif CELL_TYPE == "HeLa":
        return {
            "hidden_layer_sizes": [(80, 80), (128, 128)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(1.0, 1.5, 15)],
            "learning_rate": ["constant"],
        }
    elif CELL_TYPE == "10T12":
        return {
            "hidden_layer_sizes": [(64, 64), (80, 80)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(0.15, 1, 5)],
            "learning_rate": ["constant"],
        }
    elif CELL_TYPE == "UE7T-13":
        return {
            "hidden_layer_sizes": [(128, 128)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [i for i in np.linspace(0.2, 0.4, 15)],
            "learning_rate": ["constant"],
        }
    else:
        raise ValueError("Invalid CELL_TYPE.")


def train_mlp_cv(x, Y):
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


def scale_mse_scores(mse_scores, scaler):
    if scaler.var_ is not None:
        mse_scores_original_space = mse_scores * scaler.var_[2:]
        rmse_scores_original_space = np.sqrt(mse_scores_original_space)
        return rmse_scores_original_space
    else:
        raise ValueError(
            "scaler.var_ is None. Make sure scaler.fit() is called on valid data."
        )


def plot_3d_surface(model, title, scaler, data, rsme, save_fig=False):
    wall_shear_stress_kPa_mean, residence_time_ms_mean = scaler.mean_[:2]
    wall_shear_stress_kPa_std, residence_time_ms_std = np.sqrt(scaler.var_[:2])
    feature_mean = scaler.mean_[2]
    feature_std = np.sqrt(scaler.var_[2])

    wall_shear_stress_kPa_range = np.linspace(
        1, 5, 200
    )
    residence_time_ms_range = np.linspace(
        100, 700, 200
    )
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
    ax.set_zlabel("Cell Viability (%)" , fontweight="bold")
    
    # padding for labels
    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 18
    ax.zaxis.labelpad = 18
    
    ax.set_yticks([100, 250, 400, 550, 700])
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
    ax.set_xticklabels(['1.0', '2.0', '3.0', '4.0', '5.0'])
    
    ax.set_zticks([i for i in range(0, 101, 20)])
    
    ax.tick_params(axis='both', which='major')

    ax.set_title(
        f"{title.replace("10T12", "10T1/2")}",
    )
    ax.set_zlim(0, 100)
    ax.view_init(25, 55)

    plt.legend(edgecolor="black", fancybox=False, bbox_to_anchor=(0.5, 1.02), loc="upper center", handlelength=0.5)

    if save_fig:
        os.makedirs(f"mlp/images/{title}", exist_ok=True)
        plt.savefig(
            f"mlp/images/{title}/{title}_3d_surface_plot.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.55,
        )

    plt.show()





def save_mlp_model(mlp_model, scaler, cell_type):
    os.makedirs(f"mlp/model/{cell_type}", exist_ok=True)
    with open(f"mlp/model/{cell_type}/{cell_type}_mlp.pkl", "wb") as file:
        pickle.dump(mlp_model, file)
    with open(f"mlp/model/{cell_type}/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)


def plot_actual_vs_predicted_cv(model, x, Y, scaler):
    y_pred = cross_val_predict(model, x, Y, cv=20)

    # Scale back the predictions and actual values to original space
    y_pred_original = y_pred * np.sqrt(scaler.var_[2]) + scaler.mean_[2]
    Y_original = Y * np.sqrt(scaler.var_[2]) + scaler.mean_[2]
    
    # max value of y_pred_original is 100, so we need to clip Y_original to 100 as well
    y_pred_original = np.clip(y_pred_original, 0, 100)

    # Calculate metrics
    r2 = r2_score(Y_original, y_pred_original)
    mae = mean_absolute_error(Y_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(Y_original, y_pred_original))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Y_original, y_pred_original, alpha=0.5, edgecolor="k", s=55, color=color, zorder=100, marker=marker)
    
    min_val = np.round(min(min(original, pred) for original, pred in zip(Y_original, y_pred_original)), -1)
    
    ax.plot([min_val, 100], [min_val, 100], 'k--', lw=2)
    ax.set_xlabel("Actual (%)", fontweight="bold")
    ax.set_ylabel("Predicted (%)" , fontweight="bold")
    ax.set_title(f"{CELL_TYPE}")
    
    # Create a box with error evaluation results
    textstr = "\n".join([
        f"R-squared: {r2:.3f}",
        f"MAE: {mae:.3f}%",
        f"RMSE: {rmse:.3f}%"
    ])
    props = dict(facecolor='white', alpha=1, edgecolor='black')
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
    
    ax.grid(True)
    
    ax.set_ylim(min_val, 100)
    ax.set_xlim(min_val, 100)
    
    # Use a step size of 10 for the ticks
    ax.set_xticks(np.arange(min_val, 100 + 1, 10))
    ax.set_yticks(np.arange(min_val, 100 + 1, 10))
    
    
    # Print out data points that has greater than 8% difference between actual and predicted values
    data = pd.read_csv(f"test/{CELL_TYPE}_test_all.csv")
    for i, row in data.iterrows():
        if abs(row["relative_cell_viability"] - y_pred_original[i]) > 8:
            print(f"Actual: {row['relative_cell_viability']:.2f}, Predicted: {y_pred_original[i]:.2f}, Difference: {abs(row['relative_cell_viability'] - y_pred_original[i]):.2f}, Flow rate: {row['flow_rate_uL_per_s']:.2f}, Concentration: {row['concentration_wv']}")
    
    plt.tight_layout()
    plt.savefig(f"mlp/images/{CELL_TYPE}/{CELL_TYPE}_actual_vs_predicted.png", dpi=600, bbox_inches="tight", pad_inches=0.2)
    
    plt.show()
    
def main():
    data = preprocess_data(f"test/{CELL_TYPE}_test_all.csv")
    scaled_data, scaler = scale_data(data)
    x, Y = separate_features(scaled_data)

    mlp_model, mse_score, best_params = train_mlp_cv(x, Y[:, 0])
    rmse_scores_original_space = scale_mse_scores([mse_score], scaler)

    name = "relative_cell_viability"
    print(f"Best parameters for Multi-Layer Perceptron model for {name}: {best_params}")
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
