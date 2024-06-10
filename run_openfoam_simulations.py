import pickle
import numpy as np
import pandas as pd
import os
import time
import subprocess

# Constants
SIMULATION_FOLDER_PATH = r"openfoam_sample_code"
SIMULATION_RESULTS_CSV_PATH = rf"{SIMULATION_FOLDER_PATH}_results.csv"
NEEDLE_LENGTH = 20e-3
WEDGE_ANGLE = 5
RHO = 1e3
ETA_INF = 1e-3
NU_INF = ETA_INF / RHO

FLOW_RATE_uL_PER_S = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.5,
    4.0,
    5.0,
]
CONCENTRATION = [
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.25,
    4.5,
    4.75,
    5.0,
    5.25,
    5.5,
    5.75,
    6.0,
    6.25,
    6.5,
    6.75,
    7.0,
]
TEMPERATURE = [
    4,
    7.5,
    10,
    12.5,
    15,
    17.5,
    20,
    22.5,
    25,
    27.25,
    30,
    32.25,
    35,
    37,
    37.5,
    40,
    45,
]


def convert_flow_rate_to_wedge_flow_rate(flow_rate_uL_per_s) -> float:
    """Converts the flow rate in µL/s to the wedge flow rate in m³/s."""

    flow_rate_m3_per_s = flow_rate_uL_per_s * 1e-9
    wedge_flow_rate_m3_per_s = flow_rate_m3_per_s * (WEDGE_ANGLE / 360)

    return round(wedge_flow_rate_m3_per_s, 16)


def get_cross_power_law_parameters_from_svr_model(
    concentration, temperature
) -> tuple[float, float, float]:
    """Predicts the cross power law parameters using the SVR models."""

    with open(os.path.join("..", "model", "svr_eta_0_model_log.pkl"), "rb") as file:
        svr_eta_0_model = pickle.load(file)
    with open(os.path.join("..", "model", "svr_m_model_log.pkl"), "rb") as file:
        svr_m_model = pickle.load(file)
    with open(os.path.join("..", "model", "svr_n_model_log.pkl"), "rb") as file:
        svr_n_model = pickle.load(file)

    # Load the scaler
    with open(os.path.join("..", "model", "scaler_log.pkl"), "rb") as file:
        scaler = pickle.load(file)

    # Create a DataFrame with the input features and dummy columns
    input_data = pd.DataFrame(
        {
            "concentration": [concentration],
            "temperature": [temperature],
            "eta_0": [0],
            "m": [0],
            "n": [0],
        }
    )

    # Scale the input features
    scaled_features = scaler.transform(input_data)[:, :2]

    # Predict the values using the loaded models
    eta_0_pred_scaled = svr_eta_0_model.predict(scaled_features)
    m_pred_scaled = svr_m_model.predict(scaled_features)
    n_pred_scaled = svr_n_model.predict(scaled_features)

    # Inverse transform the predictions to original scale
    eta_0_pred_log = eta_0_pred_scaled * np.sqrt(scaler.var_[2]) + scaler.mean_[2]
    m_pred_log = m_pred_scaled * np.sqrt(scaler.var_[3]) + scaler.mean_[3]
    n_pred_log = n_pred_scaled * np.sqrt(scaler.var_[4]) + scaler.mean_[4]

    # Inverse transform the predictions from logarithmic space to the original space
    eta_0_pred = np.exp(eta_0_pred_log)
    m_pred = np.exp(m_pred_log)
    n_pred = np.exp(n_pred_log)

    nu_0_pred = eta_0_pred / RHO

    return nu_0_pred[0].round(8), m_pred[0].round(6), n_pred[0].round(6)


def update_initial_flow_rate_file(flow_rate_uL_per_s) -> None:
    """Updates the initial flow rate in the '0/U' file."""

    wedge_flow_rate_m3_per_s = convert_flow_rate_to_wedge_flow_rate(flow_rate_uL_per_s)

    with open("0/U", "r+") as f:
        lines = f.readlines()
        lines[28] = f"            value           {wedge_flow_rate_m3_per_s};\n"

        lines = lines[:55]

        f.seek(0)
        f.writelines(lines)
        f.truncate()
    print(
        f"Updated initial flow rate for flow rate {flow_rate_uL_per_s} µL/s. (wedge flow rate: {wedge_flow_rate_m3_per_s} m³/s)"
    )


def update_physical_properties_and_momentum_transport_files(
    concentration, temperature, nu_0, m, n
) -> None:
    """Updates the physical properties and momentum transport files with the predicted cross power law parameters."""

    with open("constant/physicalProperties", "r+") as f:
        lines = f.readlines()
        lines[18] = f"nu              {nu_0};\n"

        lines = lines[:21]

        f.seek(0)
        f.writelines(lines)
        f.truncate()

    with open("constant/momentumTransport", "r+") as f:
        lines = f.readlines()
        lines[23] = f" nuInf {NU_INF};\n"
        lines[24] = f" m {m};\n"
        lines[25] = f" n {n};\n"

        lines = lines[:30]

        f.seek(0)
        f.writelines(lines)
        f.truncate()

    print(
        f"\nUpdated physical properties and momentum transport files for concentration {concentration}% (w/v) and temperature {temperature}°C."
    )
    print(f"nu_0: {nu_0}, m: {m}, n: {n}")


def get_wall_shear_stress() -> float:
    """Returns the wall shear stress from the 'data.log' file."""

    with open("data.log", "r") as f:
        lines = f.readlines()
        wall_shear_stress = float(lines[2].split()[-1])

    return wall_shear_stress


def get_pressure_drop() -> float:
    """Returns the pressure drop from the 'data.log' file."""

    with open("data.log", "r") as f:
        lines = f.readlines()
        pressure_drop = float(lines[1].split()[-1])

    return pressure_drop


def get_average_velocity() -> float:
    """Returns the average velocity from the 'data.log' file."""

    with open("data.log", "r") as f:
        lines = f.readlines()
        average_velocity = float(lines[0].split()[-1])

    return average_velocity


def append_simulation_results_to_csv(
    concentration,
    temperature,
    eta_0,
    m,
    n,
    flow_rate_uL_per_s,
    wall_shear_stress,
    pressure_drop,
    average_velocity,
    residence_time,
) -> None:
    """Appends the simulation results to the CSV file."""

    if not os.path.exists(os.path.join("..", SIMULATION_RESULTS_CSV_PATH)):
        with open(os.path.join("..", SIMULATION_RESULTS_CSV_PATH), "w") as f:
            f.write(
                "concentration_wv,temperature_C,eta_0,m,n,flow_rate_uL_per_s,wall_shear_stress_kPa,pressure_drop_kPa,average_velocity_m_s,residence_time_ms\n"
            )

    with open(os.path.join("..", SIMULATION_RESULTS_CSV_PATH), "a") as f:
        f.write(
            f"{concentration},{temperature},{eta_0},{m},{n},{flow_rate_uL_per_s},{wall_shear_stress},{pressure_drop},{average_velocity},{residence_time}\n"
        )


def main() -> None:
    """Runs the simulations for all the combinations of flow rate, concentration, and temperature."""

    # Change to the simulation folder
    os.chdir(SIMULATION_FOLDER_PATH)

    num_simulations = len(FLOW_RATE_uL_PER_S) * len(CONCENTRATION) * len(TEMPERATURE)
    current_simulation = 0

    for flow_rate_uL_per_s in FLOW_RATE_uL_PER_S:
        for concentration in CONCENTRATION:
            for temperature in TEMPERATURE:

                current_simulation += 1
                start_time = time.time()

                nu_0, m, n = get_cross_power_law_parameters_from_svr_model(
                    concentration, temperature
                )

                update_physical_properties_and_momentum_transport_files(
                    concentration, temperature, nu_0, m, n
                )
                update_initial_flow_rate_file(flow_rate_uL_per_s)

                # Run the simulation and supress the output
                print(f"Running simulation ({current_simulation}/{num_simulations})...")
                subprocess.check_call(["./Allrun.sh"], stdout=subprocess.DEVNULL)
                # ! Run dos2unix Allrun.sh and Clear.sh if file not found error occurs

                wall_shear_stress = get_wall_shear_stress()
                pressure_drop = get_pressure_drop()
                average_velocity = get_average_velocity()
                residence_time = int(round(NEEDLE_LENGTH / average_velocity * 1e3, -1))

                print(
                    f"Wall shear stress: {wall_shear_stress} kPa; Pressure drop: {pressure_drop} kPa; Average velocity: {average_velocity} m/s; Residence time: {residence_time} ms"
                )

                eta_0 = nu_0 * RHO

                append_simulation_results_to_csv(
                    concentration,
                    temperature,
                    eta_0,
                    m,
                    n,
                    flow_rate_uL_per_s,
                    wall_shear_stress,
                    pressure_drop,
                    average_velocity,
                    residence_time,
                )

                try:
                    subprocess.check_call(["./Clear.sh"])
                except subprocess.CalledProcessError:
                    pass

                elapsed_time = time.time() - start_time

                print(f"Simulation completed in {round(elapsed_time)} seconds.")


if __name__ == "__main__":
    main()
