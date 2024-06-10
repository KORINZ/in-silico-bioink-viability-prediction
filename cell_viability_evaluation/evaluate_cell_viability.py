import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from typing import Sequence

IMAGE_PATH = r"APX100_4X/alg_i1g_10T12_Multichannel.png"

# Initial Parameters Placeholder
parameters = {
    "MAXIMUM_CELL_AREA": None,
    "MINIMUM_CELL_EROSION_AREA": None,
    "EROSION_ITERATIONS": None,
    "MINIMUM_LIVE_CELL_AREA": None,
    "MINIMUM_DEAD_CELL_AREA": None,
    "MORPH_ITERATIONS": 1,
    "MIN_CIRCULARITY": 0.40,
    "GREEN_THRESHOLD": 10,
    "COLOR_INTENSITY_THRESHOLD": None,
}

# Colors
LIVE_COLOR = (0, 255, 0)  # Green
DEAD_COLOR = (255, 0, 0)  # Red

# Contour Infills
LIVE_CELL_CONTOUR_FILL = -1
DEAD_CELL_CONTOUR_FILL = -1

def get_json_path(image_path: str) -> None | str:
    """Get the JSON path based on the image path."""
    json_path = "/".join(image_path.split("/")[:-2] + [image_path.split("/")[-3] + "_parameters.json"])
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}.")
        return None
    return json_path

def get_json_parameters(json_path: str) -> dict[str, float | int]:
    """Print and update the parameters based on the JSON file."""
    with open(json_path) as json_file:
        json_parameters = json.load(json_file)
        parameters.update(json_parameters)
    return json_parameters

def update_parameters_based_on_image_path(image_path: str, print_output: bool = False) -> None:
    """Update the parameters based on the image path."""

    if "10X" in image_path and "APX100" in image_path:
        parameters.update(
            {
                "MAXIMUM_CELL_AREA": 800,
                "MINIMUM_CELL_EROSION_AREA": 30,
                "EROSION_ITERATIONS": 2,
                "MINIMUM_LIVE_CELL_AREA": 10,
                "MINIMUM_DEAD_CELL_AREA": 10,
                "COLOR_INTENSITY_THRESHOLD": 120,
            }
        )
    elif "4X" in image_path and "APX100" in image_path:
        parameters.update(
            {
                "MAXIMUM_CELL_AREA": 800,
                "MINIMUM_CELL_EROSION_AREA": 100,
                "EROSION_ITERATIONS": 2,
                "MINIMUM_LIVE_CELL_AREA": 20,
                "MINIMUM_DEAD_CELL_AREA": 20,
                "COLOR_INTENSITY_THRESHOLD": 40,
            }
        )
    else:
        raise ValueError(f"Image path not recognized: {image_path}")
    if print_output:
        print("JSON file not found. Parameters updated based on image path.")
    

def load_rgb_image(image_path, print_output: bool = False) -> np.ndarray:
    "Load an image from a file path, convert it to RGB format, and alter intensities."
    
    # Update parameters based on the image path
    update_parameters_based_on_image_path(image_path, print_output=print_output)

    image = cv2.imread(image_path)
    
    if image is None:
        # Extract the base path without the percentage part
        base_path = "_".join(image_path.split("_")[:-1])
        
        # Find the image path with the matching base path
        matching_path = None
        directory = os.path.dirname(image_path)
        for path in os.listdir(directory):
            if base_path in os.path.join(directory, path).replace("\\", "/"):
                matching_path = os.path.join(directory, path)
                break
        raise ValueError(f"Image not found at {image_path}. \n\nDid you mean this image? \n{matching_path}")
    
        
    elif "_scaled" in image_path:
        raise ValueError(f"Scaled image cannot be processed.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image


def remove_low_intensity_regions(image, intensity_threshold) -> np.ndarray:
    "Remove low intensity regions from the image."
    low_intensity_mask = np.all(image < intensity_threshold, axis=-1)
    image[low_intensity_mask] = [0, 0, 0]
    return image


def preprocess_image(image) -> Sequence[np.ndarray]:
    "Preprocess the image to remove low intensity regions and find contours."
    
    image_with_removed_low_intensity_regions = remove_low_intensity_regions(
        image.copy(), parameters["COLOR_INTENSITY_THRESHOLD"]
    )

    gray = cv2.cvtColor(image_with_removed_low_intensity_regions, cv2.COLOR_RGB2GRAY)

    # Temporarily removed Otsu's thresholding
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(
        gray, cv2.MORPH_OPEN, kernel, iterations=parameters["MORPH_ITERATIONS"]
    )

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask that will hold the processed contours
    processed_mask = np.zeros_like(opening)

    oversize_contours_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > parameters["MINIMUM_CELL_EROSION_AREA"]:
            oversize_contours_count += 1
            # Create a temporary mask for each contour
            temp_mask = np.zeros_like(opening)
            cv2.drawContours(temp_mask, [cnt], -1, (255, 255, 255), -1)

            # Apply erosion to the contour on the temporary mask
            temp_mask = cv2.erode(
                temp_mask, kernel, iterations=parameters["EROSION_ITERATIONS"]
            )

            # Add the processed contour to the main mask
            processed_mask = cv2.bitwise_or(processed_mask, temp_mask)
        else:
            # For contours smaller than the minimum area, draw them as they are
            cv2.drawContours(processed_mask, [cnt], -1, (255, 255, 255), -1)

    # Find contours on the processed maskW
    processed_contours, _ = cv2.findContours(
        processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # print(f"Number of contours applied erosion: {oversize_contours_count}")

    return processed_contours


def is_valid_contour(cnt, image) -> bool:
    "Check if a contour is touching the boundary, is too large, or not circular enough, with special circularity check for smaller areas."
    
    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    # Calculate contour area
    area = cv2.contourArea(cnt)
    # Calculate perimeter
    perimeter = cv2.arcLength(cnt, True)
    # Calculate circularity, avoiding division by zero
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0

    # Check if the contour is within the image bounds
    within_bounds = (
        x > 0 and y > 0 and x + w < image.shape[1] and y + h < image.shape[0]
    )
    # Check if the area is within the allowed maximum
    within_max_area = area <= parameters["MAXIMUM_CELL_AREA"]
    # Apply circularity check only if area is less than or equal to the minimum erosion area
    circularity_check = circularity >= parameters["MIN_CIRCULARITY"]

    return within_bounds and within_max_area and circularity_check


def draw_and_count_cells(
    image, contours, use_ellipse_contours
) -> tuple[np.ndarray, int, int, int]:
    "Draw contours and count the number of live and dead cells."
    
    new_background = np.zeros_like(image)

    live_count, dead_count, total_count = 0, 0, 0

    for cnt in contours:
        if is_valid_contour(cnt, image):
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
            mean_val = cv2.mean(image, mask=mask)

            cell_color = (
                LIVE_COLOR
                if mean_val[1] > mean_val[0] + parameters["GREEN_THRESHOLD"]
                and mean_val[1] > mean_val[2] + parameters["GREEN_THRESHOLD"]
                else DEAD_COLOR
            )

            # If the red channel intensity is high, the cell is dead
            if mean_val[0] > 60:
                cell_color = DEAD_COLOR

            cnt_area = cv2.contourArea(cnt)

            # Draw live cells in green
            if (
                cell_color == LIVE_COLOR
                and cnt_area > parameters["MINIMUM_LIVE_CELL_AREA"]
            ):
                if (
                    use_ellipse_contours and len(cnt) >= 5
                ):  # Minimum of 5 points required to fit an ellipse
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(
                        new_background, ellipse, cell_color, LIVE_CELL_CONTOUR_FILL
                    )
                else:
                    cv2.drawContours(
                        new_background,
                        [cnt],
                        -1,
                        cell_color,
                        LIVE_CELL_CONTOUR_FILL,
                    )
                live_count += 1

            # Draw dead cells in red
            elif (
                cell_color == DEAD_COLOR
                and cnt_area > parameters["MINIMUM_DEAD_CELL_AREA"]
            ):
                if use_ellipse_contours and len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(
                        new_background, ellipse, cell_color, DEAD_CELL_CONTOUR_FILL
                    )
                else:
                    cv2.drawContours(
                        new_background,
                        [cnt],
                        -1,
                        cell_color,
                        DEAD_CELL_CONTOUR_FILL,
                    )
                dead_count += 1

    total_count = live_count + dead_count

    return new_background, total_count, live_count, dead_count


def calculate_cell_viability(live_count, total_count) -> float:
    "Calculate the cell viability."
    
    return round(live_count / total_count * 100, 2)


def plot_and_print_result(image, new_background) -> None:
    "Plot the original and processed images and print the cell viability."
    
    plt.figure(figsize=(18, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(new_background)
    plt.title("Contour Mask")
    plt.grid(True)

    # Calculate cell viability
    cell_viability = calculate_cell_viability(live_count, total_count)

    plt.suptitle(
        f'{image_path.replace("data/", "").replace("_sorted", "")}',
        fontsize=18,
    )
    plt.text(
    0.5,
    0.90,
    f"Cell Viability: {cell_viability}%",
    fontsize=20,
    color="black",
    transform=plt.gcf().transFigure,
    ha="center",
)
    plt.text(
        0.5,
        0.86,
        f"Live Cells: {live_count}",
        fontsize=20,
        color="green",
        transform=plt.gcf().transFigure,
        ha="center",
    )
    plt.text(
        0.5,
        0.82,
        f"Dead Cells: {dead_count}",
        fontsize=20,
        color="red",
        transform=plt.gcf().transFigure,
        ha="center",
    )
    
    plt.subplots_adjust(wspace=-0.8)

    print(f"Total Number of Cells: {total_count}")
    print(f"Number of Live Cells: {live_count}")
    print(f"Number of Dead Cells: {dead_count}")
    print(f"Cell Viability: {cell_viability}%")
    plt.tight_layout()
    plt.show()


def load_scale_bar_image(image_path) -> np.ndarray:
    "Load the scaled image from the same directory if it exists."
    
    ext = os.path.splitext(image_path)[-1].lower()
    base_path = image_path[: -len(ext)]
    scaled_path = base_path + "_scaled" + ext

    if not os.path.exists(scaled_path):
        raise ValueError(f"Image not found at {scaled_path}")
    else:
        print(f"Scale bar image found at {scaled_path}")

    image = cv2.imread(scaled_path)

    if image is None:
        raise ValueError(f"Image not found at {scaled_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    image_path = IMAGE_PATH.replace("\\", "/")
    current_folder_path = "/".join(image_path.split("/")[:-1])
    image = load_rgb_image(image_path, print_output=True)

    contours = preprocess_image(image)

    new_background, total_count, live_count, dead_count = draw_and_count_cells(
        image,
        contours,
        use_ellipse_contours=True,  # Set to False to draw original contours
    )

    try:
        image = load_scale_bar_image(image_path)
    except ValueError:
        pass

    plot_and_print_result(image, new_background)
