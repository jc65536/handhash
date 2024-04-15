import os
import random

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from clean_hands import clean_2


def rotate_and_zoom_out(image, angle, scale=1.0):
    # Get the image size and the center coordinates
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Calculate the sine and cosine of the rotation angle (in radians)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Calculate the new bounding dimensions of the image
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account the translation
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]

    # Rotate the original image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    return rotated_image


import cv2
import numpy as np


def tilt_image(image, direction='vertical', tilt=0, tilt_degree=10):
    """
    Apply a perspective tilt to an image.

    Parameters:
    - image: The image to be processed (as a numpy array).
    - direction: 'vertical' or 'horizontal' indicating the tilt direction.
    - tilt: 'forward', 'backward' for vertical, and 'left', 'right' for horizontal.
    - tilt_degree: The intensity of the tilt. Higher values result in more dramatic perspective changes.

    Returns:
    - The tilted image as a numpy array.
    """
    rows, cols = image.shape[:2]

    # Define points to map from and to
    if direction == 'vertical':
        if tilt == 0:
            # Tilt forward
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
            pts2 = np.float32([[0 + tilt_degree, 0], [cols - 1 - tilt_degree, 0], [0, rows - 1], [cols - 1, rows - 1]])
        elif tilt == 1:
            # Tilt backward
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
            pts2 = np.float32([[0, 0], [cols - 1, 0], [0 + tilt_degree, rows - 1], [cols - 1 - tilt_degree, rows - 1]])
    elif direction == 'horizontal':
        if tilt == 0:
            # Tilt left
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
            pts2 = np.float32([[0, 0 + tilt_degree], [cols - 1, 0], [0, rows - 1 - tilt_degree], [cols - 1, rows - 1]])
        elif tilt == 1:
            # Tilt right
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
            pts2 = np.float32([[0, 0], [cols - 1, 0 + tilt_degree], [0, rows - 1], [cols - 1, rows - 1 - tilt_degree]])
    else:
        raise ValueError("Invalid direction or tilt.")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    tilted_image = cv2.warpPerspective(image, matrix, (cols, rows))

    return tilted_image


def augment_image(image, num_of_augmentations=10):
    """
    Apply 2 or 3 random augmentations to an image and return a list of augmented images.
    Ensures that each image is generated with a combination of unique augmentations.

    Parameters:
    - image: The image to be processed (as a numpy array).
    - num_of_augmentations: Number of augmented images to generate.

    Returns:
    - List of augmented images.
    """
    augmented_images = []

    # Define augmentation functions
    augmentation_options = [
        ('flip_left_right', lambda img: cv2.flip(img, 1)),
        ('flip_up_down', lambda img: cv2.flip(img, 0)),
        ('rotate_and_zoom_in', lambda img: rotate_and_zoom_out(img, random.randint(-10, 10), random.uniform(0.8, 1.2))),
        ('tilt', lambda img: tilt_image(img, random.choice(['vertical', 'horizontal']), random.randint(0, 1),
                                        random.randint(5, 20))),
    ]

    for _ in range(num_of_augmentations):
        # Randomly choose to apply 2 or 3 augmentations
        num_augmentations_to_apply = random.choice([2, 3])

        # Select unique augmentation functions for this image
        selected_augmentations = random.sample(augmentation_options, num_augmentations_to_apply)

        augmented_image = image.copy()
        for _, augmentation_func in selected_augmentations:
            augmented_image = augmentation_func(augmented_image)

        augmented_images.append(augmented_image)

    return augmented_images


def process_image(image):
    # Process the image to find hand landmarks.
    results = hands.process(image)

    # Draw hand landmarks on the image.
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp.solutions.drawing_utils.draw_landmarks(
        #         image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        pass
    else:
        print("Failed to find hands!")
        return None
    conf = results.multi_handedness[0].classification[0].score
    if conf > 0.85:
        cleaned_values = clean_2(results.multi_hand_landmarks[0])
    else:
        print("Too low confidence value!", round(conf, 4))
        cleaned_values = None
    return cleaned_values



if __name__ == "__main__":
    photo_dir = "our_dataset/"
    fail_photo_dir = "our_dataset_fail/"
    result_dir = "our_dataset_npy/"
    result_photo_dir = "our_dataset_photos/"
    all_files = os.listdir(photo_dir)

    # Initialize MediaPipe Hand solution.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5)


    for folder in os.listdir(photo_dir):
        for file in os.listdir(f"{photo_dir}/{folder}"):
            print(folder, file)

            image_path = os.path.join(photo_dir, folder, file)
            image = cv2.imread(image_path)

            # rotated_image = rotate_and_zoom_out(image, angle=10, scale=0.8)
            augment_images = augment_image(image, num_of_augmentations=10)

            # Convert the image color from BGR to RGB.

            name_modifiers = ["", "f1", "f2", "f3"] + ['aug'+str(i) for i in range(1, 11)]

            flipped_image = cv2.flip(image, 1)
            upside_down_image = cv2.flip(image, 0)
            flipped_upside_down_image = cv2.flip(flipped_image, 0)
            images = [image, flipped_image, upside_down_image, flipped_upside_down_image]
            images += augment_images
            cleaned_values = []

            for img in images:
                # cv2.imshow("Random Augmentation", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = process_image(img)
                cleaned_values.append(result)

            for modifier_idx, cleaned_val in enumerate(cleaned_values):
                file_name = file.split('.')[0]
                if cleaned_val is None:
                    # cv2.imwrite(f'{fail_photo_dir}/{file_name}{name_modifiers[modifier_idx]}.jpg', img)
                    continue

                if not os.path.exists(f"{result_dir}/{folder}"):
                    os.makedirs(f"{result_dir}/{folder}")
                # if not os.path.exists(f"{result_photo_dir}/{folder}"):
                #     os.makedirs(f"{result_photo_dir}/{folder}")

                np.save(f"{result_dir}/{folder}/{file_name}{name_modifiers[modifier_idx]}.npy", cleaned_val)

                # Convert the image color back from RGB to BGR.
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f"{result_photo_dir}/{folder}_{file_name}{name_modifiers[modifier_idx]}.jpg", image)

    hands.close()

