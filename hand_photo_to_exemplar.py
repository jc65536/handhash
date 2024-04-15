import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from clean_hands import clean

photo_dir = "hand_photos/"

all_files = os.listdir(photo_dir)

right_hand_files = all_files[:len(all_files)//2]
left_hand_files = all_files[len(all_files)//2:]

# Initialize MediaPipe Hand solution.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5)


def process_hand_files(paths, hand_side):
    # Create a directory to store the processed images
    processed_dir = "processed_exemplars/" + hand_side
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.exists(f"exemplars/{hand_side}"):
        os.makedirs(f"exemplars/{hand_side}")

    paths = sorted(paths)

    for i, file in enumerate(paths):
        # Skip if it's not an image file.
        print(i, hand_side, file)

        # Read the image file using OpenCV.
        image_path = os.path.join(photo_dir, file)
        image = cv2.imread(image_path)

        # Convert the image color from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks.
        results = hands.process(image)

        # Draw hand landmarks on the image.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cleaned_values = clean(results.multi_hand_landmarks[0])
        # save the cleaned values

        # Save the processed image.
        two_digit_number = str(i).zfill(2)

        np.save(f"exemplars/{hand_side}/{two_digit_number}.npy", cleaned_values)

        # Convert the image color back from RGB to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"processed_exemplars/{hand_side}/{two_digit_number}.jpg", image)

# Process right and left hand images.
process_hand_files(right_hand_files, 'right')
process_hand_files(left_hand_files, 'left')

# Release the MediaPipe resources.
hands.close()

