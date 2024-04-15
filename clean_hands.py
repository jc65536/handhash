import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_finger_lengths(hand_landmarks):
    # Define landmark indices for the thumb, middle finger, and pinky.
    thumb_indices = [0, 1, 2, 3, 4]
    middle_finger_indices = [0, 9, 10, 11, 12]
    pinky_indices = [0, 17, 18, 19, 20]

    # Initialize variables to store the lengths.
    thumb_length = 0
    middle_finger_length = 0
    pinky_length = 0

    # Calculate the thumb length by summing the distances between consecutive joints.
    for i in range(len(thumb_indices) - 1):
        thumb_length += np.linalg.norm(hand_landmarks[thumb_indices[i]] - hand_landmarks[thumb_indices[i + 1]])

    # Calculate the middle finger length by summing the distances between consecutive joints.
    for i in range(len(middle_finger_indices) - 1):
        middle_finger_length += np.linalg.norm(hand_landmarks[middle_finger_indices[i]] - hand_landmarks[middle_finger_indices[i + 1]])

    # Calculate the pinky length by summing the distances between consecutive joints.
    for i in range(len(pinky_indices) - 1):
        pinky_length += np.linalg.norm(hand_landmarks[pinky_indices[i]] - hand_landmarks[pinky_indices[i + 1]])

    return thumb_length, middle_finger_length, pinky_length


def normalize_hand(numpy_hand, for_display=False):
    # Translate hand so the wrist is at the origin
    wrist = numpy_hand[0, :]
    numpy_hand -= wrist
    
    # Define points for plane computation
    p1, p2, p3 = numpy_hand[0], numpy_hand[5], numpy_hand[17]

    # Compute plane normal
    normal = np.cross(p2 - p1, p3 - p1)
    normal_unit = normal / np.linalg.norm(normal)

    # Align normal with Z-axis
    rotation_axis = np.cross(normal_unit, [0, 0, 1])
    rotation_angle = np.arccos(np.dot(normal_unit, [0, 0, 1]))
    rotation = R.from_rotvec(rotation_angle * rotation_axis / np.linalg.norm(rotation_axis))
    rotated_landmarks = rotation.apply(numpy_hand)

    # Scale landmarks based on finger lengths
    # Assume calculate_finger_lengths is a defined function that returns lengths
    thumb_length, middle_finger_length, pinky_length = calculate_finger_lengths(rotated_landmarks)
    total_length = thumb_length + middle_finger_length + pinky_length
    scaled_landmarks = rotated_landmarks / total_length

    # Align points 0 and 5 with the X-axis
    p0_rotated, p5_rotated = scaled_landmarks[0], scaled_landmarks[9]
    v05_rotated = p5_rotated - p0_rotated
    v05_rotated[2] = 0  # Ignore Z component
    angle_to_x_axis = np.arctan2(v05_rotated[0], v05_rotated[1])
    rotation_correction = R.from_euler('z', angle_to_x_axis + np.pi)
    final_rotated_landmarks = rotation_correction.apply(scaled_landmarks)

    # Center the hand in the desired frame
    if for_display:
        final_rotated_landmarks += [0.5, 0.5, 0]

    return final_rotated_landmarks


def hand_to_numpy(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])


def clean(hand_landmarks, for_display=False):
    numpy_hand = hand_to_numpy(hand_landmarks)
    return normalize_hand(numpy_hand, for_display)


def clean_2(landmarks):

    landmarks = hand_to_numpy(landmarks)
    landmarks = landmarks - landmarks[0]
    
    # Step 1 and Step 2 from the previous function, aligning the triangle to the Z-axis
    v1 = landmarks[5] - landmarks[0]
    v2 = landmarks[17] - landmarks[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal, z_axis)
    angle = np.arccos(np.dot(normal, z_axis))
    
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
        rotation = R.from_rotvec(angle * axis)
    else:
        rotation = R.from_rotvec([0, 0, 0])
    
    landmarks = rotation.apply(landmarks)
    
    # Additional rotation to align landmarks[0] to landmarks[9] with the X-axis
    # First, compute the direction vector from landmarks[0] to landmarks[9]
    direction = landmarks[5] - landmarks[0]
    direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
    x_axis = np.array([1, 0, 0])
    
    # Determine the axis and angle for the additional rotation
    axis_2 = np.cross(direction, x_axis)
    angle_2 = np.arccos(np.dot(direction, x_axis))
    
    if np.linalg.norm(axis_2) != 0:
        axis_2 = axis_2 / np.linalg.norm(axis_2)
        rotation_2 = R.from_rotvec(angle_2 * axis_2)
    else:
        rotation_2 = R.from_rotvec([0, 0, 0])
    
    # Apply the additional rotation
    rotated_landmarks = rotation_2.apply(landmarks)
    
    # Force landmarks[9] on the X-axis (zero out the y and z components) due to precision issues.
    rotated_landmarks[5][1] = 0
    rotated_landmarks[5][2] = 0

    palm_size = np.linalg.norm(rotated_landmarks[0] - rotated_landmarks[5]) + \
        np.linalg.norm(rotated_landmarks[5] - rotated_landmarks[9]) + \
        np.linalg.norm(rotated_landmarks[9] - rotated_landmarks[13]) + \
        np.linalg.norm(rotated_landmarks[13] - rotated_landmarks[17]) + \
        np.linalg.norm(rotated_landmarks[17] - rotated_landmarks[0])

    rotated_landmarks = rotated_landmarks / palm_size / 2


    # rotated_landmarks += [0.5, 0.5, 0]

    return rotated_landmarks