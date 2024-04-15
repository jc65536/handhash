import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle as pkl
from scipy.spatial.transform import Rotation as R
from exemplar_classifier import ExemplarClassifier
from clean_hands import clean, clean_2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
from mlp_model import create_mlp_model
from hierarchy_inferencer import HierarchyInferencer


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


learning_rate = ExponentialDecay(
    initial_learning_rate=0.005,
    decay_steps=27,
    decay_rate=0.99,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Load the model structure and weights separately
model = create_mlp_model(63, 22, 0.005, 37)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('mlp_True_2.000000e-3_bs32.h5')  # Assuming the weights are saved in this file.

hand_model = model

exemplar_classifier = ExemplarClassifier()

def changeToList(landmarks):
    results = []
    for landmark in landmarks:
        results.append(landmark.x)
        results.append(landmark.y)
        results.append(landmark.z)
    return (np.round(results, 4)).tolist()



def classify_exemplar(hand_type, hand_landmarks):
    cleaned_values = clean(hand_landmarks)
    return exemplar_classifier.classify(cleaned_values, hand_type)

classification_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g/q', 'h/u', 'i', 'k/p', 'l', 'm', 'n', 'o', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']


def classify_our_model(hand_landmarks):
    cleaned_values = clean_2(hand_landmarks)
    val = cleaned_values
    val = val.reshape(1, -1)
    prediction = hand_model.predict(val)
    confidence = np.max(prediction)
    # if confidence < 0.4:
    #     return "<Unknown>" # + str(round(confidence, 4))
    predicted_class = np.argmax(prediction, axis=1)
    # get preds higher than 0.3
    # top_preds = np.argsort(prediction[0])[-3:][::-1]
    # top_preds = top_preds[prediction[0][top_preds] > 0.3]
    # res_string = ""
    # for pred in top_preds:
    #     res_string += classification_names[pred] + " "
    predict_str = classification_names[predicted_class[0]]
    return predict_str # + "_" + str(round(confidence, 4))


if not cap.isOpened():
    print("Cannot open camera")
    exit()



while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the BGR image to RGB, as MediaPipe needs RGB images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and extract hand landmarks.
    results = hands.process(rgb_frame)

    """
    # Draw hand landmarks.

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert hand landmarks to an array format for processing
            normalized_landmarks = clean(hand_landmarks, for_display=True)
            print(normalized_landmarks.shape)
            
            # Convert normalized landmarks back to the required format for drawing
            for idx, lm in enumerate(hand_landmarks.landmark):
                lm.x, lm.y, lm.z = normalized_landmarks[idx]

            # Draw the normalized hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
        relevant_exemplar = exemplar_classifier.exemplars["right"][0].copy()
        relevant_exemplar += [0.25, 0.25, 0]
        for idx, lm in enumerate(hand_landmarks.landmark):
            lm.x, lm.y, lm.z = relevant_exemplar[idx]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)    
    """

    if results.multi_hand_landmarks:
        for i in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[i]
            x = hand_landmarks.landmark[0].x
            y = hand_landmarks.landmark[0].y
            hand_type = results.multi_handedness[i].classification[0].label
            # pred_str = classify_hierarchy(hand_landmarks)
            pred_str = classify_our_model(hand_landmarks)
            # pred_str = classify_exemplar(hand_type.lower(), hand_landmarks)
            if hand_type == 'Right':
                cv2.putText(frame, pred_str, (int(frame.shape[0]*x+40), int(frame.shape[0]*y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, pred_str, (int(frame.shape[0]*x), int(frame.shape[0]*y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # if hand_type == "Right":
            #     cv2.putText(frame, pred_str, (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # else:
            #     cv2.putText(frame, pred_str, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame with landmarks.
    cv2.imshow('Webcam Output with Hand Landmarks', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()