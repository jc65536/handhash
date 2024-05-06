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
cap = cv2.VideoCapture("margherita.mp4")



with open('handshape2ids_rm_right.pkl', 'rb') as f:
    handshape2right = pkl.load(f)
with open('handshape2ids_rm_left.pkl', 'rb') as f:
    handshape2left = pkl.load(f)

right2handshape = {v: k for k, v in handshape2right.items()}
left2handshape = {v: k for k, v in handshape2left.items()}


learning_rate = ExponentialDecay(
    initial_learning_rate=0.005,
    decay_steps=27,
    decay_rate=0.99,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Load the model structure and weights separately
model = create_mlp_model(63, 60, 0.005, 37) 
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.load_weights('mlp_True_2.000000e-3_bs4.h5')  # Assuming the weights are saved in this file.

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

classification_names = ['1', '2', '3', '3_hook', '4', '5', '6', '7', '8', 'a',
                      'b', 'b_nothumb', 'b_thumb', 'cbaby', 'obaby', 'by', 'c', 'd', 'e', 'f', 
                      'f_open', 'fly', 'fly_nothumb', 'g', 'h', 'h_hook', 'h_thumb', 'i', 'jesus', 'k',
                      'l_hook', 'middle', 'm', 'n', 'o', 'index', 'index_flex', 'index_hook', 'pincet', 'ital',
                      'ital_thumb', 'ital_nothumb', 'ital_open', 'r', 's', 'write', 't', 'v', 'v_flex',
                      'v_hook', 'v_thumb', 'w', 'y', 'ae', 'ae_thumb', 'pincet_double', 'obaby_double', 'm2', 'jesus_thumb', 
                      'No Gesture']   


def classify_our_model(hand_landmarks):
    cleaned_values = clean_2(hand_landmarks)
    val = cleaned_values
    val = val.reshape(1, -1)
    prediction = hand_model.predict(val)
    confidence = np.max(prediction)   
    predicted_class = np.argmax(prediction, axis=1)
    ret_string = classification_names[predicted_class[0]]
    top_3_conf = np.sort(prediction[0])[-3:][::-1]
    top_3_conf = np.sum(top_3_conf)

    top_2_conf = np.sort(prediction[0])[-2:][::-1]
    top_2_conf = np.sum(top_2_conf)

    if confidence > 0.9:
        return ret_string
    elif top_2_conf > 0.95:
        ret_string = ""
        for i in range(2):
            ret_string += " " + classification_names[np.argsort(prediction[0])[-2:][::-1][i]]
    elif top_3_conf > 0.95:
        ret_string = ""
        for i in range(3):
            ret_string += " " + classification_names[np.argsort(prediction[0])[-3:][::-1][i]]
    elif confidence < 0.85:
        return "No Gesture" # + str(round(confidence, 4)) 
    return ret_string
    

    return classification_names[predicted_class[0]]
    
    # get preds higher than 0.3
    top_preds = np.argsort(prediction[0])[-3:][::-1]
    top_preds = top_preds[prediction[0][top_preds] > 0.3]
    res_string = ""
    for pred in top_preds:
        res_string += classification_names[pred] + " "
    # predict_str = classification_names[predicted_class[0]]
    return res_string # + "_" + str(round(confidence, 4))

hierarchy_inferencer = HierarchyInferencer('hierarchies/confusion_based', log_str="20240314-150911")
def classify_hierarchy(hand_landmarks):
    cleaned_values = clean_2(hand_landmarks)
    val = cleaned_values
    val = val.reshape(1, -1)
    return hierarchy_inferencer.infer(val)


def classify_shape(hand_type, hand_landmarks, world_landmarks):
    hand_landmarks = changeToList(hand_landmarks)
    world_landmarks = changeToList(world_landmarks)
    if hand_type != "Left":
        return ":)"
    hand_landmarks.extend(world_landmarks)
    val = hand_landmarks
    # print(val)
    val = np.array(val)
    val = val.reshape(1, -1)
    # print(val.shape)   
    
    prediction = hand_model.predict(val)
    # return ":)"

    predicted_class = np.argmax(prediction, axis=1)
    predict_str = left2handshape[predicted_class[0]]
    return predict_str


if not cap.isOpened():
    print("Cannot open camera")
    exit()



while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video file reached.")
        break

    # Convert the BGR image to RGB, as MediaPipe needs RGB images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and extract hand landmarks.
    results = hands.process(rgb_frame)

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
    
        """
        relevant_exemplar = exemplar_classifier.exemplars["right"][0].copy()
        relevant_exemplar += [0.25, 0.25, 0]
        for idx, lm in enumerate(hand_landmarks.landmark):
            lm.x, lm.y, lm.z = relevant_exemplar[idx]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        """    

    """
    if results.multi_hand_landmarks:
        for i in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[i]
            hand_type = results.multi_handedness[i].classification[0].label
            # pred_str = classify_hierarchy(hand_landmarks)
            pred_str = classify_our_model(hand_landmarks)
            # pred_str = classify_exemplar(hand_type.lower(), hand_landmarks)
            
            if hand_type == "Right":
                cv2.putText(frame, pred_str, (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, pred_str, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    """
                
    # Display the resulting frame with landmarks.
    cv2.imshow('Video Output with Hand Landmarks', frame)

    # Press 'q' to exit the loop before the video ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
