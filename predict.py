import sys
import pickle as pkl
sys.path.append('..')  # To allow imports from parent directory

from tensorflow.keras.models import load_model
import numpy as np


with open('data/handshape2ids_rm_right.pkl', 'rb') as f:
    handshape2right = pkl.load(f)
with open('data/handshape2ids_rm_left.pkl', 'rb') as f:
    handshape2left = pkl.load(f)



def main():
    # Load model
    model = load_model('../models/my_mlp_model.h5')

    # Example input (ensure it's preprocessed as your model expects)
    example_input = np.random.rand(1, 784)  # Example shape for flattened MNIST images

    # Make prediction
    prediction = model.predict(example_input)
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predicted class: {predicted_class[0]}")


if __name__ == '__main__':
    main()
