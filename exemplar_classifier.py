import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


class ExemplarClassifier:
    def __init__(self) -> None:
        self.names = ['1', '2', '3', '3_hook', '4', '5', '6', '7', '8', 'a',
                      'b', 'b_nothumb', 'b_thumb', 'cbaby', 'obaby', 'by', 'c', 'd', 'e', 'f', 
                      'f_open', 'fly', 'fly_nothumb', 'g', 'h', 'h_hook', 'h_thumb', 'i', 'jesus', 'k',
                      'l_hook', 'middle', 'm', 'n', 'o', 'index', 'index_flex', 'index_hook', 'pincet', 'ital',
                      'ital_thumb', 'ital_nothumb', 'ital_open', 'r', 's', 'write', 'spoon', 't', 'v', 'v_flex',
                      'v_hook', 'v_thumb', 'w', 'y', 'ae', 'ae_thumb', 'pincet_double', 'obaby_double', 'm2', 'jesus_thumb']      
          
        self.exemplars = {}
        for hand in ["left", "right"]:
            exemplar_dir = f"exemplars/{hand}"
            self.exemplars[hand] = []
            for file in os.listdir(exemplar_dir):
                exemplar = np.load(os.path.join(exemplar_dir, file))
                self.exemplars[hand].append(exemplar)
            self.exemplars[hand] = np.array(self.exemplars[hand])

    def classifyl2(self, hand_landmarks, hand):
        min_distance_index = self.get_closest_idx(hand_landmarks, hand)
        name = self.names[min_distance_index]
        return name

    def get_closest_exemplar(self, hand_landmarks, hand):
        min_distance_index = self.get_closest_idx(hand_landmarks, hand)
        return self.exemplars[hand][min_distance_index]

    def get_closest_idx(self, hand_landmarks, hand):
        related_exemplars = self.exemplars[hand]
        l2_distances = np.linalg.norm(related_exemplars - hand_landmarks, axis=(1, 2))
        min_distance_index = np.argmin(l2_distances)
        print("L2 distances:", l2_distances)
        return min_distance_index


    def classify(self, hand_landmarks, hand):
        if np.isnan(hand_landmarks).any():
            return "NaN"
        
        related_exemplars = self.exemplars[hand]
        # Flatten the landmark coordinates into vectors
        vectorized_exemplars = related_exemplars.reshape(related_exemplars.shape[0], -1)
        vectorized_hand_landmarks = hand_landmarks.flatten().reshape(1, -1)
        cosine_similarities = cosine_similarity(vectorized_exemplars, vectorized_hand_landmarks)
        max_similarity_index = np.argmax(cosine_similarities)
        
        for i in range(len(cosine_similarities)):
            print(f"{self.names[i]}: {cosine_similarities[i]}")
        
        
        name = self.names[max_similarity_index] if max_similarity_index < len(self.names) else str(max_similarity_index)
        return name