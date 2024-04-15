import json
import os
from mlp_model import create_mlp_model
import numpy as np

class HierarchyInferencer:
    def __init__(self, hierarchy_dir, log_str="", model_name=""):
        self.hierarchy_dir = hierarchy_dir
        self.label_dict = None
        self.name = hierarchy_dir.split('/')[-1]
        self.top_model = None
        self.log_str = log_str
        self.model_name = model_name
        self.lower_models = {}
        self.class_names = ['1', '2', '3', '3_hook', '4', '5', '6', '7', '8', 'a',
                      'b', 'b_nothumb', 'b_thumb', 'cbaby', 'obaby', 'by', 'c', 'd', 'e', 'f', 
                      'f_open', 'fly', 'fly_nothumb', 'g', 'h', 'h_hook', 'h_thumb', 'i', 'jesus', 'k',
                      'l_hook', 'middle', 'm', 'n', 'o', 'index', 'index_flex', 'index_hook', 'pincet', 'ital',
                      'ital_thumb', 'ital_nothumb', 'ital_open', 'r', 's', 'write', 't', 'v', 'v_flex',
                      'v_hook', 'v_thumb', 'w', 'y', 'ae', 'ae_thumb', 'pincet_double', 'obaby_double', 'm2', 'jesus_thumb', 
                      'No Gesture']         
        self.input_dims = 63
        self.load_hierarchy()

    def get_mlp_model(self, input_shape, num_classes, load_path):
        model = create_mlp_model(input_shape, num_classes, 0.005, 37)
        model.load_weights(load_path)
        return model

    def load_hierarchy(self):
        all_files = os.listdir(self.hierarchy_dir)
        label_dict_files = [i for i in all_files if i.startswith('label_dict')]
        if self.log_str == "":
            self.log_str = label_dict_files[0].split('_')[-1].split('.')[0]
        
        # load label_dict        
        with open(self.hierarchy_dir + '/label_dict_' + self.log_str + '.json') as f:
            self.label_dict = json.load(f)
        
        top_model_folder = self.hierarchy_dir + '/top_model_' + self.log_str
        all_top_models = os.listdir(top_model_folder)
        if self.model_name == "":
            self.model_name = all_top_models[0]
        self.top_model_dir = top_model_folder + '/' + self.model_name
        self.top_model = self.get_mlp_model(self.input_dims, len(self.label_dict.keys()), self.top_model_dir)

        # get lower model dirs
        self.lower_model_dirs = {}
        lower_models_folder = self.hierarchy_dir + '/lower_models_' + self.log_str
        all_lower_models = os.listdir(lower_models_folder)

        for index in range(len(self.label_dict.keys())):
            to_load = str(index) + '___' + self.model_name
            if to_load in all_lower_models:
                self.lower_model_dirs[index] = lower_models_folder + '/' + to_load
            else:
                continue
        for k, v in self.lower_model_dirs.items():
            self.lower_models[k] = self.get_mlp_model(self.input_dims, len(self.label_dict[str(k)]), v)


    def infer(self, example):
        label = -1
        top_pred = self.top_model.predict(example)
        top_pred_class = top_pred.argmax()
        # print("top pred class: ", top_pred_class)
        # print("label dict: ", self.label_dict)
        # print("label_dict[str(top_pred_class)]: ", self.label_dict[str(top_pred_class)])
        if top_pred_class in self.lower_models:
            # print("Using lower model.")
            lower_pred = self.lower_models[top_pred_class].predict(example)
            lower_pred_class = lower_pred.argmax()
            # print("lower pred: ", lower_pred_class)
            label = self.label_dict[str(top_pred_class)][lower_pred_class]
        else:
            label = self.label_dict[str(top_pred_class)][0]
        return self.class_names[label]


if __name__ == '__main__':
    inferencer = HierarchyInferencer('hierarchies/confusion_based')
    example = np.load("our_dataset_npy/1/0.npy")
    example = example.flatten()
    example = example.reshape(1, -1)
    # print(inferencer.infer(example))
    # print(inferencer.top_model)
    # print(inferencer.lower_models)