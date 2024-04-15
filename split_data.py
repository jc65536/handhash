import os
import shutil

chris_photo = 'handshapes_chris'
jt_photos = 'handshapes_jt'
our_photos_online_dirs = ['jt_handshapes_online', 'chris_handshapes_online', 'shanmu_handshapes_online', 'processed_original_hand_photos', 'shanmu_handshapes_named']

result_dir = 'our_dataset'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

test = 0

def split_chris_photos(path):
    files = os.listdir(path)
    files = sorted(files)
    for file in files:
        group = int(file.split('_')[0]) + 1
        os.makedirs(f"{result_dir}/{group}", exist_ok=True)
        num_files_in_result_dir = len(os.listdir(f"{result_dir}/{group}"))
        shutil.copyfile(f"{path}/{file}", f"{result_dir}/{group}/{num_files_in_result_dir}.{file.split('.')[-1]}")

def split_jt_photos(path):
    files = os.listdir(path)
    for file in files:
        group = int(file.split('(')[1].split(')')[0])
        os.makedirs(f"{result_dir}/{group}", exist_ok=True)
        num_files_in_result_dir = len(os.listdir(f"{result_dir}/{group}"))
        shutil.copyfile(f"{path}/{file}", f"{result_dir}/{group}/{num_files_in_result_dir}.{file.split('.')[-1]}")

def process_online(path):
    files = os.listdir(path)
    for file in files:
        group = int(file.split('_')[0])
        os.makedirs(f"{result_dir}/{group}", exist_ok=True)
        num_files_in_result_dir = len(os.listdir(f"{result_dir}/{group}"))
        shutil.copyfile(f"{path}/{file}", f"{result_dir}/{group}/{num_files_in_result_dir}.{file.split('.')[-1]}")


split_chris_photos(chris_photo)
split_jt_photos(jt_photos)

for path in our_photos_online_dirs:
    process_online(path)