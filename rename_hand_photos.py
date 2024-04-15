import os
import shutil


hand_photos_dir = 'shanmu_handshapes'
results_dir = 'shanmu_handshapes_named'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)



files = os.listdir(hand_photos_dir)
# 1_57 PM_1.jpg, and 1_57 PM_10.jpg, causes issues with sorting

# for file in files:
#     x = file
#     print(int(x[:4].replace('_', ''))*10 + int(x.split('_')[2].split('.')[0]))


files = sorted(files, key=lambda x: int(x[:4].replace('_', ''))*10 + int(x.split('_')[2].split('.')[0]))

grouped_files = [files[i:i+4] for i in range(0, len(files), 4)]
for group_idx, group in enumerate(grouped_files):
    for file_num, file in enumerate(group):
        res_name = f"{group_idx+1}_{file_num}.{file.split('.')[-1]}"
        shutil.copyfile(f"{hand_photos_dir}/{file}", f"{results_dir}/{res_name}")
    
