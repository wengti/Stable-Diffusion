# This is a code rewritten based on https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py

from pathlib import Path
import os
import numpy as np
import cv2
from tqdm import tqdm


# Do take note that the sequence of this label_list matters
# As the mask images have areas where there are overlapping
# The latter label will overwrite the regions if it overlaps with the earliest mask
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = Path('./data/CelebAMask-HQ/CelebAMask-HQ-mask-anno')
folder_save = Path('./data/CelebAMask-HQ/CelebAMask-HQ-mask')
img_num = 30000

if not folder_save.is_dir():
    folder_save.mkdir(parents = True,
                      exist_ok = True)

folder_idx = 0
for num in tqdm(range(img_num)):
    
    # Create an empty mask
    output_mask = np.zeros((512, 512))  # Size of the mask
    
    
    # The mask images are separated into folders where each folder has masks for 2000 original images in sequence
    if num % 2000 == 0 and num != 0:
        folder_idx += 1
        
    for color_idx, label in enumerate(label_list):
        file_name =  folder_base / str(folder_idx) / f"{str(num).rjust(5,'0')}_{label}.png"
        if os.path.exists(file_name):
            mask = cv2.imread(file_name, 0) # 512 x 512, either 0 or 255
            output_mask[mask == 255] = color_idx + 1 # Smallest color label is 1 as 0 is reserved for black
        save_name = folder_save / f"{num}.png"
        cv2.imwrite(save_name, output_mask)