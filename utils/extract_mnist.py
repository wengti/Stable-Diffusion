from pathlib import Path
import _csv as csv
import numpy as np
import cv2
import argparse

def extract_mnist(csv_fname, save_dir):
    
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)
    
    with open(csv_fname, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):

            if idx == 0:
                continue
            
            img = np.zeros((28*28))
            img[:] = list(map(int, row[1:]))
            img = img.reshape((28,28))
            
            class_dir = save_dir / row[0]
            if not class_dir.is_dir():
                class_dir.mkdir(parents = True,
                                exist_ok = True)
                
            file = class_dir / f'{idx}.png'
            
            cv2.imwrite(file, img)
            if idx % 1000 == 0:
                print(f"[INFO] {idx} images have been saved into {save_dir}.")


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    
    # Create arguments
    parser.add_argument('--csv_fname', type = str, help = 'The csv file that consists of MNIST data')
    parser.add_argument('--save_dir', type = str, help = 'The path to the folder to save the MNIST data')
    
    # Pass the arguments
    args = parser.parse_args()
    
    # Use the arguments
    csv_fname = args.csv_fname
    save_dir = args.save_dir
    
    # Call the functions
    extract_mnist(csv_fname = csv_fname,
                  save_dir = save_dir)
    

            
