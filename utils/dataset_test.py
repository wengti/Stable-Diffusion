from matplotlib import pyplot as plt
from pathlib import Path
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import argparse
from dataset.CELEB_dataset import CELEB_dataset
from dataset.MNIST_dataset import MNIST_dataset
import yaml
from torch.utils.data import DataLoader

def MNIST_dataset_test(data, config, save_plot_dir):
    
    """
    Verify if a MNIST dataset is created correctly
    
    Args:
        data (Dataset): a MNIST custom dataset \n
        config (dict): a config file that has been read using yaml.safe_load \n
        save_plot_dir (path): a directories that should be the following structures: './result/{task_name}/dataset_test'
    """
    
    randNum = torch.randint(0, len(data)-1, (9,))
    
    # Only plot the images if not using latent data
    if not data.use_latent:    
        for idx, num in enumerate(randNum):
            train_img, condition = data[num]
            
            train_img_plt = (train_img + 1) / 2
            train_img_plt = train_img_plt.permute(1,2,0)
            plt.subplot(3,3,idx+1)
            plt.imshow(train_img_plt, cmap='gray')
            plt.axis(False)
            
            # Only have access to the 'class' in condition if its mentioned in config
            if 'class' in config['condition']:
                train_label = condition['class']
                plt.title(f'Label: {train_label}')
        
        # Save the plotted grid images
        save_plot_dir = Path(save_plot_dir)
        if not save_plot_dir.is_dir():
            save_plot_dir.mkdir(parents = True,
                                exist_ok = True)
        save_plot_file = save_plot_dir / "dataset_test_plot.png"
        plt.tight_layout()
        plt.savefig(save_plot_file)
        print(f"The dataset test plot has been saved into {save_plot_file}.")
    
    # If using latent data, no plotting will be done
    else:
        num = randNum[0]
        train_img, condition = data[num]
        
    print(f"The number of images in the dataset: {len(data)}")
    print(f"The size of an image               : {train_img.shape}")
    print(f"The range of value in an image     : {train_img.min()} to {train_img.max()}")
    print(f"The available classes in the image : {data.classes}")


def CELEB_dataset_test(data, config, save_plot_dir):
    
    """
    Verify if a MNIST dataset is created correctly
    
    Args:
        data (Dataset): a CELEB custom dataset \n
        config (dict): a config file that has been read using yaml.safe_load \n
        save_plot_dir (path): a directories that should be the following structures: './result/{task_name}/dataset_test'
    """
    
    # Get the index of random samples (note: the index here is the index in img_path_list, not the name of the image)
    num_samples = 5
    randNum = torch.randint(0, len(data) - 1, (num_samples,))
    
    # Only plot the image (and its mask) if not using latent data
    if not data.use_latent:
        
        # Plot the original images
        train_img_out = torch.cat([data[num][0][None,...] for num in randNum], dim=0) # data[num][0] -> which image to be access
        train_img_out = (train_img_out + 1) / 2
        train_img_out_grid = make_grid(train_img_out, nrow = num_samples)
        train_img_out_grid_img = ToPILImage()(train_img_out_grid)
        
        # Save the plotted original images
        save_plot_dir = Path(save_plot_dir)
        if not save_plot_dir.is_dir():
            save_plot_dir.mkdir(parents = True,
                                exist_ok = True)
        save_plot_file = save_plot_dir / "dataset_test_plot.png"
        train_img_out_grid_img.save(save_plot_file)
        print(f"The dataset test plot has been saved into {save_plot_file}.")
        
        
    # Check if there are conditions that can be plotted as image
    if len(config['condition']) > 0:
        
        # Check if 'image' is in the config's condition
        if 'image' in config['condition']:
            
            # Since num in randNum =/= name of the image, a text file is created to save the name of the image for cross checking
            text = "Images that are tested: \n"
            for num in randNum:
                text += f"{data.img_path_list[num]} \n"
            
            with open(save_plot_dir / 'Images_index.txt', 'w') as f:
                f.write(text)
            
            # Plot original images against each of the mask

            train_mask_out = None 
            for idx in range(config['mask_channels']):
                # For each of tha image, go through the same channel (where each channel represent a mask feature)
                # data[num][1]['image'] -> to access the mask image (CxHxW), 
                # [idx] -> to select the particular channels, which reduce the size to (HxW)
                # Hence require 2 None to make it into B x C x H x W
                train_mask_out_temp = torch.cat([data[num][1]['image'][idx][None, None, ...] for num in randNum], dim = 0) 
                train_mask_out_temp = train_mask_out_temp.repeat(1,3,1,1) # Since concatenating with original images which have 3 channels, they are expanded into 3 channels too
                if train_mask_out == None:
                    train_mask_out = train_mask_out_temp
                else:
                    train_mask_out = torch.cat([train_mask_out, train_mask_out_temp], dim = 0)
            
            # if not use_latent -> image in image size and not model_latent_mask_flag -> mask in image size
            if (not data.use_latent and not config['model_latent_mask_flag']):
                train_mask_out_all = train_img_out
                train_mask_out_all = torch.cat([train_mask_out_all, train_mask_out], dim=0)
            else:
                train_mask_out_all = train_mask_out
            
            # Save the grid images
            train_mask_out_all_grid = make_grid(train_mask_out_all, nrow = num_samples)
            train_mask_out_all_grid_img = ToPILImage()(train_mask_out_all_grid)
        
            save_mask_file = save_plot_dir / "mask_test_plot.png"
            train_mask_out_all_grid_img.save(save_mask_file)
            print(f"The mask test plot has been saved into {save_mask_file}.")
            
            # Print the info (shape) for a mask
            num = randNum[0]
            mask = data[num][1]['image']
            print(f"The size of a mask image: {mask.shape}")
            print(f"The range of value in a mask image: {mask.min()} to {mask.max()}")
        
        
        # Check if 'text' in the config's condition
        if 'text' in config['condition']:
            
            # Collect the captions that are selected
            captions = f"Caption that are used to describe the images are: \n\n"
            for num in randNum:
                _, condition = data[num]
                captions += f"For {data.img_path_list[num]}: \n"
                captions += condition['text'] + "\n\n"
            
            # Save the selected captions into a text file
            caption_save_file = save_plot_dir / 'Images_caption.txt'
            with open(caption_save_file, 'w') as f:
                f.write(captions)
            print(f"The captions selected to describe the images have been saved into {caption_save_file}.")
    
    # For accessing other details in the image
    num = randNum[0]
    train_img, condition = data[num]
    
    print(f"The number of images in the dataset: {len(data)}")
    print(f"The size of an image               : {train_img.shape}")
    print(f"The range of value in an image     : {train_img.min()} to {train_img.max()}")
    

if __name__ == '__main__':
    
    # When the script is run, it will ask for the config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = 'The path to the config file in .yaml format')
    
    args = parser.parse_args()
    config_path = args.config
    
    # Read config
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Directories to save result
    result_dir = Path(f'./result/{config['task_name']}')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    # Directories to save plot (Reside within result_dir)
    save_plot_dir = result_dir / 'dataset_test'
    if not save_plot_dir.is_dir():
        save_plot_dir.mkdir(parents = True,
                            exist_ok = True)
    
    
    # Test for MNIST
    if config['dataset_type'] == 'MNIST':
        train_data = MNIST_dataset(directory = config['data_path'],
                                  config = config, 
                                  result_dir = None,
                                  use_latent = False)
        
        MNIST_dataset_test(data = train_data, 
                           config = config, 
                           save_plot_dir = save_plot_dir)
        
    # Test for CELEB
    elif config['dataset_type'] == 'CELEB':
        train_data = CELEB_dataset(directory = config['data_path'],
                                   config = config,
                                   result_dir = None,
                                   use_latent = False)
        
        CELEB_dataset_test(data = train_data, 
                           config = config, 
                           save_plot_dir = save_plot_dir)
        
# =============================================================================
#         data_loader = DataLoader(dataset = train_data,
#                                  batch_size = config['ldm_batch_size'],
#                                  shuffle = True,
#                                  num_workers = 4)
#         
# 
#         print(f"[INFO] The loaded data loader is as following: ")
#         train_img_batch, train_condition_batch = next(iter(data_loader))
#         print(f"Number of batches in the data loader: {len(data_loader)}")
#         print(f"Number of images in 1 batch         : {train_img_batch.shape[0]}")
#         print(f"Shape of an image                   : {train_img_batch[0].shape}")
#         if 'image' in config['condition']:
#             print(f"train_condition_batch (mask)        : {train_condition_batch['image'].shape}")
#         
# 
#         print("\n")
# =============================================================================

    
    
    