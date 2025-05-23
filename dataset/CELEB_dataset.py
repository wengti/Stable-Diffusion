from torch.utils.data import Dataset
from pathlib import Path
from utils.diffusion_utils import load_latent
from PIL import Image
from torchvision import transforms
import cv2
import os
import numpy as np
import torch
import random

class CELEB_dataset(Dataset):
    """
    A custom dataset for CELEB
    
    Args: 
        directory (Path): \n
        Setup by giving a directory that has the following format: \n
            | \n 
            |_ celeba-caption (Text file that describes the images) \n
            |_ CelebA-HQ-img (Original images) \n
            |_ CelebAMask-HQ-mask (Processed mask images, obtained via utils.create_celeb_mask) \n
            |_ CelebAMask-HQ-mask-anno (Preprcessed mask images)\n
        
        config (dict): A config with .yaml extension loaded by yaml.safe_load \n
        result_dir (path): Expecting a directories that contain a directory named 'latent_data' and within that directories consist of multiple .pkl files that store dicts in the following format {image_path: latent_data}. The 'latent_data' folder can be obtained using tools.infer_VQVAE \n
        use_latent (boolean): Decide whether to output latent_data or images \n
        model_latent_mask_flag (boolean): Must be set to True, if the dataset is loading images in the original space, but will be compressed later on into latent spatial size. It is important to make sure mask in the latent spatial size this way, otherwise will remain in original image spatial. \n
                
        
      Returns:
          img (float tensor): 1x28x28, in the range of -1 to 1 \n
          latent_data (float tensor) : z_channels x h x w
          condition (dict): keys include {'image' and/or 'text'} \n
                            'image' -> return the mask image in the shape of mask_channels x img_size x img_size \n
                            'text' -> return a string that is pre-tokenized, but is a description for that image \n
               
    """
    
    def __init__(self, directory, config, result_dir = None, use_latent = False, model_latent_mask_flag = False):
        self.directory = Path(directory)
        img_directory = self.directory / 'CelebA-HQ-img'
        
        self.config = config
        self.im_size = config['im_size']
        self.im_channels = config['im_channels']
        
        # Variables that are only needed if image condition is used
        if 'image' in config['condition']:
            self.mask_channels = config['mask_channels']
            self.mask_size = config['mask_size']
        
        # The path list that consists of all images
        self.img_path_list = list(img_directory.glob('*.jpg'))
            
        # Load latent data if input consists of result_dir and use_latent is True
        self.result_dir = result_dir
        self.use_latent = use_latent
        
        self.model_latent_mask_flag = model_latent_mask_flag
        
        # Only apply load_latent functions to get the dictionary of combined {img_path : latent_data} if latent_data is needed
        if self.result_dir is not None and self.use_latent:
            self.latent_data_dict = load_latent(result_dir = self.result_dir)
        
        
    
    def __len__(self):
        return len(self.img_path_list)
    
    def load_image(self, index):
        img =  Image.open(self.img_path_list[index])
        return img
    
    
    
    def load_mask(self, img_name, outpt_mask_size):
        """
        Load the mask image in the shape of mask_channels x img_size x img_size
        
        Args:
            img_name (str): Name of the image (for instance: for 9.png, expected to be '9')
            
        Returns:
            output_mask (float tensor): Mask image in the shape of mask_channels x img_size x img_size or mask_channels x latent_size x latent_size
        """

        mask_file = self.directory / 'CelebAMask-HQ-mask' / f"{img_name}.png"
        mask_img = cv2.imread(mask_file, 0) # 512 x 512 
        
        # Each channel in the mask representing one feature from the mask with a value of 1
        output_mask = np.zeros((self.mask_channels, self.mask_size, self.mask_size)) # mask_channels x 512 x 512
        for idx in range(self.mask_channels): 
            output_mask[idx, mask_img == (idx+1)] = 1
        
        output_mask = torch.from_numpy(output_mask) # Convert to torch tensor
        
        # Interpolate using nearest mode (default) to masks_channels x im_size x im_size
        # The function requests input in the shape of B x C x in0 x in1.... ink and the size parameter in the size of out0 x out1.... outk
        output_mask = torch.nn.functional.interpolate(output_mask[None, ...], size = (outpt_mask_size, outpt_mask_size))
        output_mask = output_mask[0] # Remove the uncessary batch dim, ensuring output as mask_channels x im_size x im_size
        
        return output_mask

    
    def load_text(self, img_name):
        """
        Load the captions into a list and randomly select one of them to be returned
        
        Args:
            img_name (str): Name of the image (for instance: for 9.png, expected to be '9')
        
        Returns:
            output_caption (str): A random selected captions used to describe the image
        """
        
        # Compile all available captions into one list
        text_file = self.directory / 'celeba-caption' / f'{img_name}.txt'
        captions = []
        with open(text_file, 'r') as f:
            for line in f.readlines():
                captions.append(line.strip()) # strip removes any leading and trailing white spaces
        
        # Randomly select one from the list
        #random.sample returns a list, even if getting only one sample, hence [0] is necessary
        output_caption = random.sample(captions, k=1)[0]
        return output_caption
        
        
    
    def __getitem__(self, index):
        
        # Decide whether to load latent_data or the original image
        if self.result_dir is not None and self.use_latent:
            latent_data = self.latent_data_dict[self.img_path_list[index]][0]
            outpt_mask_size = latent_data.shape[-1] # Obtain the expected output mask shape for the data
        else:
            img = self.load_image(index)
            simple_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((self.im_size, self.im_size)),
                                                   transforms.CenterCrop((self.im_size, self.im_size))])
            img = simple_transform(img) # CxHxW, range from 0 to 1
            img = (2*img) - 1 # CxHxW, range from 0 to 1
            
            outpt_mask_size = img.shape[-1] # Obtain the expected output mask shape for the data
        
        
        condition = {}
        if len(self.config['condition']) > 0:
            img_name = os.path.split(self.img_path_list[index])[1].split('.')[0] # A number in str format, which is the name of the image
            
            if 'image' in self.config['condition']:
                
                # This flag needs to be set to True when mask images are used, but the dataset is set to output original images instead of latent data
                # By default, the size of the mask will follow the shape of the output image / latent data by the dataset (determined by use_latent flag in config)
                # However, a special edge case would be when original image is output by the dataset, but the model will compress the original image before training
                # Now the input in latent spatial size, but the mask is still in image spatial size
                if self.model_latent_mask_flag:
                    outpt_mask_size = self.config['im_size'] // (2 ** (sum(self.config['ac_down_sample'])))
                
                mask = self.load_mask(img_name = img_name,
                                      outpt_mask_size = outpt_mask_size) # Load the mask image for the corresponding image, in shape of mask_channels x img_size x img_size
                condition['image'] = mask # Assign it to the dictionary
            
            if 'text' in self.config['condition']:
                text = self.load_text(img_name = img_name)
                condition['text'] = text
                
        
        # Decide whether to load latent_data or the original image
        if self.result_dir is not None and self.use_latent:
            return latent_data, condition
        else:
            return img, condition
            
    

