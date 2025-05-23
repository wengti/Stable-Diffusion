from pathlib import Path
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from utils.diffusion_utils import load_latent

def find_classes(directory):
    directory = Path(directory)
    
    class_names = sorted(entry.name for entry in os.scandir(directory))
    if not class_names:
        raise FileNotFoundError("[WARNING] The provided directory in find_classes is not valid. Please check.")
    
    class_to_label = {}
    for idx, name in enumerate(class_names):
        class_to_label[name] = idx
    
    return class_names, class_to_label

class MNIST_dataset(Dataset):
    """
    A custom dataset for MNIST
    
    Args: 
        directory (Path): \n
        Setup by giving a directory that has the following format: \n
            | \n
            |_ 0 \n
               |_ 6709.png \n
            | \n
            |_ 1 \n
               |_ 109.png \n
        
        config (dict): A config with .yaml extension loaded by yaml.safe_load \n
        result_dir (path): Expecting a directories that contain a directory named 'latent_data' and within that directories consist of multiple .pkl files that store dicts in the following format {image_path: latent_data}. The 'latent_data' folder can be obtained using tools.infer_VQVAE \n
        use_latent (boolean): Decide whether to output latent_data or images \n
        model_latent_mask_flag (boolean): Must be set to True, if the dataset is loading images in the original space, but will be compressed later on into latent spatial size. It is important to make sure mask in the latent spatial size this way, otherwise will remain in original image spatial. \n        
        
      Returns:
          img (float tensor): 1x28x28, in the range of -1 to 1 \n
          latent_data (float tensor) : z_channels x h x w
          condition (dict): keys include {'class'} -> return the integer class label \n
               
    """
    
    def __init__(self, directory, config, result_dir = None, use_latent = False, model_latent_mask_flag = False):
        directory = Path(directory)
        self.config = config
        self.im_size = self.config['im_size']
        
        self.img_path_list = list(directory.glob('*/*.png'))
        self.classes, self.class_to_label = find_classes(directory)
        self.result_dir = result_dir
        self.use_latent = use_latent
        
        # Only apply load_latent functions to get the dictionary of combined {img_path : latent_data} if latent_data is needed
        if self.result_dir is not None and self.use_latent:
            self.latent_data_dict = load_latent(result_dir = self.result_dir)
        
    def load_image(self, index):
        img = Image.open(self.img_path_list[index])
        return img
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        
        condition = {}
        if 'class' in self.config['condition']:
            class_name = self.img_path_list[index].parent.stem
            class_label = self.class_to_label[class_name]
            condition = {'class' : class_label}
        
        if self.result_dir is not None and self.use_latent:
            latent_data = self.latent_data_dict[self.img_path_list[index]][0]
            return latent_data, condition
        
        else:   
            img = self.load_image(index)
            simple_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((self.im_size,self.im_size))])
            img = simple_transform(img) # CxHxW, range from 0 to 1
            img = (2*img) - 1 # CxHxW, range from 0 to 1
            return img, condition
    
