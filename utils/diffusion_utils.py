import os
from pathlib import Path
import pickle
import torch

def load_latent(result_dir):
    
    """
    Combine multiple content of multiple .pkl files into one dictionary that has the following format -> {image_path: latent_data}
    
    Args:
        result_dir (path): Expecting a directories that contain a directory named 'latent_data' and within that directories consist of
                           multiple .pkl files that store dicts in the following format {image_path: latent_data}. The 'latent_data' folder
                           can be obtained using tools.infer_VQVAE
    
    Returns:
        latent_data (dict): A dictionaries that consist of all the combined latent_data into one dict in the following format -> {image_path: latent_data}
                       
    """
    
    result_dir = Path(result_dir)
    latent_file = result_dir / 'latent_data'
    if not os.path.exists(latent_file):
        raise FileNotFoundError(f"[WARNING] The required latent file cannot be found. Please check.")
    
    latent_path = list(latent_file.glob('*.pkl'))
    latent_data = {}
    for path in latent_path:
        with open(path, 'rb') as f:
            latent_temp = pickle.load(f)
            for key in latent_temp.keys():
                latent_data[key] = latent_temp[key]
    
    return latent_data





def drop_class_condition(class_one_hot, class_drop_prob, device):
    """
    Randomly drop one hot class embedding
    
    Args:
        class_one_hot (int tensor): Expected on cuda, in the shape of (B, num_classes), in the format of one_hot \n
        class_drop_prob (float) : The threshold that decides whether to drop the one hot class embedding \n
        device (str): 'cuda ' or 'device' \n
    
    Returns:
        class_one_hot (int tensor): Expected on cuda, in the shape of (B, num_classes), in the format of one_hot
    """
    
    if class_drop_prob > 0:
        drop_mask = torch.zeros(class_one_hot.shape[0], 1).to(device) # Create a zero array that match the shape of the class_one_hot
        drop_mask = drop_mask.uniform_(0,1) > class_drop_prob # Replace each zero with a random uniformly distributed number and compare with the drop probabilty, return True if more than threshold, else 0
        return class_one_hot * drop_mask # Mask out those that are below the threshold
        
    else:
        return class_one_hot
    

def drop_image_condition(mask_image, image_drop_prob, device):
    """
    Randomly drop mask images condition
    
    Args:
        mask image (int tensor): Expected on cuda, in the shape of (B, mask_ch, img_h, img_w) \n
        image_drop_prob (float) : The threshold that decides whether to drop the mask image \n
        device (str): 'cuda ' or 'device' \n
    
    Returns:
        mask image (int tensor): Expected on cuda, in the shape of (B, mask_ch, img_h, img_w)
    """
    
    if image_drop_prob > 0:
        drop_mask = torch.zeros(mask_image.shape[0], 1, 1, 1).to(device) # Create a zero array that match the shape of the mask_image
        drop_mask = drop_mask.uniform_(0,1) > image_drop_prob # Replace each zero with a random uniformly distributed number and compare with the drop probabilty, return True if more than threshold, else 0
        return mask_image * drop_mask # Mask out those that are below the threshold
    else:
        return mask_image
    

def drop_text_condition(context_embedding, text_drop_prob, empty_text_embed, device):
    
    """
    Randomly drop text condition
    
    Args:
        context_embedding (float tensor): Expected on cuda, in the shape of (B, L, context_dim) \n
        text_drop_prob (float) : The threshold that decides whether to drop the context embedding \n
        empty_text_embed (float tensor): Replacement if a context embedding were to be dropped, expected on cuda, in the shape of (B, L, context_dim) \n
        device (str): 'cuda ' or 'device' \n
    
    Returns:
        context_embedding (float tensor): Expected on cuda, in the shape of (B, L, context_dim)
    """
    
    if text_drop_prob > 0:
        drop_mask = torch.zeros(context_embedding.shape[0]).to(device)
        drop_mask = drop_mask.uniform_(0,1) < text_drop_prob
        context_embedding[drop_mask, :, :] = empty_text_embed[0] # Empty_text_embed has the shape of 1 x 77 x context_dim, therefore [0] is needed
        
        return context_embedding
    else:
        return context_embedding
    
    
    
