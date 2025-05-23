import argparse
import yaml
import torch
from pathlib import Path
from models.UNet import UNet
from tqdm import tqdm
from scheduler.linear_noise_scheduler import linear_noise_scheduler
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from models.VQVAE import VQVAE
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from dataset.CELEB_dataset import CELEB_dataset
import random
from einops import rearrange

def infer(config_path):
    
    # Read config file
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Result dir
    result_dir = Path(f"./result/{config['task_name']}/")
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    # Grid plot dir
    grid_plot_dir = result_dir / "Reverse Diffusion"
    if not grid_plot_dir.is_dir():
        grid_plot_dir.mkdir(parents = True,
                            exist_ok = True)
    
    # Load LDM model
    model_file = result_dir / 'LDM.pt'
    
    model = UNet(config = config).to(device)
    model.load_state_dict(torch.load(f = model_file,
                                     weights_only = True))
    
    # Create noise scheduler
    scheduler = linear_noise_scheduler(config = config,
                                       device = device)
    
    
    # Load autoencoder
    autoencoder_dir = Path(config['ac_save_path'])
    autoencoder_file = autoencoder_dir / 'VQVAE.pt'
    
    autoencoder = VQVAE(config = config).to(device)
    autoencoder.load_state_dict(torch.load(f = autoencoder_file,
                                           weights_only = True))
    
    
    
    
    
    
    
    ### Noise ###
    
    
    # Create random input noise 
    # Size -> determined by number of down sample executed
    inpt_size = config['im_size'] // (2 ** (sum(config['ac_down_sample'])))
    
    inpt_noise = torch.randn(config['num_samples'], config['z_channels'], inpt_size, inpt_size).to(device)
    out = inpt_noise
    
    
    
    
    
    
    ### Conditions ###
    
    
    # Classifier free guidance
    cf_guidance_value = config['cf_guidance']
    
    # Class conditions
    class_one_hot = None
    negative_class_one_hot = None
    if 'class' in config['condition']:
        # Setting class conditions
        class_info = torch.randint(0, config['num_classes'], (config['num_samples'], ))
        class_one_hot = torch.nn.functional.one_hot(class_info, num_classes = config['num_classes'])
        class_one_hot = class_one_hot.to(device)
        tqdm.write(f"[INFO] Class condition is found. The input class condition is: {class_info}.\n")
        
        # Save the tested text condition into a text file
        class_file = grid_plot_dir / "tested_class_condition.txt"
        class_save_info = str(class_info.tolist())
        class_str = f"The tested class conditions are {class_save_info}."
        with open(class_file, 'w') as f:
            f.write(class_str)
        
        
        # Setting negative class conditions for classifier free guidance
        if cf_guidance_value > 1:
            negative_class_one_hot = class_one_hot * 0
            negative_class_one_hot = negative_class_one_hot.to(device)
    
    
    if 'image' in config['condition'] or 'text' in config['condition']:
        mask_dataset = CELEB_dataset(directory = config['data_path'],
                                     config = config,
                                     result_dir = config['ac_save_path'],
                                     use_latent = config['use_latent'],
                                     model_latent_mask_flag = config['model_latent_mask_flag'])
    
    # Image conditions
    mask_image = None
    negative_mask_image = None
    if 'image' in config['condition']:
        # Setting image conditions
        # Randomly select mask images from the dataset and concatenate them in batch axis
        mask_idxs = torch.randint(0, len(mask_dataset), (config['num_samples'],))
        mask_image = torch.cat([mask_dataset[mask_idx][1]['image'][None,...] for mask_idx in mask_idxs], dim = 0) # B x 18 x H x W
        mask_image = mask_image.to(device)
        tqdm.write(f"[INFO] Image condition is found. The input image condition has a size of {mask_image.shape}.\n")
        
        # Save the tested mask condition into an image file
        mask_image_file = grid_plot_dir / "tested_mask_condition.png"
        
        mask_image_save = rearrange(mask_image.cpu(), 'B M H W -> (B M) 1 H W')
        
        mask_image_grid = make_grid(mask_image_save,
                                    nrow = config['mask_channels'])
        mask_image_grid_img = ToPILImage()(mask_image_grid)
        mask_image_grid_img.save(mask_image_file)
        
        # Setting negative image conditions for classifier free guidance
        if cf_guidance_value > 1:
            negative_mask_image = torch.zeros_like(mask_image)
            negative_mask_image = negative_mask_image.to(device)
        
        
        
        
        
    # Text conditions
    context_embedding = None
    negative_context_embedding = None
    if 'text' in config['condition']:
        # Setting text conditions
        # Change text_prompt for different input text condition
        # text_prompt = ["She is a woman with blond hair. She is wearing lipstick."]
        
        # If image condition is used, select the text caption that describes the mask, else randomly select
        if 'image' in config['condition']:
            text_prompt = []
            for mask_idx in mask_idxs:
                text_prompt.append(mask_dataset[mask_idx][1]['text'])    
        else:
            text_prompt = []
            text_idx = random.randint(0, len(mask_dataset)-1)
            text_prompt.append(mask_dataset.load_text(str(text_idx)))
        
        
        text_tokenizer, text_model = get_tokenizer_and_model(model_type = config['text_model'],
                                                             device = device)
        
        context_embedding = get_text_representation(text = text_prompt,
                                                    text_tokenizer = text_tokenizer,
                                                    text_model = text_model,
                                                    device = device)
        
        # If there is only one embedding, but need to generate more than 1 sample, clone them
        if context_embedding.shape[0] == 1:
            context_embedding = context_embedding.repeat(config['num_samples'], 1, 1) # repeat up to number of samples in batch axis
            context_embedding = context_embedding.to(device)
            tqdm.write(f"[INFO] Text condition is found. The input text condition is: {text_prompt}.\n")
        
        # Save the tested text condition into a text file
        text_file = grid_plot_dir / "tested_text_condition.txt"
        text_save_info = ""
        for text in text_prompt:
            text_save_info += text + "\n"
        with open(text_file, 'w') as f:
            f.write(text_save_info)
        
        # Setting negative text conditions for classifier free guidance
        if cf_guidance_value > 1:
            empty_prompt = [""]
            
            negative_context_embedding = get_text_representation(text = empty_prompt,
                                                                 text_tokenizer = text_tokenizer,
                                                                 text_model = text_model,
                                                                 device = device)
            negative_context_embedding = negative_context_embedding.repeat(config['num_samples'], 1, 1) # repeat up to number of samples in batch axis
            negative_context_embedding = negative_context_embedding.to(device)
            
        

    
    
    ### Main Inference ###
    
    model.eval()
    autoencoder.eval()
    with torch.inference_mode():
        
        for t in tqdm(list(reversed(range(config['num_timesteps'])))):
            
            pred_noise = model(x = out,
                               time_step = torch.tensor(t).repeat(out.shape[0]).to(device),
                               class_one_hot = class_one_hot,
                               mask_image = mask_image,
                               context_embedding = context_embedding)
            
            if cf_guidance_value > 1:
                negative_pred_noise = model(x = out,
                                            time_step = torch.tensor(t).repeat(out.shape[0]).to(device),
                                            class_one_hot = negative_class_one_hot,
                                            mask_image = negative_mask_image,
                                            context_embedding = negative_context_embedding)
                
                pred_noise = negative_pred_noise + cf_guidance_value * (pred_noise - negative_pred_noise)
                
            
            out = scheduler.sample_prev(x = out,
                                        t = t,
                                        pred_noise = pred_noise)
            
            # Plot and save the images
            if t % 100 == 0:
        
                out_plot = out
                    
                # Plot and save the image
                out_plot = torch.clamp(out_plot, -1, 1)
                out_plot = (out_plot + 1) / 2
                out_plot = out_plot.detach().cpu()
                
                grid = make_grid(out_plot, 
                                 nrow = config['samples_per_row'])
                grid_plot = ToPILImage()(grid)
                
                grid_plot_file = grid_plot_dir / f"{t}.png"
                grid_plot.save(grid_plot_file)
                tqdm.write(f"An instance of reverse diffusion has been plotted and saved into {grid_plot_file}")
                
                # Scale it to the same size as the decoded image
                scale_up_out_plot = torch.nn.functional.interpolate(out_plot, (config['im_size'], config['im_size']))
                
                scale_grid = make_grid(scale_up_out_plot, 
                                       nrow = config['samples_per_row'])
                scale_grid_plot = ToPILImage()(scale_grid)
                
                scale_grid_plot_file = grid_plot_dir / f"scaled_{t}.png"
                scale_grid_plot.save(scale_grid_plot_file)
                tqdm.write(f"An instance of reverse diffusion has been plotted and saved into {scale_grid_plot_file}")
                
                
                
                
                if t == 0:
                    # Decode it into the image space only if its the final step
                    decoded_plot = autoencoder.decode(x = out)
                    
                    # Plot and save the image
                    decoded_plot = torch.clamp(decoded_plot, -1, 1)
                    decoded_plot = (decoded_plot + 1) / 2
                    decoded_plot = decoded_plot.detach().cpu()
                
                    decoded_grid = make_grid(decoded_plot, 
                                             nrow = config['samples_per_row'])
                    decoded_grid_plot = ToPILImage()(decoded_grid)
                    
                    decoded_grid_plot_file = grid_plot_dir / "Final Decode.png"
                    decoded_grid_plot.save(decoded_grid_plot_file)
                    tqdm.write(f"The decoded final instance of reverse diffusion has been plotted and saved into {decoded_grid_plot_file}")
                
                
    
    
    
    
    
    

if __name__ == '__main__':
    # Create Argument Parser to obtain config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = 'The path to the config file in .yaml format')
    
    args = parser.parse_args()
    config_path = args.config
    
    # Call the training process
    infer(config_path = config_path)
    
    

