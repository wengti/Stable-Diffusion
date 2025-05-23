import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from dataset.MNIST_dataset import MNIST_dataset
from dataset.CELEB_dataset import CELEB_dataset
from utils.dataset_test import MNIST_dataset_test, CELEB_dataset_test
from torch.utils.data import DataLoader
from torchinfo import summary
from models.UNet import UNet
from models.VQVAE import VQVAE
import torch.nn as nn
from scheduler.linear_noise_scheduler import linear_noise_scheduler
import os
import argparse
from utils.diffusion_utils import drop_class_condition, drop_image_condition, drop_text_condition
from utils.text_utils import get_tokenizer_and_model, get_text_representation

def train(config_path, load_file = None):
    
    """
    Carry out the training process
    
    Args:
        config_path (path): A config with .yaml extension loaded by yaml.safe_load \n
        load_file (path): A directories that consist of the file for trained model and optimizer. If provided, training will be resumed
                          from the provided check point in load_file\n
    """
    
    assert config_path is not None, "[WARNING] No config_path is provided. Please check."
    
    
    # Read config
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            tqdm.write(exc)
    
    if config['show_info']:
        tqdm.write("[INFO] The loaded configuration is as following: ")
        for key in config.keys():
            tqdm.write(f"{key} : {config[key]}")
        tqdm.write("\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Seed
    # torch.manual_seed(config['seed'])
    # torch.cuda.manual_seed(config['seed'])
    
    # Result save path
    result_dir = Path(f'./result/{config['task_name']}')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    
    model_file = result_dir / "LDM.pt" # File to save the trained VQVAE
    train_info_file = result_dir / "train_info.pt" # File to save the train info, including epoch, optimizers and losses
    
    # Directories to save plot during dataset_test(Reside within result_dir)
    save_plot_dir = result_dir / 'dataset_test'
    if not save_plot_dir.is_dir():
        save_plot_dir.mkdir(parents = True,
                            exist_ok = True)
    
    
    ###########################################################################
    
    
    # 1. Create dataset and verify it
    use_latent = config['use_latent']
    
    
    if config['dataset_type'] == 'MNIST':
        # Create dataset
        train_data = MNIST_dataset(directory = config['data_path'],
                                   config = config,
                                   result_dir = config['ac_save_path'],
                                   use_latent = config['use_latent'],
                                   model_latent_mask_flag = config['model_latent_mask_flag'])
        
        if config['show_info']:
            # Verify the dataset
            tqdm.write("[INFO] The loaded dataset is as following: ")
            MNIST_dataset_test(data = train_data,
                               config = config,
                               save_plot_dir = save_plot_dir)
            tqdm.write("\n")
    
    elif config['dataset_type'] == 'CELEB':
        # Create dataset
        train_data = CELEB_dataset(directory = config['data_path'],
                                   config = config,
                                   result_dir = config['ac_save_path'],
                                   use_latent = config['use_latent'],
                                   model_latent_mask_flag = config['model_latent_mask_flag'])
        
        if config['show_info']:
            # Verify the dataset
            tqdm.write(f"[INFO] The loaded dataset is as following: ")
            CELEB_dataset_test(data = train_data,
                               config = config,
                               save_plot_dir = save_plot_dir)
            tqdm.write("\n")
    
    
    
    # 2. Create a dataloader and verify it 
    train_data_loader = DataLoader(dataset = train_data,
                                   batch_size = config['ldm_batch_size'],
                                   shuffle = True,
                                   num_workers = config['num_workers'])
    
    if config['show_info']:
        tqdm.write(f"[INFO] The loaded data loader is as following: ")
        train_img_batch, train_condition_batch = next(iter(train_data_loader))
        tqdm.write(f"Number of batches in the data loader: {len(train_data_loader)}")
        tqdm.write(f"Number of images in 1 batch         : {train_img_batch[0].shape}")
        if 'image' in config['condition']:
            tqdm.write(f"train_condition_batch (mask)        : {train_condition_batch['image'].shape}")
        tqdm.write("\n")
    
    
    
    # 3. Create models & verify models
    model = UNet(config = config).to(device)

    if config['show_model']:
        test_size = config['im_size'] // (2**(sum(config['ac_down_sample'])))
        summary(model = model,
                input_size = (config['ldm_batch_size'],config['z_channels'], test_size, test_size),
                col_names = ['input_size' ,'output_size', 'trainable', 'num_params'],
                row_settings = ['var_names'])
    
    
    # 4. Create optimizer, loss functions, scheduler and load autoencoder if needed
    
    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = config['ldm_lr'])
    
    scheduler = linear_noise_scheduler(config = config,
                                       device = device)
    
    # Load model if latent data is not used directly
    if not use_latent:
        assert os.path.exists(config['ac_save_path']), "[WARNING] Provided path to the autoencoder save file is not valid. Please check config."
        
        load_autoencoder_file = Path(config['ac_save_path']) / 'VQVAE.pt'
        tqdm.write(f"[INFO] The status of using latent data: {config['use_latent']}")
        tqdm.write(f"[INFO] Loading the autoencoder from {load_autoencoder_file}... ...")
        
        autoencoder = VQVAE(config = config).to(device)
        autoencoder.load_state_dict(torch.load(f = load_autoencoder_file,
                                               weights_only = True))
        
        # Freeze the loaded model
        for params in autoencoder.parameters():
            params.requires_grad = False
        
    # Load the text model
    if 'text' in config['condition']:
        # Load the tokenizer and text model to be used based on provided name in config
        text_tokenizer, text_model = get_tokenizer_and_model(model_type = config['text_model'],
                                                             device = device)
        
        # Get the replacement for text embedding that is to be dropped
        empty_text_embed = get_text_representation(text = [''],
                                                   text_tokenizer = text_tokenizer,
                                                   text_model = text_model,
                                                   device = device) # 1 x 77 x context_dim
        
    
    
    # 5. If continue training, models and optimizer needs to be loaded with pre-trained weights
    prev_trained_epoch = 0
    if load_file is not None:
        
        load_file = Path(load_file)
        load_model = load_file / 'LDM.pt'
        load_train_info = load_file / 'train_info.pt'

        assert os.path.exists(load_model), f"[INFO] {load_model} is not a valid file. Please check."
        assert os.path.exists(load_train_info), f"[INFO] {load_train_info} is not a valid file. Please check."
        
        model.load_state_dict(torch.load(f = load_model,
                                         weights_only = True))

        prev_train_info = torch.load(f = load_train_info)
        optimizer.load_state_dict(prev_train_info['optimizer'])
        
        prev_trained_epoch = prev_train_info['epoch'] + 1 # Add the number of epochs that have been trained before to the current epoch
                                                          # +1 because the epoch was saved starting from 0 
        
        print("[INFO] Previous training was stopped at:")
        for key in prev_train_info.keys():
            if key != 'optimizer':
                print(f"{key}: {prev_train_info[key]}")
    
    
    # 6. Training Loops   
    num_epochs = config['ldm_epochs']
    optimizer.zero_grad()
    
    model.train()    
    for epoch in tqdm(range((num_epochs - prev_trained_epoch))):
        
        epoch = epoch + prev_trained_epoch
        loss_list = []
        
        for batch, (inpt_imgs, condition) in enumerate(tqdm(train_data_loader)):
            
            inpt_imgs = inpt_imgs.to(device)
            
            # Get the latent data by calling the autoencoder is use_latent flag is False
            if not use_latent:
                autoencoder.eval()
                with torch.no_grad():
                    inpt_imgs, _, _ = autoencoder.encode(x = inpt_imgs)
            
            # Prepare noisy image using scheduler
            B = inpt_imgs.shape[0]
            t = torch.randint(0, config['num_timesteps'], (B,)).to(device)
            noise = torch.randn_like(inpt_imgs).to(device)
            noisy_imgs = scheduler.add_noise(x = inpt_imgs,
                                             t = t,
                                             noise = noise)
            
            
            # Setting up class conditions
            class_one_hot = None
            if 'class' in config['condition']:
                # Obtain the class in batches
                class_info = condition['class'].to(device) 
                
                # Convert the class info into one hot vector
                class_one_hot = torch.nn.functional.one_hot(class_info, 
                                                            num_classes = config['num_classes']).to(device)
                
                # Pass it to randomly drop the one hot vector if below a threshold
                class_one_hot = drop_class_condition(class_one_hot = class_one_hot,
                                                     class_drop_prob = config['class_drop_prob'],
                                                     device = device)
            
            # Setting up image conditions
            mask_image = None
            if 'image' in config['condition']:
                
                # Obtain the mask image
                mask_image = condition['image'].to(device)
                
                # Pass it to randomly drop the mask if below a threshold
                mask_image = drop_image_condition(mask_image = mask_image,
                                                  image_drop_prob = config['image_drop_prob'],
                                                  device = device)
                
            
            # Setting up text conditions
            context_embedding = None
            if 'text' in config['condition']:
                
                # Obtain the list of captions for this batch
                caption = condition['text'] # ['caption1', 'caption2', 'caption3'....]
                
                # Get their corresponding embedding
                context_embedding = get_text_representation(caption, 
                                                            text_tokenizer = text_tokenizer,
                                                            text_model = text_model,
                                                            device = device) # B x 77 x context_dim
                
                context_embedding = context_embedding.to(device)
                
                # Pass it to randomly drop the embedding if below a threshold
                context_embedding = drop_text_condition(context_embedding = context_embedding,
                                                        text_drop_prob = config['text_drop_prob'],
                                                        empty_text_embed = empty_text_embed,
                                                        device = device)
            
            # Perform prediction
            pred_noise = model(x = noisy_imgs,
                               time_step = t,
                               class_one_hot = class_one_hot,
                               mask_image = mask_image,
                               context_embedding = context_embedding)
            
            # Loss computation 
            loss = loss_fn(pred_noise, noise)
            loss_list.append(loss.item())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print Info
        avg_loss = sum(loss_list) / len(loss_list)
        tqdm.write(f"[INFO] Current Epoch: {epoch}")
        tqdm.write(f"Training Loss : {avg_loss:.4f}")
        
        # Save models and train info
        train_info = {'epoch': epoch,
                      'optimizer': optimizer.state_dict(),
                      'loss': avg_loss}
        
        torch.save(model.state_dict(), model_file)
        torch.save(train_info, train_info_file)



if __name__ == '__main__':
    # Create Argument Parser to obtain config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = 'The path to the config file in .yaml format')
    parser.add_argument('--load_file', type = str,
                        help = 'File that consists of models and optimizer for continue training')
    
    args = parser.parse_args()
    config_path = args.config
    load_file = args.load_file
    
    # Call the training process
    train(config_path = config_path,
          load_file = load_file)

# =============================================================================
#     tqdm.write("[INFO] Training Model 1: LDM MNIST CLASS")
#     config_path = Path('./config/LDM_MNIST_CLASS.yaml')
#     train(config_path = config_path)
# =============================================================================
    
    
# =============================================================================
#     tqdm.write("[INFO] Training Model 2: LDM CELEB UNCONDITIONAL")
#     config_path = Path('./config/LDM_CELEB_UNC.yaml')
#     load_file = Path('./result/LDM_CELEB_UNC')
#     train(config_path = config_path,
#           load_file = load_file)
# =============================================================================
    

# =============================================================================
#     tqdm.write("[INFO] Training Model 3: LDM CELEB TEXT")
#     config_path = Path('./config/LDM_CELEB_TEXT.yaml')
#     train(config_path = config_path)
# =============================================================================
    
    
# =============================================================================
#     tqdm.write("[INFO] Training Model 4: LDM CELEB IMAGE TEXT")
#     config_path = Path('./config/LDM_CELEB_IMAGE_TEXT.yaml')
#     load_file = Path('./result/LDM_CELEB_IMAGE_TEXT')
#     train(config_path = config_path,
#           load_file = load_file)
# =============================================================================
    
    
# =============================================================================
#     tqdm.write("[INFO] Training Model 5: LDM CELEB IMAGE TEXT")
#     config_path = Path('./config/LDM_CELEB_IMAGE.yaml')
#     train(config_path = config_path)
# =============================================================================
    
    
       
            
            
        
        
 
