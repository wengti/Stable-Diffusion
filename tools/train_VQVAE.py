import yaml
import argparse
import torch
from dataset.MNIST_dataset import MNIST_dataset
from dataset.CELEB_dataset import CELEB_dataset
from utils.dataset_test import MNIST_dataset_test, CELEB_dataset_test
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from models.VQVAE import VQVAE
from torchinfo import summary
import torch.nn as nn
from models.discriminator import discriminator
from models.lpips import LPIPS
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import os

def train(config_path, load_file = None):
    
    """
    Carry out the training process
    
    Args:
        config_path (path): A config with .yaml extension loaded by yaml.safe_load \n
        load_file (path): A directories that consist of the file for trained VQVAE, discriminator and optimizer. If provided, training will be resumed
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
        tqdm.write(f"[INFO] The loaded configuration is as following: ")
        for key in config.keys():
            tqdm.write(f"{key} : {config[key]}")
        tqdm.write("\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Seed
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    # Result save path
    result_dir = Path(f'./result/{config['task_name']}_X')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    
    model_file = result_dir / "VQVAE.pt" # File to save the trained VQVAE
    discriminator_file = result_dir / "discriminator.pt" # File to save the trained discriminator 
    train_info_file = result_dir / "train_info.pt" # File to save the train info, including epoch, optimizers and losses
    
    # Directories to save plot during dataset_test(Reside within result_dir)
    save_plot_dir = result_dir / 'dataset_test'
    if not save_plot_dir.is_dir():
        save_plot_dir.mkdir(parents = True,
                            exist_ok = True)
    
    # Directories to store the reconstruction results during training
    reconstruction_dir = result_dir / 'reconstruction' 
    if not reconstruction_dir.is_dir():
        reconstruction_dir.mkdir(parents = True,
                                 exist_ok = True)
        
    
    
    
    
    
    
    
    
    # 1. Create dataset and verify it
    if config['dataset_type'] == 'MNIST':
        # Create dataset
        train_data = MNIST_dataset(directory = config['data_path'],
                                   config = config)
        
        if config['show_info']:
            # Verify the dataset
            tqdm.write(f"[INFO] The loaded dataset is as following: ")
            MNIST_dataset_test(data = train_data,
                               config = config,
                               save_plot_dir = save_plot_dir)
            tqdm.write("\n")
    
    elif config['dataset_type'] == 'CELEB':
        # Create dataset
        train_data = CELEB_dataset(directory = config['data_path'],
                                   config = config)
        
        if config['show_info']:
            # Verify the dataset
            tqdm.write(f"[INFO] The loaded dataset is as following: ")
            CELEB_dataset_test(data = train_data,
                               config = config,
                               save_plot_dir = save_plot_dir)
            tqdm.write("\n")
    
    
    # 2. Create a dataloader and verify it 
    train_data_loader = DataLoader(dataset = train_data,
                                   batch_size = config['ac_batch_size'],
                                   shuffle = True,
                                   num_workers = config['num_workers'])
    
    if config['show_info']:
        tqdm.write(f"[INFO] The loaded data loader is as following: ")
        train_img_batch, train_condition_batch = next(iter(train_data_loader))
        tqdm.write(f"Number of batches in the data loader: {len(train_data_loader)}")
        tqdm.write(f"Number of images in 1 batch         : {train_img_batch[0].shape}")
        tqdm.write("\n")
    
    # 3. Create models & verify models
    model = VQVAE(config = config).to(device)

    if config['show_model']:
        summary(model = model,
                input_size = (1,config['im_channels'], config['im_size'], config['im_size']),
                col_names = ['input_size' ,'output_size', 'trainable', 'num_params'],
                row_settings = ['var_names'])
    
    # 4. Create discriminator and lpips model
    model_disc = discriminator(config = config).to(device)
    model_lpips = LPIPS().to(device)
    
    
    # 4. Create optimizer and loss functions
    
    recon_loss_fn = nn.MSELoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    
    optimizer_g = torch.optim.Adam(params = model.parameters(),
                                   lr = config['ac_lr'],
                                   betas = (0.5, 0.999))
    optimizer_d = torch.optim.Adam(params = model_disc.parameters(),
                                   lr = config['ac_lr'],
                                   betas = (0.5, 0.999))
    
    
    
    # 5. If continue training, models and optimizer needs to be loaded with pre-trained weights
    prev_trained_epoch = 0
    if load_file is not None:
        
        load_file = Path(load_file)
        load_model = load_file / 'VQVAE.pt'
        load_discriminator = load_file / 'discriminator.pt'
        load_train_info = load_file / 'train_info.pt'

        assert os.path.exists(load_model), f"[INFO] {load_model} is not a valid file. Please check."
        assert os.path.exists(load_discriminator), f"[INFO] {load_discriminator} is not a valid file. Please check."
        assert os.path.exists(load_train_info), f"[INFO] {load_train_info} is not a valid file. Please check."
        
        model.load_state_dict(torch.load(f = load_model,
                                         weights_only = True))
        model_disc.load_state_dict(torch.load(f = load_discriminator,
                                              weights_only = True))
        
        prev_train_info = torch.load(f = load_train_info)
        optimizer_g.load_state_dict(prev_train_info['optimizer_g'])
        optimizer_d.load_state_dict(prev_train_info['optimizer_d'])
        
        config['disc_start_step'] = 0 # In continue training, assume previous waiting time for discriminator is no longer needed
        prev_trained_epoch = prev_train_info['epoch'] + 1 # Add the number of epochs that have been trained before to the current epoch
                                                          # +1 because the epoch was saved starting from 0 
        
        print(f"[INFO] Previous training was stopped at:")
        for key in prev_train_info.keys():
            if key != 'optimizer_g' and key != 'optimizer_d':
                print(f"{key}: {prev_train_info[key]}")
    
    
    
    
    
    
    
    
    
    # 6. Training Loops
    num_epochs = config['ac_epochs']
    disc_start_step = config['disc_start_step']
    step = 0
    
    model.train() # Enterring training mode
    model_disc.train() # Enterring training mode
    model_lpips.eval() # Enterring evaluation mode
    
    # Clear gradients first
    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    
    # The beginning of 1 epoch
    for epoch in tqdm(range(num_epochs- prev_trained_epoch)):
        
        epoch = epoch + prev_trained_epoch # Only effective if the training is continued
        
        recon_loss_per_epoch = []
        codebook_loss_per_epoch = []
        commitment_loss_per_epoch = []
        perceptual_loss_per_epoch = []
        adv_loss_g_per_epoch = []
        g_loss_per_epoch = []
        adv_loss_d_per_epoch = []
        
        # The beginning of 1 step
        for batch, (inpt_imgs, _) in enumerate(tqdm(train_data_loader)):
            
            # Increase number of forward propagation step
            step += 1
            
            ###################### Training the VQVAE as a generator ######################
            
            # Reconstruct the images using the encoder-decocder
            inpt_imgs = inpt_imgs.to(device)
            output_imgs, _, vqvae_losses = model(inpt_imgs)
            
            # Compute reconstruction loss
            recon_loss = recon_loss_fn(output_imgs, inpt_imgs)
            
            recon_loss_per_epoch.append(recon_loss.item())
            
            # Obtain codebook and commitment loss
            codebook_loss = vqvae_losses['codebook_loss']
            commitment_loss = vqvae_losses['commitment_loss']
            
            codebook_loss_per_epoch.append(codebook_loss.item())
            commitment_loss_per_epoch.append(commitment_loss.item())
            
            # Compute perceptual loss
            perceptual_loss = torch.mean(model_lpips(output_imgs, inpt_imgs))
            
            perceptual_loss_per_epoch.append(perceptual_loss.item())
            
            # Compute discriminator loss for the generator (if number of forward propagation steps is above threshold)
            adv_loss_g = 0
            if step > disc_start_step:
                disc_out_g = model_disc(output_imgs)
                adv_loss_g = adv_loss_fn(disc_out_g, torch.ones(disc_out_g.shape).to(device))
                
                adv_loss_g_per_epoch.append(adv_loss_g.item())
            
            
            # Sum up the product of loss with their respective weight
            loss = recon_loss + \
                    config['adv_loss_weight'] * adv_loss_g + \
                        config['codebook_weight'] * codebook_loss + \
                            config['commitment_weight'] * commitment_loss + \
                                config['perceptual_weight'] * perceptual_loss
            
            g_loss_per_epoch.append(loss.item())
            
            # Perform backpropagation to update the dL / dx in each of the parameter x
            loss = loss / config['ac_acc_steps']
            loss.backward()
            
            # Perform learning rate update when the number of accumulation step is achieved
            if step % config['ac_acc_steps'] == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
            
                
            
            
            ###################### Training the discriminator ######################
            if step > disc_start_step:
                fake_pred = model_disc(output_imgs.detach())
                real_pred = model_disc(inpt_imgs)
                
                # Compute the discriminator adversarial loss
                adv_loss_d_fake = adv_loss_fn(fake_pred, torch.zeros(fake_pred.shape).to(device))
                adv_loss_d_real = adv_loss_fn(real_pred, torch.ones(real_pred.shape).to(device))
                adv_loss_d = (adv_loss_d_fake + adv_loss_d_real) * 0.5 
                
                adv_loss_d_per_epoch.append(adv_loss_d.item())
                
                # Multiply with weight and divide by number of steps
                adv_loss_d =  config['adv_loss_weight'] * adv_loss_d 
                adv_loss_d = adv_loss_d / config['ac_acc_steps']
                adv_loss_d.backward()
                
                # Perform learning rate update when the number of accumulation step is achieved
                if step % config['ac_acc_steps'] == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                
                
            
            # Image saving logics
            if step % config['ac_im_save_steps'] == 0:
                sample_size = min(8, output_imgs.shape[0])
                output_imgs = output_imgs[:sample_size].detach().cpu() # Select only 8 images when there is a lot of them
                output_imgs = torch.clamp(output_imgs, -1, 1) # Keep the image in the range of -1 to 1
                output_imgs = (output_imgs + 1) / 2 
                
                inpt_imgs = inpt_imgs[:sample_size].detach().cpu()
                inpt_imgs = (inpt_imgs + 1) / 2
                
                output = torch.cat([output_imgs, inpt_imgs], dim=0)
                grid = make_grid(output, nrow = sample_size)
                save_ims = ToPILImage()(grid)
                save_ims.save(reconstruction_dir/f"{step}.png")
            
            
            
            
        # Perform backpropagation for the final few steps that did not reach accumulation steps
        optimizer_g.step()
        optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        
        
        # Compute average loss in 1 epoch
        recon_loss_per_epoch = sum(recon_loss_per_epoch) / len(recon_loss_per_epoch)
        codebook_loss_per_epoch = sum(codebook_loss_per_epoch) / len(codebook_loss_per_epoch)
        commitment_loss_per_epoch = sum(commitment_loss_per_epoch) / len(commitment_loss_per_epoch)
        perceptual_loss_per_epoch = sum(perceptual_loss_per_epoch) / len(perceptual_loss_per_epoch)
        g_loss_per_epoch = sum(g_loss_per_epoch) / len(g_loss_per_epoch)

        if len(adv_loss_g_per_epoch) != 0:
            adv_loss_g_per_epoch = sum(adv_loss_g_per_epoch) / len(adv_loss_g_per_epoch)
        else: 
            adv_loss_g_per_epoch = 0
        
        if len(adv_loss_d_per_epoch) != 0:
            adv_loss_d_per_epoch = sum(adv_loss_d_per_epoch) / len(adv_loss_d_per_epoch)
        else:
            adv_loss_d_per_epoch = 0
        
        
        # print Info
        tqdm.write(f"[INFO] Current Epoch: {epoch}")
        tqdm.write("[INFO] For Generator: ")
        tqdm.write(f"Total Loss: {g_loss_per_epoch:.4f}")
        tqdm.write(f"Reconstruction Loss: {recon_loss_per_epoch:.4f}")
        tqdm.write(f"Codebook Loss: {codebook_loss_per_epoch:.4f}")
        tqdm.write(f"Commitment Loss: {commitment_loss_per_epoch:.4f}")
        tqdm.write(f"Perceptual Loss: {perceptual_loss_per_epoch:.4f}")
        tqdm.write(f"AdversariaL Loss: {adv_loss_g_per_epoch:.4f}")
        tqdm.write(f"[INFO] For Discriminator: ")
        tqdm.write(f"Adversarial Loss: {adv_loss_d_per_epoch:.4f}\n")
        
        
        
        # Save models and train info
        train_info = {'epoch': epoch,
                      'optimizer_g': optimizer_g.state_dict(),
                      'optimizer_d': optimizer_d.state_dict(),
                      'total_loss': g_loss_per_epoch,
                      'recon_loss': recon_loss_per_epoch,
                      'codebook_loss': codebook_loss_per_epoch,
                      'commitment_loss': commitment_loss_per_epoch,
                      'perceptual_loss': perceptual_loss_per_epoch,
                      'adversarial_loss_g': adv_loss_g_per_epoch,
                      'adversarial_loss_d': adv_loss_d_per_epoch}
        
        train_info_file = result_dir / f"train_info_{epoch}.pt" # File to save the train info, including epoch, optimizers and losses
        
        torch.save(model.state_dict(), model_file)
        torch.save(model_disc.state_dict(), discriminator_file)
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
    
