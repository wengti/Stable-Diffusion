import yaml
from models.VQVAE import VQVAE
import torch
from pathlib import Path
from dataset.MNIST_dataset import MNIST_dataset
from dataset.CELEB_dataset import CELEB_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import argparse

def infer(config_path, save_latent_flag = False, infer_latent_flag = False, ):
    
    """
    Perform inference
    
    Args:
        config_path (path): A config with .yaml extension loaded by yaml.safe_load \n
        save_latent_flag (boolean): Decide whether to save latent or not \n
        infer_latent_flag (boolean): Decide whether to use random pre-saved latent data as input for inference \n
    
    """
    
    assert config_path is not None, "[WARNING] No config_path is provided. Please check."
    
    
    # Read config
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Result directories
    result_dir = Path(f'./result/{config['task_name']}')
    if not result_dir.is_dir():
        result_dir.mkdir(parents = True,
                         exist_ok = True)
    
    # The directories that store the generated latent data
    latent_save_dir = result_dir / 'latent_data'
    if not latent_save_dir.is_dir():
        latent_save_dir.mkdir(parents = True,
                              exist_ok = True)
    
    
    # Pretrained weight 
    model_file = result_dir / 'VQVAE.pt'
    
    # Create datasets
    if config['dataset_type'] == 'MNIST':
        train_data = MNIST_dataset(directory = Path(config['data_path']),
                                   config = config)
        
        if infer_latent_flag:
            # Only needed for infer_latent mode
            train_latent_data = MNIST_dataset(directory = Path(config['data_path']),
                                              config = config,
                                              result_dir = result_dir,
                                              use_latent = True)
    
    elif config['dataset_type'] == 'CELEB':
        train_data = CELEB_dataset(directory = Path(config['data_path']),
                                   config = config)
        
        if infer_latent_flag:
            # Only needed for infer_latent mode
            train_latent_data = CELEB_dataset(directory = Path(config['data_path']),
                                              config = config,
                                              result_dir = result_dir,
                                              use_latent = True)
    
    # Create dataloader
    if save_latent_flag:
        # Only needed for latent_saving mode
        train_data_loader = DataLoader(dataset = train_data,
                                       batch_size = 1,
                                       shuffle = False,
                                       num_workers = config['num_workers'])
    
            
    # Load models
    model = VQVAE(config = config).to(device)
    model.load_state_dict(torch.load(f = model_file,
                                     weights_only = True))
    
    
    model.eval()
    with torch.inference_mode():
        
        # Save latent space
        if save_latent_flag:
            print(f"[INFO] Entering latent data saving mode...")
            latent_data = {}
            part_count = 0
            
            for idx, (inpt_img, _) in enumerate(tqdm(train_data_loader)):
                
                # Complete the encoding process
                inpt_img = inpt_img.to(device)
                outpt_latent, _, _ = model.encode(inpt_img)
                
                # Save the latent data into a dictionary {img_path: latent_data}
                # The latent data better be saved in cpu to avoid errors in the future when loading onto cuda again
                latent_data[train_data.img_path_list[idx]] = outpt_latent.to('cpu')
                
                # For every 1000 input images, save the latent data
                if (idx+1) % 1000 == 0:
                    
                    latent_save_file = latent_save_dir / f"Part_{part_count}.pkl"
                    with open(latent_save_file, "wb") as f:
                        pickle.dump(obj = latent_data,
                                    file = f)
                    latent_data = {}
                    part_count += 1
            
            # Save the remainder 
            if len(latent_data) > 0:
                latent_save_file = latent_save_dir / f"Part_{part_count}.pkl"
                with open(latent_save_file, "wb") as f:
                    pickle.dump(obj = latent_data,
                                file = f)
            print(f"[INFO] Latent saving has been completed with result saved in {result_dir}\n")
        
        
        # Infer with latent data 
        if infer_latent_flag:
            print(f"[INFO] Entering inference mode using latent data")
                
            # Sample 8 numbers of images
            num_samples = 8
            rand_num = torch.randint(0, len(train_data)-1 , (num_samples,))
            inpt_imgs = torch.cat([train_data[num][0][None,...] for num in rand_num], dim=0)
            inpt_imgs = (inpt_imgs + 1) / 2
            
            # Decode their corresponding latent data
            inpt_latent = torch.cat([train_latent_data[num][0][None, ...] for num in rand_num], dim = 0).to(device)
            output_imgs = model.decode(inpt_latent)
            output_imgs = torch.clamp(output_imgs, -1, 1)
            output_imgs = output_imgs.detach().cpu()
            output_imgs = (output_imgs + 1) / 2
            
            # Generate grid images for comparison
            output = torch.cat([output_imgs, inpt_imgs], dim = 0)
            grid = make_grid(output, nrow = num_samples)
            grid_img = ToPILImage()(grid)
            grid_img.save(result_dir / "Reconstruction_with_latent_data.png")
            
            print(f"[INFO] Inference has been completed with result saved in {result_dir}\n")
                
                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str,
                        help = "The directories to a path that contains a .yaml file")
    parser.add_argument('--save_latent_flag', action = 'store_true',
                        help = "The flag to decide whether to save latent data")
    parser.add_argument('--infer_latent_flag', action = 'store_true',
                        help = "The flag to decide whether to execute inference using latent data")
    
    args = parser.parse_args()
    config_path = args.config
    save_latent_flag = args.save_latent_flag
    infer_latent_flag = args.infer_latent_flag
    
    infer(config_path = config_path,
          save_latent_flag = save_latent_flag,
          infer_latent_flag = infer_latent_flag)
    
    
                
            
            
            
        
        
    
    