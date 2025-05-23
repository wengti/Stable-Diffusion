import torch


class linear_noise_scheduler():
    """
    A linear noise scheduler that can be used to do forward and reverse diffusion
    
    Args:
        config (dict): A config with .yaml extension loaded by yaml.safe_load \n
        device (string): 'cuda' or 'cpu'
    """
    
    def __init__(self, config, device):
        
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.num_timesteps = config['num_timesteps']
        
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x, t, noise):
        
        """
        Forward diffusion
        
        Args:
            x (float tensor): Input image, on the same device as the model, in the shape of B x im_ch x im_h x im_w \n
            t (int tensor): A batch of time steps, in the shape of B \n
            noise (float tensor): Noise to be added, on the same device as the model, in the shape of B x im_ch x im_h x im_w \n
            
        Returns:
            out (float tensor): Input image, but are modified with noise, in the shape of B x im_ch x im_h x im_w \n
        """
        
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        out = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return out 
    
    def sample_prev(self, x, t, pred_noise):
        
        """
        Reverse diffusion
        
        Args:
            x (float tensor): Input image, on the same device as the model, in the shape of B x im_ch x im_h x im_w \n
            t (int): A single time step \n
            pred_noise (float tensor): The noise that was predicted by a model and to be removed from x, on the same device as the model, in the shape of B x im_ch x im_h x im_w \n
            
        Returns:
            out (float tensor): output image with some noise removed and then added back, in the shape of B x im_ch x im_h x im_w
        """
        
        mean = (1 - self.alpha[t]) / (torch.sqrt(1 - self.alpha_hat[t])) * pred_noise
        mean = x - mean
        mean = mean / torch.sqrt(self.alpha[t])
        
        if t == 0:
            out = mean
        else:
            variance = (1 - self.alpha[t]) * (1 - self.alpha_hat[t-1]) / (1 - self.alpha_hat[t])
            std_dev = torch.sqrt(variance)
            z = torch.randn_like(mean)
            out = mean + std_dev * z
        
        return out
        
        