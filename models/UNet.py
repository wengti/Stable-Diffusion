import torch
import torch.nn as nn
from models.blocks import down_block, mid_block, up_block, sinusoidal_embedding
import random

class UNet(nn.Module):
    
    """
    A UNet structure to be used as the backbone of a Diffusion Model that can accept text, image and class conditioning.
    
    Args:
        config (dict): A config with .yaml extension loaded by yaml.safe_load \n
    """
    
    def __init__(self, config):
        super().__init__()
        
        assert config['ldm_down_channels'][-1] == config['ldm_mid_channels'][0], "[WARNING] Last down channel does not match with the first mid channel."
        assert config['ldm_down_channels'][-2] == config['ldm_mid_channels'][-1], "[WARNING] First up channel does not match with the last mid channel."
        assert len(config['ldm_down_sample']) == (len(config['ldm_down_channels']) - 1), "[WARNING] The number of down sample should be exactly one less than the number of down channels."
        assert len(config['ldm_down_attn']) == (len(config['ldm_down_channels']) - 1), "[WARNING] The number of down attention flag should be exactly one less than the number of down channels."
        assert len(config['ldm_mid_attn']) == (len(config['ldm_mid_channels']) - 1), "[WARNING] The number of mid attention flag should be exactly one less than the number of mid channels."
        
        self.config = config
        
        
        # Setup if there are class conditions
        if 'class' in config['condition']:
            self.class_embedding = nn.Embedding(num_embeddings = config['num_classes'] ,
                                                embedding_dim = config['time_embedding_dim'])
        
        
        # Setup if there are image conditions
        if 'image' in config['condition']:
            self.image_input_conv = nn.Conv2d(in_channels = config['mask_channels'],
                                              out_channels = config['mask_channels'],
                                              kernel_size = 1,
                                              stride = 1,
                                              padding = 0)
            
            self.concat_conv = nn.Conv2d(in_channels = config['z_channels'] + config['mask_channels'],
                                         out_channels = config['ldm_down_channels'][0],
                                         kernel_size = 3,
                                         stride = 1,
                                         padding = 1)
        else:
            self.input_conv = nn.Conv2d(in_channels = config['z_channels'],
                                        out_channels = config['ldm_down_channels'][0],
                                        kernel_size = 3,
                                        stride = 1,
                                        padding = 1)
        
        # Setup if there are text conditions 
        self.cross_attn_flag = False
        self.context_dim = None
        if 'text' in config['condition']:
            self.cross_attn_flag = True 
            self.context_dim = config['text_embedding_dim']
        
            
        
        self.down_blocks = nn.ModuleList([])
        for i in range(len(config['ldm_down_channels']) - 1):
            self.down_blocks.append(down_block(in_channels = config['ldm_down_channels'][i],
                                               out_channels = config['ldm_down_channels'][i+1],
                                               norm_channels = config['ldm_norm_channels'],
                                               num_layers = config['ldm_num_down_layers'],
                                               down_sample_flag = config['ldm_down_sample'][i],
                                               time_embedding_dim = config['time_embedding_dim'],
                                               self_attn_flag = config['ldm_down_attn'][i],
                                               num_heads = config['ldm_num_heads'],
                                               cross_attn_flag = self.cross_attn_flag,
                                               context_dim = self.context_dim))
        
        self.mid_blocks = nn.ModuleList([])
        for i in range(len(config['ldm_mid_channels']) - 1):
            self.mid_blocks.append(mid_block(in_channels = config['ldm_mid_channels'][i],
                                             out_channels = config['ldm_mid_channels'][i+1],
                                             norm_channels = config['ldm_norm_channels'],
                                             num_layers = config['ldm_num_mid_layers'],
                                             time_embedding_dim = config['time_embedding_dim'],
                                             self_attn_flag = config['ldm_mid_attn'],
                                             num_heads = config['ldm_num_heads'],
                                             cross_attn_flag = self.cross_attn_flag,
                                             context_dim = self.context_dim))
        
        ldm_up_channels = list(reversed(config['ldm_down_channels']))
        ldm_up_sample = list(reversed(config['ldm_down_sample']))
        ldm_up_attn = list(reversed(config['ldm_down_attn']))
        
        self.up_blocks = nn.ModuleList([])
        for i in range(len(ldm_up_channels) - 1):
            self.up_blocks.append(up_block(in_channels = ldm_up_channels[i+1] * 2,
                                           out_channels = config['ldm_conv_out_channels'] if i == (len(ldm_up_channels) - 2) else ldm_up_channels[i+2],
                                           norm_channels = config['ldm_norm_channels'],
                                           num_layers = config['ldm_num_up_layers'],
                                           up_sample_flag = ldm_up_sample[i],
                                           time_embedding_dim = config['time_embedding_dim'],
                                           self_attn_flag = ldm_up_attn[i],
                                           num_heads = config['ldm_num_heads'],
                                           cross_attn_flag = self.cross_attn_flag,
                                           context_dim = self.context_dim))
        
        self.output_conv = nn.Sequential(nn.GroupNorm(num_groups = config['ldm_norm_channels'],
                                                      num_channels = config['ldm_conv_out_channels']),
                                         nn.SiLU(),
                                         nn.Conv2d(in_channels = config['ldm_conv_out_channels'],
                                                   out_channels = config['z_channels'],
                                                   kernel_size = 3,
                                                   stride = 1,
                                                   padding = 1))
        
        self.time_projection = nn.Sequential(nn.Linear(in_features = self.config['time_embedding_dim'],
                                                       out_features = self.config['time_embedding_dim']),
                                             nn.SiLU(),
                                             nn.Linear(in_features = self.config['time_embedding_dim'],
                                                       out_features = self.config['time_embedding_dim']))
        
        
        
    
    def forward(self, x, time_step = None, class_one_hot = None, mask_image = None, context_embedding = None):
        
        """
        Forward propagation of the UNet structure.
        
        Args:
            x (float tensor): A batch of input images, modified with noise, on cuda device, in the range of -1 to 1, in the shape of B x im_channels x H x W in the latent space \n
            time_step (int tensor): input time step to the block, expected to be on cuda, in the shape of (B,) \n
            class_one_hot (int tensor): input of class information, expected to be on cuda, in the form of one hot, in the shape of (B, num_classes)
            mask_image (float tensor): Mask image in the shape of (B, mask_channels, H x W), expected to be on cuda, should contains values of either 0 or 1 \n
            context_embedding (float tensor): input context embedding (for cross attention), expected to be on cuda, in the shape of (B, token_length, context_dim) \n
            
        Return:
            out (float tensor): Predicted noise that was added to the input image
        """
        
        # Enterring testing mode if no time_step is provided
        if time_step == None:
            print("[INFO] Enterring testing mode... ")
            print("Assign time_embedding to a random value...")
            device = next(self.parameters()).device
            time_step = torch.randint(0, self.config['num_timesteps'], (1,)).to(device)
            
            if 'class' in self.config['condition']:
                print(f"Creating temporary one_hot class info...")
                class_one_hot = torch.zeros(1, self.config['num_classes']).to(device)
                one_idx = random.randint(0, self.config['num_classes']-1)
                class_one_hot[0][one_idx] = 1
            
            if 'image' in self.config['condition']:
                print("Assign mask_image to a random value...")
                B,_,H,W = x.shape
                mask_image = torch.randn(B, self.config['mask_channels'], H, W).to(device)
            
            if 'text' in self.config['condition']:
                print("Assign context_embedding to a random value...")
                context_embedding = torch.randn(1, 10, self.config['text_embedding_dim']).to(device)
                
                
        # Convert time steps to time embedding
        
        time_embedding = sinusoidal_embedding(time_step = time_step,
                                              time_embedding_dim = self.config['time_embedding_dim'])
        
        time_embedding = self.time_projection(time_embedding)
        
        # Add class embedding to time embedding if needed
        if 'class' in self.config['condition']:
            class_embedding = torch.matmul(class_one_hot.float(), self.class_embedding.weight) # B x time_embedding
            
            time_embedding += class_embedding # Add to the time embedding
    
    
        # Check the requirements for image conditioning
        if 'image' in self.config['condition']:
            assert mask_image != None, "[WARNING] Expecting an input of mask image as the model is image-conditioned."
            assert mask_image.shape[0] == x.shape[0], "[WARNING] The batch number between input image and mask image does not match."
            assert mask_image.shape[2] == x.shape[2], "[WARNING] The spatial size between input image and mask image does not match."
            
        # Check the requirements for text conditioning
        if 'text' in self.config['condition']:
            assert context_embedding != None, "[WARNING] Expecting an input of context embedding as the model is text-conditioned."
        
        
        
        
        
        out = x
        # Input convolutional block
        if 'image' in self.config['condition']:
            mask_input = self.image_input_conv(mask_image.float())
            out = torch.cat([out, mask_input], dim=1)
            out = self.concat_conv(out)
        else:   
            out = self.input_conv(out)
        
        # Down block
        down_maps = []
        for block in self.down_blocks:
            down_maps.append(out)
            out = block(out, time_embedding, context_embedding)
        
        # Mid block
        for block in self.mid_blocks:
            out = block(out, time_embedding, context_embedding)
            
        # Up block
        for block in self.up_blocks:
            down_map = down_maps.pop()
            out = block(out, down_map, time_embedding, context_embedding)
        
        # Output convolutional block
        out = self.output_conv(out)
        
        return out

