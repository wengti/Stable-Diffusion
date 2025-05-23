import torch
import torch.nn as nn
from models.blocks import down_block, mid_block, up_block



class VQVAE(nn.Module):
    """
    Vector-Quantized Variational Auto Encoder
    
    Args:
        config (dict): A config with .yaml extension loaded by yaml.safe_load \n
    """
    
    def __init__(self, config):
        super().__init__()
        
        assert config['ac_down_channels'][-1] == config['ac_mid_channels'][0], "[WARNING] The last down channel do not match the first mid channel"
        assert config['ac_down_channels'][-1] == config['ac_mid_channels'][-1], "[WARNING] The first up channel do not match last mid channel"
        assert len(config['ac_down_sample']) == len(config['ac_down_channels']) - 1, "[WARNING] The number of down sample should be exactly one less than the number of down channels."
        
        ############# Encoder #############
        self.enc_in_conv = nn.Conv2d(in_channels = config['im_channels'],
                                     out_channels = config['ac_down_channels'][0],
                                     kernel_size = 3,
                                     stride = 1,
                                     padding = 1)
        
        
        self.encoder_blocks = nn.ModuleList([])
        for i in range(len(config['ac_down_channels']) - 1):
            self.encoder_blocks.append(down_block(in_channels = config['ac_down_channels'][i],
                                                  out_channels = config['ac_down_channels'][i+1],
                                                  norm_channels = config['ac_norm_channels'],
                                                  num_layers = config['ac_num_down_layers'],
                                                  down_sample_flag = config['ac_down_sample'][i],
                                                  ))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(config['ac_mid_channels']) - 1):
            self.encoder_mids.append(mid_block(in_channels = config['ac_mid_channels'][i],
                                               out_channels = config['ac_mid_channels'][i+1],
                                               norm_channels = config['ac_norm_channels'],
                                               num_layers = config['ac_num_mid_layers'],
                                               self_attn_flag = config['ac_mid_attn'][i],
                                               num_heads = config['ac_num_heads']))
        
        self.encoder_out = nn.Sequential(nn.GroupNorm(num_groups = config['ac_norm_channels'],
                                                      num_channels = config['ac_mid_channels'][-1]),
                                         nn.SiLU(),
                                         nn.Conv2d(in_channels = config['ac_mid_channels'][-1],
                                                   out_channels = config['z_channels'],
                                                   kernel_size = 3,
                                                   stride = 1,
                                                   padding = 1))
        
        ############# Quantization #############
        self.prequant_conv = nn.Conv2d(in_channels = config['z_channels'],
                                       out_channels = config['z_channels'],
                                       kernel_size = 1,
                                       stride = 1,
                                       padding = 0)
        
        self.codebook = nn.Embedding(num_embeddings = config['codebook_size'],
                                     embedding_dim = config['z_channels'])
        
        ############# Decoder #############
        self.postquant_conv = nn.Conv2d(in_channels = config['z_channels'],
                                        out_channels = config['z_channels'],
                                        kernel_size = 1,
                                        stride = 1,
                                        padding = 0)
        
        # Attributes for up blocks, reverse to the down blocks
        ac_up_channels = list(reversed(config['ac_down_channels']))
        ac_up_mid_channels = list(reversed(config['ac_mid_channels']))
        ac_up_sample = list(reversed(config['ac_down_sample']))
        
        self.decoder_in_conv = nn.Conv2d(in_channels = config['z_channels'],
                                         out_channels = ac_up_channels[0],
                                         kernel_size = 3,
                                         stride = 1,
                                         padding = 1)
        
        self.decoder_mids = nn.ModuleList([])
        for i in range(len(ac_up_mid_channels) - 1):
            self.decoder_mids.append(mid_block(in_channels = ac_up_mid_channels[i],
                                               out_channels = ac_up_mid_channels[i+1],
                                               norm_channels = config['ac_norm_channels'],
                                               num_layers = config['ac_num_mid_layers'],
                                               self_attn_flag = config['ac_mid_attn'][i],
                                               num_heads = config['ac_num_heads']))
        
        self.decoder_blocks = nn.ModuleList([])
        for i in range(len(ac_up_channels) - 1):
            self.decoder_blocks.append(up_block(in_channels = ac_up_channels[i],
                                                out_channels = ac_up_channels[i+1],
                                                norm_channels = config['ac_norm_channels'],
                                                num_layers = config['ac_num_up_layers'],
                                                up_sample_flag = ac_up_sample[i]))
        
        self.decoder_out = nn.Sequential(nn.GroupNorm(num_groups = config['ac_norm_channels'],
                                                      num_channels = ac_up_channels[-1]),
                                         nn.SiLU(),
                                         nn.Conv2d(in_channels = ac_up_channels[-1],
                                                   out_channels = config['im_channels'],
                                                   kernel_size = 3,
                                                   stride = 1,
                                                   padding = 1))
    
    def encode(self, x):
        
        """
        Perform the action of encoding and quantizing an image into tensor in the shape of B x z_ch x h x w
        
        
        Args:
            x (float tensor): A batch of input images, on cuda device, in the range of -1 to 1, in the shape of B x im_channels x H x W in the latent space
        
        Returns:
            quant_out (float tensor): The resultant latent data, on cuda device, in the shape of B x z_ch x h x w \n
            loss (dict) : A dictionary with the keys of 'codebook_loss' and 'commitment_loss' that returns a float tensor each \n
            closest_idx (int tensor): The index of the resultant latent data in reference to the codebook, on cuda device, in the shape of B x h x w \n
        """
        
        # Encode
        out = x
        out = self.enc_in_conv(out)
        for block in self.encoder_blocks:
            out = block(out)
        for block in self.encoder_mids:
            out = block(out)
        out = self.encoder_out(out)
        
        # Quantize
        quant_in = out
        quant_in = self.prequant_conv(quant_in)
        
        B, z_ch, H, W = quant_in.shape
        quant_in = quant_in.reshape(B, z_ch, H*W)
        quant_in = quant_in.transpose(1,2)
        
        dist = torch.cdist(x1 = quant_in,
                           x2 = self.codebook.weight[None, :, :].repeat(B, 1, 1)) # B x (H*W) x cb_size
        closest_idx = torch.argmin(dist, dim=2) # B x (H*W)
        
        quant_out = torch.index_select(self.codebook.weight, 0, closest_idx.reshape(-1)) # (B*H*W) x z_channels
        quant_out = quant_out.reshape(B, H*W, z_ch) # B x (H*W) x z_channels
        
        # out -> B z_channels H W
        out = out.reshape(B, z_ch, H*W) # B x z_channels x (H*W)
        out = out.transpose(1,2) # B x (H*W) x z_channels
        
        codebook_loss = torch.mean((quant_out - out.detach()) ** 2)
        commitment_loss = torch.mean((quant_out.detach() - out) ** 2)
        loss = {'codebook_loss': codebook_loss,
                'commitment_loss': commitment_loss}
        
        # Reparameterization trick
        quant_out = out + (quant_out - out).detach() # B x (H*W) x z_channels
        quant_out = quant_out.transpose(1,2) # B x z_channels x (H*W)
        quant_out = quant_out.reshape(B, z_ch, H, W) # B x z_channels x H x W
        
        # Reshape closest index
        closest_idx = closest_idx.reshape(B, H, W)
        
        return quant_out, loss, closest_idx
    
    def decode(self, x):
        
        """
        Perform the action of decoding a latent data back into images
        
        
        Args:
            x (float tensor): A batch of latent data, on cuda device, in the shape of B x z_channels x h x w in the latent space
        
        Returns:
            out (float tensor): The resultant decoded images, on cuda device, in the shape of B x im_channels x H x W \n
        
        """
        
        out = x
        out = self.postquant_conv(out)
        out = self.decoder_in_conv(out)
        for block in self.decoder_mids:
            out = block(out)
        for block in self.decoder_blocks:
            out = block(out)
        out = self.decoder_out(out)
        
        return out
    
    def forward(self, x):
        
        """
        Complete Forward Propagation of VQVAE
        
        
        Args:
            x (float tensor): A batch of input images, on cuda device, in the range of -1 to 1, in the shape of B x im_channels x H x W in the latent space
        
        Returns:
            out (float tensor): The resultant decoded images, on cuda device, in the shape of B x im_channels x H x W \n
            quant_out (float tensor): The resultant latent data, on cuda device, in the shape of B x z_ch x h x w \n
            loss (dict) : A dictionary with the keys of 'codebook_loss' and 'commitment_loss' that returns a float tensor each \n 
        """
        
        quant_out, loss, _ = self.encode(x)
        out = self.decode(quant_out)
        return out, quant_out, loss
        
        
    
        
        
            
            

