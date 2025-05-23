import torch
import torch.nn as nn


def sinusoidal_embedding(time_step, time_embedding_dim):
    
    """
    Function that converts a batch of time_step into sinusoidal embedding
    
    Args:
        time_step (float tensor): Expected in the shape of (B,) and on cuda device \n
        time_embedding_dim (int): Dimensions of each embedding \n
    
    Returns:
        out (float tensor): on the same device as time_step, in the shape of (B, time_embedding_dim)
    """
    
    assert time_embedding_dim % 2 == 0, print(f"[WARNING] Provided time embedding dimension is not divisible by 2.")
    
    device = time_step.device
    batch_size = time_step.shape[0]
    
    index = torch.arange(0, time_embedding_dim // 2)[None, :].to(device).repeat(batch_size, 1) # B x (time_embedding_dim / 2)
    factor = 2 * index / time_embedding_dim
    factor = 10000 ** factor # B x (time_embedding_dim / 2)
    
    sin = torch.sin(time_step[:, None] / factor) # B x (time_embedding_dim / 2)
    cosine = torch.cos(time_step[:, None] / factor) # B x (time_embedding_dim / 2)
    
    out = torch.cat([sin, cosine], dim=1)
    return out


class down_block(nn.Module):
    """
    A block that acts as a down sampling block and can consists of multiple layers
    
    Args:
        in_channels (int) : Channel dimensions of the input to this block \n
        out_channels (int) : Channel dimensions of the expected output from this block \n
        norm_channels (int) : Number of groups in the channel dimensions to be normalized, in_channels and out_channels must be divisible by norm_channels \n
        num_layers (int) : Number of layers within a block \n
        down_sample_flag (boolean): Decide if down sampling to be executed \n
        time_embedding_dim : Embedding dimension of the input time embedding \n
        self_attn_flag (boolean): Decide if including self attention mechanisms in each layer within the block \n
        num_heads (int) : Number of heads in the self attention and cross attention blocks \n
        cross_attn_flag (boolean) : Decide if including cross attention mechansims in each layer within the block \n
        context_dim (int): The dimension of the context embedding \n
    """
    
    def __init__(self, in_channels, out_channels, norm_channels, num_layers, down_sample_flag = False,
                 time_embedding_dim = None, self_attn_flag = False, num_heads = None, cross_attn_flag = False, context_dim = None):
        
        """
        Initialize down block
        """
        super().__init__()
        assert in_channels % norm_channels == 0 , "[WARNING] Input channel is not divisible by norm_channels"
        assert out_channels % norm_channels == 0, "[WARNING] Output_channel is not divisible by norm_channels"
        
        self.self_attn_flag = self_attn_flag
        self.cross_attn_flag = cross_attn_flag
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        
        self.first_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = norm_channels, 
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(num_layers)])
        
        if time_embedding_dim is not None:
            self.t_embed_layer = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_features = time_embedding_dim,
                              out_features = out_channels)
                    ) for _ in range(num_layers)])
        
        self.second_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(num_layers)])
        
        if self_attn_flag:
            assert num_heads is not None, "[WARNING] Num heads cannot be none with self attention blocks"
            
            self.self_attn_norm = nn.ModuleList([
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels)
                for _ in range(num_layers)])
            
            self.self_attn_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(num_layers)])
        
        if cross_attn_flag:
            assert context_dim is not None, "[WARNING] Context dimension cannot be none with cross attention blocks"
            assert num_heads is not None, "[WARNING] Num heads cannot be none with cross attention blocks"
            
            self.context_proj = nn.ModuleList([
                nn.Linear(in_features = context_dim,
                          out_features = out_channels)
                for _ in range(num_layers)])
            
            self.cross_attn_norm = nn.ModuleList([
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels)
                for _ in range(num_layers)])
            
            self.cross_attn_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(num_layers)])
        
        self.residual_conv_block = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(num_layers)])
        
        self.down_sample = nn.Conv2d(in_channels = out_channels,
                                     out_channels = out_channels,
                                     kernel_size = 4,
                                     stride = 2,
                                     padding = 1) if down_sample_flag else nn.Identity()
    
    def forward(self, x, time_embedding = None, context_embedding = None):
        
        """
        Forward propagation for down block.
        
        Args:
            x (float tensor): input image to the block, expected to be on cuda, in the shape of (B, in_channels ,H,W) \n
            time_embedding (float tensor): input time embedding to the block, expected to be on cuda, in the shape of (B,time_embedding_dim) \n
            context_embedding (float tensor): input context embedding (for cross attention), expected to be on cuda, in the shape of (B, token_length, context_dim) \n
        
        Returns:
            out (float tensor): output feature maps, on cuda, in the shape of (B, out_channels, H / 2, W / 2) if down_sample_flag else (B, out_channels, H, W)
        """
        
        out = x
        
        for i in range(self.num_layers):
            
            res_in = out
        
            out = self.first_resnet_block[i](out)
            if time_embedding is not None:
                out = out + self.t_embed_layer[i](time_embedding)[:, :, None, None]
            out = self.second_resnet_block[i](out)
            out = out + self.residual_conv_block[i](res_in)
            
            if self.self_attn_flag:
                B, C, H, W = out.shape
                attn_in = out
                
                attn_in = out.reshape(B, C, H*W)
                attn_in = self.self_attn_norm[i](attn_in)
                attn_in = attn_in.transpose(1,2)
                attn_out, _ = self.self_attn_block[i](attn_in, attn_in, attn_in)
                attn_out = attn_out.transpose(1,2)
                attn_out = attn_out.reshape(B, C, H, W)
                
                out = attn_out + out
            
            if self.cross_attn_flag:
                
                assert context_embedding is not None, "[WARNING] Context embedding cannot be none with the cross attention blocks"
                B, C, H, W = out.shape
                attn_in = out
                
                attn_in = out.reshape(B, C, H*W)
                attn_in = self.cross_attn_norm[i](attn_in)
                attn_in = attn_in.transpose(1,2)
                context_input = self.context_proj[i](context_embedding)
                attn_out, _ = self.cross_attn_block[i](attn_in, context_input, context_input)
                attn_out = attn_out.transpose(1,2)
                attn_out = attn_out.reshape(B, C, H, W)
                
                out = attn_out + out
        
        # Downscale
        out = self.down_sample(out)
        return out





class mid_block(nn.Module):
    
    """
    A block that acts as a mid block and can consists of multiple layers
    
    Args:
        in_channels (int) : Channel dimensions of the input to this block \n
        out_channels (int) : Channel dimensions of the expected output from this block \n
        norm_channels (int) : Number of groups in the channel dimensions to be normalized, in_channels and out_channels must be divisible by norm_channels \n
        num_layers (int) : Number of layers within a block \n
        time_embedding_dim : Embedding dimension of the input time embedding \n
        self_attn_flag (boolean): Decide if including self attention mechanisms in each layer within the block \n
        num_heads (int) : Number of heads in the self attention and cross attention blocks \n
        cross_attn_flag (boolean) : Decide if including cross attention mechansims in each layer within the block \n
        context_dim (int): The dimension of the context embedding \n
    """
    
    def __init__(self, in_channels, out_channels, norm_channels, num_layers,
                 time_embedding_dim = None, self_attn_flag = False, num_heads = None, cross_attn_flag = False, context_dim = None):
        
        """
        Initialize mid block
        """
        super().__init__()
        assert in_channels % norm_channels == 0, "[WARNING] Input channel is not divisible by norm_channels"
        assert out_channels % norm_channels == 0, "[WARNING] Output_channel is not divisible by norm_channels"
        
        self.self_attn_flag = self_attn_flag
        self.cross_attn_flag = cross_attn_flag
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        
        self.first_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = norm_channels, 
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(num_layers + 1)])
        
        if time_embedding_dim is not None:
            self.t_embed_layer = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_features = time_embedding_dim,
                              out_features = out_channels)
                    ) for _ in range(num_layers + 1)])
        
        self.second_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(num_layers + 1)])
        
        if self_attn_flag:
            assert num_heads is not None, "[WARNING] Num heads cannot be none with self attention blocks"
            
            self.self_attn_norm = nn.ModuleList([
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels)
                for _ in range(num_layers)])
            
            self.self_attn_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(num_layers)])
        
        if cross_attn_flag:
            assert context_dim is not None, "[WARNING] Context dimension cannot be none with cross attention blocks"
            assert num_heads is not None, "[WARNING] Num heads cannot be none with cross attention blocks"
            
            self.context_proj = nn.ModuleList([
                nn.Linear(in_features = context_dim,
                          out_features = out_channels)
                for _ in range(num_layers)])
            
            self.cross_attn_norm = nn.ModuleList([
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels)
                for _ in range(num_layers)])
            
            self.cross_attn_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(num_layers)])
        
        self.residual_conv_block = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(num_layers + 1)])
        
    
    def forward(self, x, time_embedding = None, context_embedding = None):
        
        """
        Forward propagation for mid block.
        
        Args:
            x (float tensor): input image to the block, expected to be on cuda, in the shape of (B, in_channels ,H, W) \n
            time_embedding (float tensor): input time embedding to the block, expected to be on cuda, in the shape of (B,time_embedding_dim) \n
            context_embedding (float tensor): input context embedding (for cross attention), expected to be on cuda, in the shape of (B, token_length, context_dim) \n
        
        Returns:
            out (float tensor): output feature maps, on cuda, in the shape of (B, out_channels, H, W)
        """
        
        out = x
        res_in = out
        
        out = self.first_resnet_block[0](out)
        if time_embedding is not None:
            out = out + self.t_embed_layer[0](time_embedding)[:, :, None, None]
        out = self.second_resnet_block[0](out)
        out = out + self.residual_conv_block[0](res_in)
        
        for i in range(self.num_layers):
            
            if self.self_attn_flag:
                B, C, H, W = out.shape
                attn_in = out
                
                attn_in = out.reshape(B, C, H*W)
                attn_in = self.self_attn_norm[i](attn_in)
                attn_in = attn_in.transpose(1,2)
                attn_out, _ = self.self_attn_block[i](attn_in, attn_in, attn_in)
                attn_out = attn_out.transpose(1,2)
                attn_out = attn_out.reshape(B, C, H, W)
                
                out = attn_out + out
            
            if self.cross_attn_flag:
                
                assert context_embedding is not None, "[WARNING] Context embedding cannot be none with the cross attention blocks"
                B, C, H, W = out.shape
                attn_in = out
                
                attn_in = out.reshape(B, C, H*W)
                attn_in = self.cross_attn_norm[i](attn_in)
                attn_in = attn_in.transpose(1,2)
                context_input = self.context_proj[i](context_embedding)
                attn_out, _ = self.cross_attn_block[i](attn_in, context_input, context_input)
                attn_out = attn_out.transpose(1,2)
                attn_out = attn_out.reshape(B, C, H, W)
                
                out = attn_out + out
                
            res_in = out
        
            out = self.first_resnet_block[i+1](out)
            if time_embedding is not None:
                out = out + self.t_embed_layer[i+1](time_embedding)[:, :, None, None]
            out = self.second_resnet_block[i+1](out)
            out = out + self.residual_conv_block[i+1](res_in)
        
        return out
            





class up_block(nn.Module):
    
    """
    A block that acts as a up sampling block and can consists of multiple layers.
    
    Args:
        in_channels (int) : Channel dimensions of the input to this block \n
        out_channels (int) : Channel dimensions of the expected output from this block \n
        norm_channels (int) : Number of groups in the channel dimensions to be normalized, in_channels and out_channels must be divisible by norm_channels \n
        num_layers (int) : Number of layers within a block \n
        up_sample_flag (boolean): Decide if up sampling to be executed \n
        time_embedding_dim : Embedding dimension of the input time embedding \n
        self_attn_flag (boolean): Decide if including self attention mechanisms in each layer within the block \n
        num_heads (int) : Number of heads in the self attention and cross attention blocks \n
        cross_attn_flag (boolean) : Decide if including cross attention mechansims in each layer within the block \n
        context_dim (int): The dimension of the context embedding \n
    """
    
    def __init__(self, in_channels, out_channels, norm_channels, num_layers, up_sample_flag = False,
                 time_embedding_dim = None, self_attn_flag = False, num_heads = None, cross_attn_flag = False, context_dim = None):
        
        """
        Initialize up block
        """
        super().__init__()
        assert in_channels % norm_channels == 0 , "[WARNING] Input channel is not divisible by norm_channels"
        assert out_channels % norm_channels == 0, "[WARNING] Output_channel is not divisible by norm_channels"
        
        self.self_attn_flag = self_attn_flag
        self.cross_attn_flag = cross_attn_flag
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        
        self.first_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = norm_channels, 
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(num_layers)])
        
        if time_embedding_dim is not None:
            self.t_embed_layer = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_features = time_embedding_dim,
                              out_features = out_channels)
                    ) for _ in range(num_layers)])
        
        self.second_resnet_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(num_layers)])
        
        if self_attn_flag:
            assert num_heads is not None, "[WARNING] Num heads cannot be none with self attention blocks"
            
            self.self_attn_norm = nn.ModuleList([
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels)
                for _ in range(num_layers)])
            
            self.self_attn_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(num_layers)])
        
        if cross_attn_flag:
            assert context_dim is not None, "[WARNING] Context dimension cannot be none with cross attention blocks"
            assert num_heads is not None, "[WARNING] Num heads cannot be none with cross attention blocks"
            
            self.context_proj = nn.ModuleList([
                nn.Linear(in_features = context_dim,
                          out_features = out_channels)
                for _ in range(num_layers)])
            
            self.cross_attn_norm = nn.ModuleList([
                nn.GroupNorm(num_groups = norm_channels,
                             num_channels = out_channels)
                for _ in range(num_layers)])
            
            self.cross_attn_block = nn.ModuleList([
                nn.MultiheadAttention(embed_dim = out_channels,
                                      num_heads = num_heads,
                                      batch_first = True)
                for _ in range(num_layers)])
        
        self.residual_conv_block = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(num_layers)])
        
        self.up_sample = nn.ConvTranspose2d(in_channels = in_channels,
                                            out_channels = in_channels,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 1) if up_sample_flag else nn.Identity()
        
        self.up_sample_with_down_map = nn.ConvTranspose2d(in_channels = in_channels // 2,
                                                          out_channels = in_channels // 2,
                                                          kernel_size = 4,
                                                          stride = 2,
                                                          padding = 1) if up_sample_flag else nn.Identity()
    
    def forward(self, x, down_map = None, time_embedding = None, context_embedding = None):
        
        """
        Forward propagation for up block.
        
        Args:
            x (float tensor): input image to the block, expected to be on cuda, in the shape of (B, in_channels ,H,W) if down_map is None, else (B, in_channels/2, H, W) \n
            down_map (float tensor): input down map to the block, expected to be on cuda, in the shape (B, in_channels/2 , H, W) \n
            time_embedding (float tensor): input time embedding to the block, expected to be on cuda, in the shape of (B,time_embedding_dim) \n
            context_embedding (float tensor): input context embedding (for cross attention), expected to be on cuda, in the shape of (B, token_length, context_dim) \n
        
        Returns:
            out (float tensor): output feature maps, on cuda, in the shape of (B, out_channels, H, W) if down_sample_flag else (B, out_channels, 2*H, 2*W)
        """
        
        out = x
        
        # Up scale
        if down_map is None:
            out = self.up_sample(out)
        else:
            out = self.up_sample_with_down_map(out)
            out = torch.cat([out, down_map], dim = 1)
        
        for i in range(self.num_layers):
            
            res_in = out
        
            out = self.first_resnet_block[i](out)
            if time_embedding is not None:
                out = out + self.t_embed_layer[i](time_embedding)[:, :, None, None]
            out = self.second_resnet_block[i](out)
            out = out + self.residual_conv_block[i](res_in)
            
            if self.self_attn_flag:
                B, C, H, W = out.shape
                attn_in = out
                
                attn_in = out.reshape(B, C, H*W)
                attn_in = self.self_attn_norm[i](attn_in)
                attn_in = attn_in.transpose(1,2)
                attn_out, _ = self.self_attn_block[i](attn_in, attn_in, attn_in)
                attn_out = attn_out.transpose(1,2)
                attn_out = attn_out.reshape(B, C, H, W)
                
                out = attn_out + out
            
            if self.cross_attn_flag:
                
                assert context_embedding is not None, "[WARNING] Context embedding cannot be none with the cross attention blocks"
                B, C, H, W = out.shape
                attn_in = out
                
                attn_in = out.reshape(B, C, H*W)
                attn_in = self.cross_attn_norm[i](attn_in)
                attn_in = attn_in.transpose(1,2)
                context_input = self.context_proj[i](context_embedding)
                attn_out, _ = self.cross_attn_block[i](attn_in, context_input, context_input)
                attn_out = attn_out.transpose(1,2)
                attn_out = attn_out.reshape(B, C, H, W)
                
                out = attn_out + out

        return out
        
        
    