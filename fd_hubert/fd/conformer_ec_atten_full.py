import torch

from torch import nn
import math
from functools import partial
from einops import rearrange, repeat
# from utils.hparams import hparams
from torch.nn.utils import weight_norm
from local_attention import LocalAttention
import torch.nn.functional as F

from .SWN import SwitchNorm1d


#import fast_transformers.causal_product.causal_product_cuda



def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()
def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""
    
    def __init__(self, 
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout,channels,encoder_hidden):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.encoder_hidden=encoder_hidden
        self.channels=channels

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])
        self.poe = RelPositionalEncoding(self.channels, dropout_rate=0.3)
        
    #  METHODS  ########################################################################################################
    
    def forward(self, phone,):
        
        # apply all layers to the input
        for (i, layer) in enumerate(self._layers):
            # phone = self.poe(phone)
            phone = layer(phone)
        # provide the final sequence
        return phone


# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """
    
    def __init__(self, parent: PCmer):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        self.diffusion_embedding = nn.Conv1d(1, parent.channels, 1)

        # condition
        self.conditioner_projection = nn.Conv1d(parent.encoder_hidden, parent.channels, 1)
        
        
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.norms = nn.LayerNorm(parent.dim_model)
        self.normss = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)

        
        # selfatt -> fastatt: performer!
        # self.attn = SelfAttention(dim = parent.dim_model,
        #                           heads = parent.num_heads,
        #                           causal = False)
        self.kvatt=kvSelfAttention(dim = parent.dim_model,
                                  heads = parent.num_heads,
                                 )


        
    #  METHODS  ########################################################################################################

    def forward(self, phone,):
        # phone=phone.transpose(1, 2)
        # phone=phone.transpose(1, 2)
        # compute attention sub-layer

        phone = self.norms(phone + (self.kvatt(phone,)))

        
        phone = phone + (self.conformer(phone))

        
        return phone






class kvSelfAttention(nn.Module):
    def __init__(self, dim=1280 , heads=20, dim_head=64,
                 dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads


        self.heads = heads



        # print (heads, nb_features, dim_head)
        # name_embedding = torch.zeros(110, heads, dim_head, dim_head)
        # self.name_embedding = nn.Parameter(name_embedding, requires_grad=True)

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()
        # torch.nn.init.zeros_(self.name_embedding)
        # print (torch.sum(self.name_embedding))

    def forward(self, x):


        # cross_attend = exists(context)

        # context = default(context, x)
        # context_mask = default(context_mask, mask) if not cross_attend else context_mask
        # print (torch.sum(self.name_embedding))
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_flash=True, ):
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        return self.dropout(out)
@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            #nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            # nn.BatchNorm1d
            SwitchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def linear_attention(q, k, v):
    if v is None:
        #print (k.size(), q.size())
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out

    else:
        k_cumsum = k.sum(dim = -2) 
        #k_cumsum = k.sum(dim = -2)
        D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)

        context = torch.einsum('...nd,...ne->...de', k, v)
        #print ("TRUEEE: ", context.size(), q.size(), D_inv.size())
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out



class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x) + self.dropout(pos_emb)

class Conformer(nn.Module):
    def __init__(self, in_dims=768):
        super().__init__()
        
        self.encoder_hidden = 768
        self.layers = 2
        self.channels = 1280
        
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(in_dims, self.channels, 1),
                nn.ReLU(),
                nn.Conv1d(self.channels, self.channels, 1)) 
                
        # diffusion step
        self.diffusion_embedding = nn.Conv1d(1, self.channels, 1)
        
        # condition
        self.conditioner_projection = nn.Conv1d(self.encoder_hidden, self.channels, 1)

        
        # transformer
        self.decoder = PCmer(
            num_layers=self.layers,
            num_heads=20,
            dim_model=self.channels,
            dim_keys=self.channels,
            dim_values=self.channels,
            residual_dropout=0.1,
            attention_dropout=0.1,channels=self.channels,encoder_hidden=self.encoder_hidden)
            
        # out
        self.norm = nn.LayerNorm(self.channels)
        self.dense_out = weight_norm(nn.Linear(self.channels, in_dims))


    def forward(self, spec, ):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        # x = self.stack(spec[:, 0])  # B, C, T
        #x = x + self.diffusion_embedding(diffusion_step.unsqueeze(-1).unsqueeze(-1) / 1000) + self.conditioner_projection(cond) # B, C, T
        # confition=self.diffusion_embedding(diffusion_step.unsqueeze(-1).unsqueeze(-1) / 1000) + self.conditioner_projection(cond)
        x=self.conditioner_projection(spec)


        x = self.decoder(x.transpose(1, 2)) # B, T, C
        # x = self.norm(x) # B, T, C
        x = self.dense_out(x) # B, T, M
        x = x.transpose(1, 2).unsqueeze(1) # B, 1, M ,T
        return x
