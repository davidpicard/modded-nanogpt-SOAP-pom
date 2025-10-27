import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import KVCache

import math
from copy import deepcopy

from models import compom
from models.rotary import Rotary, apply_rotary_emb


def rmsnorm(x0, eps=1e-3):
    """RMS normalization function (matching reference implementation)."""
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class RMSNorm(nn.Module):
    """RMS normalization module."""
    
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.type_as(x0)

#
# class Rotary(torch.nn.Module):
#     """Rotary position embeddings."""
#
#     def __init__(self, dim, base=10000):
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)
#         self.seq_len_cached = None
#         self.cos_cached = None
#         self.sin_cached = None
#
#     def forward(self, x):
#         seq_len = x.shape[1]
#         if seq_len != self.seq_len_cached:
#             self.seq_len_cached = seq_len
#             t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#             freqs = torch.outer(t, self.inv_freq).to(x.device)
#             self.cos_cached = freqs.cos()
#             self.sin_cached = freqs.sin()
#         return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
#
#
# def apply_rotary_emb(x, cos, sin):
#     """Apply rotary embeddings."""
#     assert x.ndim == 4  # multihead attention
#     d = x.shape[3]//2
#     x1 = x[..., :d]
#     x2 = x[..., d:]
#     y1 = x1 * cos + x2 * sin
#     y2 = x1 * (-sin) + x2 * cos
#     return torch.cat([y1, y2], 3)


class CausalSelfComPoM(nn.Module):
    """Causal self-attention using Polynomial Mixer."""

    def __init__(self, n_embd, degree, expand, n_head, n_groups, layernorm=False, use_rope: bool = True):
        super().__init__()
        self.degree = degree
        self.expand = expand
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_groups = n_groups
        self.head_dim = self.n_embd // self.n_head
        self.pom = compom.ComPoM(self.n_embd, self.degree, self.expand, self.n_groups, self.n_head, False, layernorm=layernorm, use_rope=use_rope)
        # self.rotary = Rotary(self.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0)
        # x = x.view(B, T, 1, C)
        # cos, sin = self.rotary(x)
        # x = apply_rotary_emb(x, cos, sin).view(B, T, C)
        return self.pom(x, x, mask)

    def ar_forward(self, xq, state):
        return self.pom.ar_forward(xq, state)

    def reset(self, state):
        return self.pom.reset(state)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, degree, expand, n_head, use_rope: bool = True, context_window=-1):
        super().__init__()
        self.degree = degree
        self.expand = expand
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.use_rope = use_rope
        if use_rope:
            self.rotary = Rotary(self.head_dim)
        self.context_window = context_window

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        if self.use_rope:
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        if self.context_window > 0:
            window_mask = torch.logical_xor(torch.ones(T, T, dtype=torch.bool).tril(diagonal=0), torch.ones(T, T, dtype=torch.bool).tril(diagonal=-self.context_window)).to(q.device)
            # print("***** using windowed mask!!!")
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False, attn_mask=window_mask)
        else:
            # mask = torch.tril(torch.ones((T, T))).unsqueeze(0).to(q.device)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

    @torch.no_grad
    def ar_forward(self, x, state):
        return self.ar_forward_kv(x, state)

    @torch.no_grad
    def ar_forward_x(self, x, state):
        state = deepcopy(state)
        B, T, D = x.shape
        if 'x_pred' in state:
            N = state['n']
            state['x_pred'][:, N:N+T, :] = x
            state['n'] = N+T
            x = state['x_pred'][:, 0:N+T, :]
        else:
            # print(f" MHA allocating x cache")
            state['x_pred'] = torch.zeros((B, state['max_len'], self.n_embd), dtype=x.dtype).to(x.device)
            state['x_pred'][:, 0:T, :] = x
            state['n'] = T
        _, N, _ = x.shape
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, N, self.n_head, self.head_dim)
        q = q.view(B, N, self.n_head, self.head_dim)
        v = v.view(B, N, self.n_head, self.head_dim)
        if self.use_rope:
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        if self.context_window > 0 and N > self.context_window:
            q = q[:, N-self.context_window:N, :]
            k = k[:, N-self.context_window:N, :]
            v = v[:, N-self.context_window:N, :]
            N = self.context_window
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        # print(f" MHA y: {y.shape} T: {T} N: {N}")
        y = y[:,:,N-T:N,:]
        y = y.transpose(1, 2).contiguous().view(B, T, D) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, state

    @torch.no_grad
    def ar_forward_kv(self, x, state):
        state = deepcopy(state)
        B, T, D = x.shape
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        if 'kv_cache' in state:
            N = state['n']
            state['kv_cache'][0, :, N:N + T, :, :] = k.unsqueeze(0)
            state['kv_cache'][1, :, N:N + T, :, :] = v.unsqueeze(0)
            k = state['kv_cache'][0, :, 0:N + T, :, :]
            v = state['kv_cache'][1, :, 0:N + T, :, :]
            N = N+T
            state['n'] = N
        else:
            # print(f" MHA allocating x cache")
            state['kv_cache'] = torch.zeros((2, B, state['max_len'], self.n_head, self.head_dim), dtype=q.dtype).to(q.device)
            state['kv_cache'][0, :, 0:T, :, :] = k.unsqueeze(0)
            state['kv_cache'][1, :, 0:T, :, :] = v.unsqueeze(0)
            state['n'] = T
            N = T
        if self.use_rope:
            current_pos = torch.arange(N-T, N, dtype=torch.long, device = q.device)
            cos, sin = self.rotary.position_forward(current_pos, state['max_len'], device=q.device)
            q = apply_rotary_emb(q, cos, sin)
            current_pos = torch.arange(0, N, dtype=torch.long, device = k.device)
            cos, sin = self.rotary.position_forward(current_pos, state['max_len'], device=k.device)
            # print(f" MHA k: {k.shape} cos: {cos.shape} sin: {sin.shape}")
            k = apply_rotary_emb(k, cos, sin)
        if self.context_window > 0 and N > self.context_window:
            k = k[:, N - self.context_window:N, :, :]
            v = v[:, N - self.context_window:N, :, :]
            N = self.context_window
        mask = torch.ones(N, N, dtype=torch.bool).tril(diagonal=0)[N-T:N, :].unsqueeze(0).to(q.device)
        # print(f" MHA q: {q.shape} k: {k.shape} v: {v.shape} m: {mask}")
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        # print(f" MHA y: {y.shape} T: {T} N: {N}")
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, state

    def reset(self, state):
        state['n'] = 0
        self.rotary.forward(torch.arange(0, state['max_len'], dtype=torch.long).unsqueeze(0).to(self.rotary.inv_freq.device))
        return state


class MLP(nn.Module):
    """Multi-layer perceptron block."""
    
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with PoM attention and MLP."""
    
    def __init__(self, mixing_layer, n_embd, n_layer):
        super().__init__()
        self.attn = deepcopy(mixing_layer) #CausalSelfPoM(n_embd, degree, expand, n_head)
        self.mlp = MLP(n_embd)
        # Reinitialize with pytorch defaults
        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        #         if module.bias is not None:
        #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #             nn.init.uniform_(module.bias, -bound, bound)
        #     elif isinstance(module, nn.Conv1d):
        #         nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        #         if module.bias is not None:
        #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #             nn.init.uniform_(module.bias, -bound, bound)
        #     elif isinstance(module, nn.Embedding):
        #         nn.init.normal_(module.weight, mean=0, std=1)
        # self.attn_scale = (1 / (2 * n_layer)**0.5)
        self.attn_scale = nn.Parameter(torch.ones((1, 1, n_embd))*(1 / (2 * n_layer)**0.5), requires_grad=True)
        self.mlp_scale = nn.Parameter(torch.ones((1, 1, n_embd)) * (1 / (2 * n_layer) ** 0.5), requires_grad=True)
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights_)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp_scale * self.mlp(rmsnorm(x))
        return x

    @torch.no_grad
    def ar_forward(self, x: torch.Tensor, state):
        dx, state = self.attn.ar_forward(rmsnorm(x), state)
        x = x + self.attn_scale * dx
        x = x + self.mlp_scale * self.mlp(rmsnorm(x))
        return x, state

    def reset(self, state):
        state = self.attn.reset(state)
        return state


class GPT(nn.Module):
    """GPT model with Polynomial Mixer attention."""
    
    def __init__(self, mixing_layer, vocab_size: int = 50257, seq_length: int = 1024, n_layer: int = 12, n_head: int = 12, n_embd: int = 768, use_rope: bool = True, hybrid: int = 0,
                 context_window: int = -1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        self.use_rope = use_rope
        self.hybrid = hybrid

        if use_rope:
            m = []
            for i in range(self.n_layer):
                if (self.hybrid>0) and (i%self.hybrid) == self.hybrid-1:
                    b = Block(CausalSelfAttention(n_embd=self.n_embd, degree=2, expand=2, n_head=self.n_embd//64, use_rope=True, context_window=context_window), self.n_embd, n_layer)
                    m.append(b)
                    print(f"Layer {i}: {m[-1]}")
                else:
                    b = Block(mixing_layer, self.n_embd, self.n_layer)
                    m.append(b)
                    print(f"Layer {i}: {b}")
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(self.vocab_size, self.n_embd),
                h=nn.ModuleList(m),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(self.vocab_size, self.n_embd),
                wpe=nn.Embedding(self.seq_length, self.n_embd),
                h=nn.ModuleList([Block(mixing_layer, self.n_embd, self.n_layer) for _ in range(self.n_layer)]),
            ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_logits: bool = True):
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices for loss computation
            return_logits: Whether to return logits
            
        Returns:
            Tuple of (logits, loss) if targets provided, else just logits
        """
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if not self.use_rope:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = x + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad
    def ar_forward(self, idx, state):
        b, t = idx.size()
        x = self.transformer.wte(idx)
        if not self.use_rope:
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device) + state[0]['n']
            pos_emb = self.transformer.wpe(pos).view(1, t, self.n_embd)
            x = x + pos_emb

        new_state = []
        for l in range(len(self.transformer.h)):
            x, s = self.transformer.h[l].ar_forward(x, state[l])
            new_state.append(s)
        x = rmsnorm(x)
        logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
        logits = logits.float()  # use tf32/fp32 for logits
        return logits, new_state

    def reset(self, batch_size):
        state = []
        for l in range(len(self.transformer.h)):
            s = {'bs': batch_size, 'max_len': self.seq_length}
            state.append(self.transformer.h[l].reset(s))
        return state

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, precondition_frequency: int =1):
        """
        Configure optimizers for the model.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam betas
            
        Returns:
            Combined optimizer
        """
        from torch.optim import AdamW   
        # from models.optimizers.soap import SOAP  # Import raw SOAP class
        
        # # Create optimizers for different parameter groups
        # optimizers = []
        
        # # AdamW for lm_head parameters (wte weights are tied to lm_head, so we only include lm_head)
        # lm_head_optimizer = AdamW(
        #     self.lm_head.parameters(),
        #     lr=learning_rate,
        #     betas=betas,
        #     weight_decay=0  # No weight decay for lm_head
        # )
        # optimizers.append(lm_head_optimizer)
        # # SOAP for transformer layers - use raw SOAP class like reference
        # transformer_optimizer = SOAP(
        #     self.transformer.h.parameters(),
        #     lr=learning_rate,
        #     betas=(0.95, 0.95),  # Fixed betas for SOAP
        #     weight_decay=weight_decay,  # No weight decay for transformer layers
        #     precondition_frequency=precondition_frequency  # Fixed precondition frequency
        # )
        # transformer_optimizer = AdamW(
        #     self.transformer.h.parameters(),
        #     lr=learning_rate,
        #     betas=(0.9, 0.9999),
        #     weight_decay=weight_decay
        # )
        # optimizers.append(transformer_optimizer)

        if self.use_rope:
            optimizer = AdamW([{
                'params': self.lm_head.parameters(),
                'lr': learning_rate,
                'betas': betas,
                'weight_decay': 0
            },
            {
                'params': self.transformer.h.parameters(),
                'lr': learning_rate,
                'betas': betas,
                'weight_decay': weight_decay
            }])
        else:
            optimizer = AdamW([{
                'params': self.lm_head.parameters(),
                'lr': learning_rate,
                'betas': betas,
                'weight_decay': 0
            },
            {
                'params': self.transformer.wpe.parameters(),
                'lr': learning_rate,
                'betas': betas,
                'weight_decay': 0
            },
            {
                'params': self.transformer.h.parameters(),
                'lr': learning_rate,
                'betas': betas,
                'weight_decay': weight_decay
            }])
        
        return optimizer