import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class Config:
    # system
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    # Add these to your Config class:
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn
    # model
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.2
    bias = False
    
    # dataset
    batch_size = 16
    block_size = 256
    data_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_size = 50257  # GPT-2 vocab size
    
    # training
    max_iters = 100000
    learning_rate = 10e-4
    weight_decay = 1e-2
    grad_clip = 1.0
    eval_interval = 500
    eval_iters = 200
    warmup_iters = 200
    lr_decay_iters = 100000
    min_lr = 6e-5
    decay_lr = False  # whether to decay learning rate
    
    # checkpointing
    always_save_checkpoint = True
    init_from = 'scratch'

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, split):
        self.split = split
        self.data = np.memmap(os.path.join(Config.data_dir, f'{split}.bin'), 
                            dtype=np.uint16, mode='r')
    
    def __len__(self):
        return len(self.data) - Config.block_size
    
    def get_batch(self, batch_size):
        ix = torch.randint(len(self) - Config.block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+Config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+Config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(Config.device), y.to(Config.device)
        return x, y

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.n_embd = config.n_embd  # <-- Add this line to store the embedding dimension
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # <-- Now this will work
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = F.dropout(self.c_proj(y), p=self.dropout, training=self.training)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        x = F.dropout(self.gelu(self.c_fc(x)), p=self.dropout, training=self.training)
        x = F.dropout(self.c_proj(x), p=self.dropout, training=self.training)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        
        # initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn # full param name
                
                # Skip duplicate parameters that are weight-tied
                if pn == 'weight' and isinstance(m, torch.nn.Linear) and mn == 'lm_head':
                    continue  # skip the lm_head.weight because it's tied to wte.weight
                
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                else:
                    # catch-all for any other parameters (shouldn't be any in this architecture)
                    no_decay.add(fpn)
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # manually ensure embeddings are in no_decay
        no_decay.add('transformer.wte.weight')
        no_decay.add('transformer.wpe.weight')
        
        # remove any overlap between sets
        decay.difference_update(no_decay)
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.95))
        return optimizer
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------
def get_lr(it):
    if not Config.decay_lr:
        return Config.learning_rate
    if it < Config.warmup_iters:
        return Config.learning_rate * it / Config.warmup_iters
    if it > Config.lr_decay_iters:
        return Config.min_lr
    decay_ratio = (it - Config.warmup_iters) / (Config.lr_decay_iters - Config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return Config.min_lr + coeff * (Config.learning_rate - Config.min_lr)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(Config.eval_iters)
        for k in range(Config.eval_iters):
            X, Y = loader.get_batch(Config.batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def train():
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train_loader = DataLoaderLite('train')
    val_loader = DataLoaderLite('val')
    
    model = GPT(Config)
    model.to(Config.device)
    

    optimizer = model.configure_optimizers()
    
    best_val_loss = float('inf')
    for iter in range(Config.max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter % Config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss and Config.always_save_checkpoint:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), 'best_model.pth')
        
        xb, yb = train_loader.get_batch(Config.batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
        optimizer.step()
    
    print("Training complete!")

if __name__ == '__main__':
    print("=== System Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("WARNING: Running on CPU - performance will be slow")
        print("Install CUDA toolkit if you have an NVIDIA GPU")
    
    train()