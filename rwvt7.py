import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

@dataclass
class RWKVConfig7:
    vocab_size: int = 50304
    n_embd: int = 768
    n_layer: int = 12
    dim_att: int = 768
    dim_ffn: int = 3072
    n_head: int = 12
    head_size: int = 64
    head_size_divisor: int = 8

class RWKVTimeMix7(nn.Module):
    def __init__(self, config: RWKVConfig7):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size

        # Parameters
        self.time_decay = mx.zeros((config.n_head, config.head_size))
        self.time_faaaa = mx.zeros((config.n_head, config.head_size))

        self.receptance = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.key = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.value = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.output = nn.Linear(config.dim_att, config.n_embd, bias=False)
        self.gate = nn.Linear(config.n_embd, config.dim_att, bias=False)

    def forward(self, x, state):
        B, T, C = x.shape
        H = self.n_head

        xx = mx.roll(x, shift=1, axis=-2)
        xr = x * self.time_decay + xx * (1 - self.time_decay)

        r = self.receptance(xr)
        k = self.key(xr)
        v = self.value(xr)
        g = mx.sigmoid(self.gate(xr))

        # Compute attention (RWKV-7 optimized)
        out = mx.zeros_like(x)
        for t in range(T):
            rt = r[:, t:t+1, :]
            kt = k[:, t:t+1, :]
            vt = v[:, t:t+1, :]

            attn = mx.matmul(kt, vt) * g[:, t:t+1, :]
            state = state * self.time_decay + attn
            out[:, t:t+1, :] = mx.sigmoid(rt) * state

        return self.output(out), state


class RWKVChannelMix7(nn.Module):
    def __init__(self, config: RWKVConfig7):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)

    def forward(self, x):
        xx = mx.roll(x, shift=1, axis=-2)
        k = mx.relu(self.key(x + xx)) ** 2
        return mx.sigmoid(self.receptance(x)) * self.value(k)


class RWKVBlock7(nn.Module):
    def __init__(self, config: RWKVConfig7):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.att = RWKVTimeMix7(config)
        self.ffn = RWKVChannelMix7(config)

    def forward(self, x, state):
        attn_out, state = self.att(self.ln1(x), state)
        x = x + attn_out
        return x + self.ffn(self.ln2(x)), state


class RWKV7(nn.Module):
    def __init__(self, config: RWKVConfig7):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [RWKVBlock7(config) for _ in range(config.n_layer)]
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, state):
        x = self.emb(x)
        for block in self.blocks:
            x, state = block(x, state)
        x = self.ln_out(x)
        return self.head(x), state
