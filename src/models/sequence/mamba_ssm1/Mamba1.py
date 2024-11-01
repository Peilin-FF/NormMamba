# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    
import sys
sys.path.insert(0,"/root/autodl-tmp/MLRA/src/models/sequence/mamba_ssm1")

from NormSelection import NormSelection

class Mamba1(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        T=5,
        num_channels=8,
        num_groups=4,  # New parameter for GroupNorm
        norm_type = "None",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        self.T = T
        self.num_groups = num_groups
        print("norm_type:",norm_type)
        self.norm_type = norm_type
        self.num_channels = num_channels
        self.affine = True
        self.eps = 1e-5
        self.dim = 3
        self.momentum = 0.1
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        

        #self.norm = nn.BatchNorm1d(num_features=self.d_inner)
        #self.norm = nn.GroupNorm(num_groups=64,num_channels=self.d_inner)
        #self.norm = nn.InstanceNorm1d(num_features=self.d_inner,affine=True)
        #self.norm = nn.LayerNorm(normalized_shape=self.d_inner,elementwise_affine=True)
        #self.norm =RMSNorm(hidden_size=self.d_inner)
        # self.norm_front = nn.BatchNorm1d(num_features=self.d_model)
        # self.norm_back = RMSNorm(hidden_size=self.d_inner)nn.BatchNorm1d(num_features=self.d_model)
        
        #self.norm_front =RMSNorm(hidden_size=self.d_inner)
        self.norm_back =RMSNorm(hidden_size=self.d_inner)
        

        
        
        
        #self.norm = BatchNormMean(num_features=self.d_inner)
        #self.norm = BatchNormMeanfire(num_features=self.d_inner, total_steps=steps_per_epoch)
        #self.norm = BatchNormL2(num_features=self.d_inner)
        #self.norm = BatchNormExp(num_features=self.d_inner)
        #self.norm = BatchNormSig(num_features=self.d_inner)
        #self.norm = BatchGroupNorm(num_features=self.d_inner, num_groups=4, num_channels=8, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0)
        #self.norm=ScaleNorm(num_features=self.d_inner, eps=1e-5)
        #self.norm = DBN(num_features=self.d_inner, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine)
        #self.norm = IterNorm(num_features=self.d_inner, num_groups=self.num_groups, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine)
        #self.norm = InstanceGroupItN(num_features=self.d_inner, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=0)
        #self.norm = ItNInstance(num_features=self.d_inner, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=0),
        #None
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        
        # hidden_states = self.norm_front(hidden_states)
        
        # hidden_states=hidden_states.permute(0,2,1)
        # hidden_states = self.norm_front(hidden_states)
        # hidden_states=hidden_states.permute(0,2,1)
        
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        self.use_fast_path = 0 #byforce
        
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        self.use_fast_path = 0
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")

            # y = y.permute(0,2,1)
            # y = self.norm_back(y)  # Apply GroupNorm here
            # y = y.permute(0,2,1)
            
            y = self.norm_back(y)  # Apply GroupNorm here
            
            out = self.out_proj(y)
            
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    def get_global_step(self):
        # 这里需要获取全局的训练步数
        # 如果您使用的是 PyTorch Lightning，可以从 trainer 中获取 global_step
        # 如果您不使用 Lightning，需要自行维护一个全局计数器

        # 示例：假设在模型中维护一个计数器
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        self.global_step += 1
        return self.global_step
if __name__ == '__main__' :
    
    x = torch.randn(2, 64, 32).to("cuda")
    main=Mamba1(d_model=32,expand=16).to("cuda")
    print(main)
    y=main(x)
    print(y.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Parameter

# class BatchNorm(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
#         super(BatchNorm, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine
        
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)
#         self.running_mean.zero_()
#         self.running_var.fill_(1)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device
#         self.running_mean = self.running_mean.to(device)
#         self.running_var = self.running_var.to(device)

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         if self.training:
#             mean = x.mean(dim=0)
#             var = x.var(dim=0, unbiased=False)

#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

#             x_hat = (x - mean) / torch.sqrt(var + self.eps)
#         else:
#             x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

#         output = x_hat.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}'

# class BatchNormAbs(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
#         super(BatchNormAbs, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine
        
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)
#         self.running_mean.zero_()
#         self.running_var.fill_(1)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device
#         self.running_mean = self.running_mean.to(device)
#         self.running_var = self.running_var.to(device)

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         # 计算最大绝对值
#         max_abs = torch.max(torch.abs(x), dim=0, keepdim=True)[0]

#         # 最大绝对值归一化
#         x_normalized = x / (max_abs + self.eps)

#         output = x_normalized.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}'

# class BatchNormMean(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
#         super(BatchNormMean, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine
        
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)
#         self.running_mean.zero_()
#         self.running_var.fill_(1)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device
#         self.running_mean = self.running_mean.to(device)
#         self.running_var = self.running_var.to(device)

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         # 计算均值和最小值
#         mean = x.mean(dim=0, keepdim=True)
#         min_val = x.min(dim=0, keepdim=True)[0]
#         max_val = x.max(dim=0, keepdim=True)[0]

#         # 均值归一化
#         x_normalized = (x - mean) / (max_val - min_val + self.eps)

#         output = x_normalized.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}' 

# import torch
# import torch.nn as nn
# import math

# class BatchNormMeanfire(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True, alpha_max=1.0, alpha_min=0.1, total_steps=1000):
#         super(BatchNormMeanfire, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine

#         if self.affine:
#             self.weight = nn.Parameter(torch.Tensor(num_features))
#             self.bias = nn.Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))

#         self.reset_parameters()

#         # 余弦退火参数
#         self.alpha_max = alpha_max
#         self.alpha_min = alpha_min
#         self.total_steps = total_steps  # 需要根据训练过程设定
#         self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)
#         self.running_mean.zero_()
#         self.running_var.fill_(1)

#     def set_global_step(self, global_step):
#         self.global_step = torch.tensor(global_step, dtype=torch.long, device=self.global_step.device)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device
#         self.running_mean = self.running_mean.to(device)
#         self.running_var = self.running_var.to(device)
#         self.global_step = self.global_step.to(device)

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         # 计算均值和最大最小值
#         mean = x.mean(dim=0, keepdim=True)
#         min_val = x.min(dim=0, keepdim=True)[0]
#         max_val = x.max(dim=0, keepdim=True)[0]

#         # 如果处于训练模式，计算 alpha
#         if self.training:
#             current_step = self.global_step.item()
#             T = self.total_steps
#             alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 + math.cos(math.pi * (current_step % T) / T))
#         else:
#             # 在评估模式下，使用固定的 alpha 值
#             alpha = self.alpha_min

#         # 均值归一化，加入 alpha 缩放
#         x_normalized = (x - mean) / ((max_val - min_val + self.eps) * alpha)

#         # 恢复输入的形状
#         output = x_normalized.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return (f'num_features={self.num_features}, momentum={self.momentum}, '
#                 f'eps={self.eps}, affine={self.affine}, '
#                 f'alpha_max={self.alpha_max}, alpha_min={self.alpha_min}, '
#                 f'total_steps={self.total_steps}')

        
# class BatchNormL2(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
#         super(BatchNormL2, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine
        
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.running_mean = torch.zeros(num_features)
#         self.running_var = torch.ones(num_features)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)
#         self.running_mean.zero_()
#         self.running_var.fill_(1)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device
#         self.running_mean = self.running_mean.to(device)
#         self.running_var = self.running_var.to(device)

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         # 计算 L2 范数
#         l2_norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)) + self.eps

#         # L2 归一化
#         x_normalized = x / l2_norm

#         output = x_normalized.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}'

# class BatchNormExp(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
#         super(BatchNormExp, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine
        
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         # 计算指数归一化
#         exp_x = torch.exp(x)
#         sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
#         x_normalized = exp_x / (sum_exp_x + self.eps)

#         output = x_normalized.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}'
# class BatchNormSig(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
#         super(BatchNormSig, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
#         self.affine = affine
        
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features))
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)

#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected input with 3 dimensions (batch, features, sequence length), but got {input.dim()}."
#         assert input.size(1) == self.num_features, f"Expected {self.num_features} features, but got {input.size(1)}."

#         # 确保所有张量在同一设备上
#         device = input.device

#         # 转置并重塑输入以适应批归一化
#         x = input.transpose(0, 1).contiguous().view(-1, self.num_features)

#         # Sigmoid 归一化
#         x_normalized = 1 / (1 + torch.exp(-x))

#         output = x_normalized.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()

#         if self.affine:
#             output = output * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

#         return output

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}'

# class BatchGroupNorm(nn.Module):
#     def __init__(self, num_features, num_groups=1, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
#                  *args, **kwargs):
#         """"""
#         super(BatchGroupNorm, self).__init__()
#         if num_channels > 0:
#             assert num_features % num_channels == 0
#             num_groups = num_features // num_channels
#         if num_groups>num_features:
#             num_groups=num_features
#         assert num_features % num_groups == 0
#         self.num_features = num_features
#         self.num_groups = num_groups
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.mode = mode
#         self.shape = [1] * dim
#         self.shape[1] = num_features

#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*self.shape))
#             self.bias = Parameter(torch.Tensor(*self.shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.register_buffer('running_mean', torch.zeros(self.num_groups))
#         self.register_buffer('running_var', torch.ones(self.num_groups))
#         self.reset_parameters()

#     def reset_running_stats(self):
#         self.running_mean.zero_()
#         self.running_var.fill_(1)

#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             #nn.init.uniform_(self.weight)
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)

#     def forward(self, input: torch.Tensor):
#         training = self.mode > 0 or (self.mode == 0 and self.training)
#         assert input.dim() == self.dim and input.size(1) == self.num_features
#         sizes = input.size()
#         reshaped = input.reshape(sizes[0] * sizes[1] // self.num_groups, self.num_groups, *sizes[2:self.dim])
#         output = F.batch_norm(reshaped, self.running_mean, self.running_var, training=training, momentum=self.momentum,
#                               eps=self.eps)
#         output = output.view_as(input)
#         if self.affine:
#             output = output * self.weight + self.bias
#         return output

#     def extra_repr(self):
#         return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                'mode={mode}'.format(**self.__dict__)

# class ScaleNorm(nn.Module):
#     def __init__(self, num_features, momentum=0.1, dim=3, eps=1e-5, frozen=False, affine=True, *args, **kwargs):
#         super(ScaleNorm, self).__init__()
#         self.frozen = frozen
#         self.num_features = num_features
#         self.momentum = momentum
#         self.dim = dim
#         self.eps = eps
#         self.shape = [1 for _ in range(dim)]
#         self.shape[1] = self.num_features
#         self.affine = affine
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*self.shape))
#         self.register_buffer('running_std', torch.zeros(self.num_features, 1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.affine:
#             nn.init.ones_(self.weight)
#         self.running_std.fill_(1)

#     def forward(self, input: torch.Tensor):
#         assert input.size(1) == self.num_features and self.dim == input.dim()
#         x = input.transpose(0, 1).contiguous().view(self.num_features, -1)
#         if self.training and not self.frozen:
#             std = x.std(-1, keepdim=True) + self.eps
#             xn = x/std
#             self.running_std = (1. - self.momentum) * self.running_std + self.momentum * std.data
#         else:
#             xn = x/self.running_std
#         output = xn.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()
#         if self.affine:
#             output = output * self.weight
#         return output

#     def extra_repr(self):
#         return '{num_features}, momentum={momentum}, frozen={frozen}, affine={affine}'.format(**self.__dict__)

# class DBN_Single(torch.nn.Module):
#     def __init__(self, num_features, dim=3, eps=1e-3, momentum=0.1, affine=True,
#                  *args, **kwargs):
#         super(DBN_Single, self).__init__()
#         # assert dim == 4, 'DBN is not support 2D'
#         self.eps = eps
#         self.momentum = momentum
#         self.num_features = num_features
#         self.affine = affine
#         self.dim = dim
#         shape = [1] * dim
#         shape[1] = self.num_features

#         self.register_buffer('running_mean', torch.zeros(num_features, 1))
#         # running whiten matrix
#         self.register_buffer('running_projection', torch.eye(num_features))


#     def forward(self, X: torch.Tensor):
#         x = X.transpose(0, 1).contiguous().view(self.num_features, -1)
#         d, m = x.size()
#         if self.training:
#             # calculate centered activation by subtracted mini-batch mean
#             mean = x.mean(-1, keepdim=True)
#             self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
#             xc = x - mean
#             # calculate covariance matrix
#             sigma = torch.addmm(
#                 input=torch.eye(self.num_features).to(X), 
#                 mat1=xc, 
#                 mat2=xc.transpose(0, 1), 
#                 beta=self.eps, 
#                 alpha=1. / m
#             )
#             # reciprocal of trace of Sigma: shape [g, 1, 1]
#             u, eig, _ = sigma.svd()
#            # eig, u = sigma.symeig(eigenvectors=True)
#             scale = eig.rsqrt()
#             #print(scale)
#             wm = u.matmul(scale.diag()).matmul(u.t())
#             self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm.data
#         else:
#             xc = x - self.running_mean
#             wm = self.running_projection
#         xn = wm.mm(xc)
#         Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
#         return Xn

# class DBN(torch.nn.Module):
#     def __init__(self, num_features, num_channels=16, dim=3, eps=1e-3, momentum=0.1, affine=True,
#                  *args, **kwargs):
#         super(DBN, self).__init__()
#         # assert dim == 4, 'DBN is not support 2D'
#         self.eps = eps
#         self.momentum = momentum
#         self.num_features = num_features
#         self.num_channels = num_channels
#         num_groups = (self.num_features-1) // self.num_channels + 1 
#         self.num_groups = num_groups
#         self.DBN_Groups = torch.nn.ModuleList(
#             [DBN_Single(num_features = self.num_channels, eps=eps, momentum=momentum) for _ in range(self.num_groups-1)]
#         )
#         num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
#         self.DBN_Groups.append(DBN_Single(num_features = num_channels_last, eps=eps, momentum=momentum))

#       #  print('DBN by ZCA-------m_perGroup:' + str(self.num_channels) + '---nGroup:' + str(self.num_groups) + '---MM:' + str(self.momentum) + '---Affine:' + str(affine))
#         self.affine = affine
#         self.dim = dim
#         shape = [1] * dim
#         shape[1] = self.num_features
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*shape))
#             self.bias = Parameter(torch.Tensor(*shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         # self.reset_running_stats()
#         if self.affine:
#             torch.nn.init.ones_(self.weight)
#             torch.nn.init.zeros_(self.bias)

#     def forward(self, X: torch.Tensor):
#         X_splits = torch.split(X, self.num_channels, dim=1)
#         X_hat_splits = []
#         for i in range(self.num_groups):
#             X_hat_tmp = self.DBN_Groups[i](X_splits[i])
#             X_hat_splits.append(X_hat_tmp)
#         X_hat = torch.cat(X_hat_splits, dim=1)
#         # affine
#         if self.affine:
#             return X_hat * self.weight + self.bias
#         else:
#             return X_hat

#     def extra_repr(self):
#         return '{num_features}, num_channels={num_channels}, eps={eps}, ' \
#                'momentum={momentum}, affine={affine}'.format(**self.__dict__)

# class iterative_normalization_py(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, X, running_mean, running_wmat, num_features, T, eps, momentum, training):
#         ctx.g = X.size(1) // num_features
#         ctx.T = T  # 保存 T 到 ctx 中，以便在 backward 中使用
#         x = X.transpose(0, 1).contiguous().view(num_features, -1)
#         d, m = x.size()
#         saved = []
#         if training:
#             mean = x.mean(-1, keepdim=True)
#             xc = x - mean
#             saved.append(xc)
#             P = [None] * (T + 1)
#             P[0] = torch.eye(d).to(X)
#             Sigma = torch.addmm(torch.eye(num_features).to(X) * eps, xc, xc.transpose(0, 1), alpha=1. / m)
#             rTr = (Sigma * P[0]).sum((0, 1), keepdim=True).reciprocal_()
#             saved.append(rTr)
#             Sigma_N = Sigma * rTr
#             saved.append(Sigma_N)
#             for k in range(T):
#                 P[k + 1] = torch.addmm(P[k], torch.matrix_power(P[k], 3), Sigma_N, beta=1.5, alpha=-0.5)
#             saved.extend(P)
#             wm = P[T].mul_(rTr.sqrt())
#             running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
#             running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
#         else:
#             xc = x - running_mean
#             wm = running_wmat
#         xn = wm.mm(xc)
#         Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
#         ctx.save_for_backward(*saved)
#         return Xn

#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         grad, = grad_outputs
#         saved = ctx.saved_tensors
#         xc = saved[0]
#         rTr = saved[1]
#         sn = saved[2].transpose(-2, -1)
#         P = saved[3:]
#         d, m = xc.size()

#         g_ = grad.transpose(0, 1).contiguous().view_as(xc)
#         g_wm = g_.mm(xc.transpose(-2, -1))
#         g_P = g_wm * rTr.sqrt()
#         wm = P[ctx.T]  # 使用在 forward 中保存的 ctx.T
#         g_sn = 0
#         for k in range(ctx.T, 1, -1):
#             P[k - 1].transpose_(-2, -1)
#             P2 = P[k - 1].mm(P[k - 1])
#             g_sn += P2.mm(P[k - 1]).mm(g_P)
#             g_tmp = g_P.mm(sn)
#             g_P.addmm_(1.5, -0.5, g_tmp, P2)
#             g_P.addmm_(1, -0.5, P2, g_tmp)
#             g_P.addmm_(1, -0.5, P[k - 1].mm(g_tmp), P[k - 1])
#         g_sn += g_P
#         g_tr = ((-sn.mm(g_sn) + g_wm.transpose(-2, -1).mm(wm)) * P[0]).sum((0, 1), keepdim=True) * P[0]
#         g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
#         g_x = torch.addmm(wm.mm(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
#         grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
#         return grad_input, None, None, None, None, None, None, None




# class IterNorm_Single(torch.nn.Module):
#     def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=3, eps=1e-5, momentum=0.1, affine=True,
#                  *args, **kwargs):
#         super(IterNorm_Single, self).__init__()
#         # assert dim == 4, 'IterNorm is not support 2D'
#         self.T = T
#         self.eps = eps
#         self.momentum = momentum
#         self.num_features = num_features
#         self.affine = affine
#         self.dim = dim
#         shape = [1] * dim
#         shape[1] = self.num_features

#         self.register_buffer('running_mean', torch.zeros(num_features, 1))
#         # running whiten matrix
#         self.register_buffer('running_wm', torch.eye(num_features))


#     def forward(self, X: torch.Tensor):
#         X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_features, self.T,  self.eps, self.momentum, self.training)
#         return X_hat

# class IterNorm(torch.nn.Module):
#     def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=3, eps=1e-5, momentum=0.1, affine=True,
#                  *args, **kwargs):
#         super(IterNorm, self).__init__()
#         # assert dim == 4, 'IterNorm is not support 2D'
#         self.T = T
#         self.eps = eps
#         self.momentum = momentum
#         self.num_features = num_features
#         self.num_channels = num_channels
#         num_groups = (self.num_features-1) // self.num_channels + 1
#         self.num_groups = num_groups
#         self.iterNorm_Groups = torch.nn.ModuleList(
#             [IterNorm_Single(num_features = self.num_channels, eps=eps, momentum=momentum, T=T) for _ in range(self.num_groups-1)]
#         )
#         num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
#         self.iterNorm_Groups.append(IterNorm_Single(num_features = num_channels_last, eps=eps, momentum=momentum, T=T))
         
#         self.affine = affine
#         self.dim = dim
#         shape = [1] * dim
#         shape[1] = self.num_features
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*shape))
#             self.bias = Parameter(torch.Tensor(*shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         # self.reset_running_stats()
#         if self.affine:
#             torch.nn.init.ones_(self.weight)
#             torch.nn.init.zeros_(self.bias)


#     def reset_projection(self):
#         for i in range(self.num_groups):
#             self.iterNorm_Groups[i].running_mean.fill_(0)
#             #self.iterNorm_Groups[i].running_wm = torch.eye(self.iterNorm_Groups[i].running_wm.size()[0]).to(self.iterNorm_Groups[i].running_wm)
#             self.iterNorm_Groups[i].running_wm.fill_(0)

#     def forward(self, X: torch.Tensor):
#         X_splits = torch.split(X, self.num_channels, dim=1)
#         X_hat_splits = []
#         for i in range(self.num_groups):
#             X_hat_tmp = self.iterNorm_Groups[i](X_splits[i])
#             X_hat_splits.append(X_hat_tmp)
#         X_hat = torch.cat(X_hat_splits, dim=1)
#         # affine
#         if self.affine:
#             return X_hat * self.weight + self.bias
#         else:
#             return X_hat

#     def extra_repr(self):
#         return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
#                'momentum={momentum}, affine={affine}'.format(**self.__dict__)

# class InstanceGroupItN(nn.Module):
#     def __init__(self, num_features, num_groups=32, T=5, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
#                  *args, **kwargs):
#         super(InstanceGroupItN, self).__init__()
#         if num_channels > 0:
#             num_groups = num_features // num_channels
#         self.num_features = num_features
#         self.num_groups = num_groups
#         self.T = T
#         if self.num_groups>num_features:
#             self.num_groups=num_features
#         assert self.num_features % self.num_groups == 0
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.mode = mode

#         self.shape = [1] * dim
#         self.shape[1] = num_features

#        # print('InstanceGroupItN --- num_groups=', self.num_groups, '--T=', self.T)
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*self.shape))
#             self.bias = Parameter(torch.Tensor(*self.shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()


#     def reset_parameters(self):
#         # self.reset_running_stats()
#         if self.affine:
#             #nn.init.uniform_(self.weight)
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)

#     def matrix_power3(self, Input):
#         B = torch.bmm(Input, Input)
#         return torch.bmm(B, Input)

#     def forward(self, input: torch.Tensor):
#         if input.numel() == 0: return input
#         size = input.size()
#         assert input.dim() == self.dim and size[1] == self.num_features
#         x = input.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
#         x = x.reshape(size[0], self.num_groups, -1)


#         #x = input.view(size[0], self.num_groups, -1)

#         IG, d, m = x.size()
#         mean = x.mean(-1, keepdim=True)
#         x_mean = x - mean
#         P = [torch.Tensor([]) for _ in range(self.T+1)]
#         sigma = x_mean.matmul(x_mean.transpose(1, 2)) / m

#         P[0] = torch.eye(d).to(x).expand(sigma.shape)
#         M_zero = sigma.clone().fill_(0)
#         trace_inv = torch.addcmul(M_zero, sigma, P[0]).sum((1, 2), keepdim=True).reciprocal_()
#         sigma_N=torch.addcmul(M_zero, sigma, trace_inv)
#         for k in range(self.T):
#             P[k+1] = torch.baddbmm(
#                 input=P[k], 
#                 batch1=self.matrix_power3(P[k]), 
#                 batch2=sigma_N, 
#                 beta=1.5, 
#                 alpha=-0.5
#             )
#         wm = torch.addcmul(M_zero, P[self.T], trace_inv.sqrt())
#         y = wm.matmul(x_mean)
#         output = y.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
#         output = output.view_as(input)

#         #output = y.view_as(input)

#         if self.affine:
#             output = output * self.weight + self.bias
#         return output

#     def extra_repr(self):
#         return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                'mode={mode}'.format(**self.__dict__)

# class ItNInstance(nn.Module):
#     def __init__(self, num_features, num_groups=32, T=5, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
#                  *args, **kwargs):
#         super(ItNInstance, self).__init__()
#         if num_channels > num_features:
#             num_channels = num_features
#         if num_channels > 0:
#             num_groups = num_features // num_channels
#         self.num_features = num_features
#         self.num_groups = num_groups
#         self.T = T
#         assert self.num_features % self.num_groups == 0
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.mode = mode

#         self.shape = [1] * dim
#         self.shape[1] = num_features

#        # print('ItNInstance --- num_groups=', self.num_groups, '--T=', self.T)
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(*self.shape))
#             self.bias = Parameter(torch.Tensor(*self.shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()


#     def reset_parameters(self):
#         # self.reset_running_stats()
#         if self.affine:
#             #nn.init.uniform_(self.weight)
#             nn.init.ones_(self.weight)
#             nn.init.zeros_(self.bias)

#     def matrix_power3(self, Input):
#         B = torch.bmm(Input, Input)
#         return torch.bmm(B, Input)

#     def forward(self, input: torch.Tensor):
#         size = input.size()
#         assert input.dim() == self.dim and size[1] == self.num_features
#         x = input.reshape(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])

#         x = x.reshape(size[0] * self.num_groups, size[1] // self.num_groups, -1)

#         #print(x.size())
#         IG, d, m = x.size()
#         mean = x.mean(-1, keepdim=True)
#         x_mean = x - mean
#         P = [torch.Tensor([]) for _ in range(self.T+1)]
#         sigma = x_mean.matmul(x_mean.transpose(1, 2)) / m

#         P[0] = torch.eye(d).to(x).expand(sigma.shape)
#         M_zero = sigma.clone().fill_(0)
#         trace_inv = torch.addcmul(M_zero, sigma, P[0]).sum((1, 2), keepdim=True).reciprocal_()
#         sigma_N=torch.addcmul(M_zero, sigma, trace_inv)
#         for k in range(self.T):
#             P[k + 1] = torch.baddbmm(P[k], self.matrix_power3(P[k]), sigma_N, beta=1.5, alpha=-0.5)
#         wm = torch.addcmul(M_zero, P[self.T], trace_inv.sqrt())
#         y = wm.matmul(x_mean)
#         output = y.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
#         output = output.view_as(input)
#         if self.affine:
#             output = output * self.weight + self.bias
#         return output

#     def extra_repr(self):
#         return '{num_features}, num_groups={num_groups}, eps={eps},' \
#                ' momentum={momentum}, affine={affine}, ' \
#                'mode={mode}'.format(**self.__dict__)