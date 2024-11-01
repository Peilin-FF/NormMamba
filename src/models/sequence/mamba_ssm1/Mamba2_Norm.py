# import sys
# print(sys.path)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
#from mamba_ssm.ops.triton.Batchnorm import BatchNorm
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin
import sys
sys.path.insert(0,"/root/autodl-tmp/MLRA/src/models/sequence/mamba_ssm1")
from NormSelection import NormSelection

class Mamba2_Norm(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=16,
        headdim=16,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        num_channels = 8,
        T = 5,
        num_groups = 4, 
        norm_type = "None",
        dim = 3
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        self.T = T
        self.num_groups = num_groups
        self.norm_type = norm_type
        self.num_channels = num_channels
        self.dim = 3
        self.eps = 1e-5
        self.momentum = 0.1
        self.affine = True
        self.mode = 0

        # Order: [z, x, B, C, dt]
        
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True
        
        print("norm_type:",norm_type)
        
        #self.norm = nn.BatchNorm1d(num_features=self.d_inner)
        #self.norm = nn.GroupNorm(num_groups=4,num_channels=self.d_inner)
        #self.norm = nn.InstanceNorm1d(num_features=self.d_inner,affine=True)
        #self.norm = BatchGroupNorm(num_features=self.d_inner, num_groups=4, num_channels=8, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0)
        #self.norm=ScaleNorm(num_features=self.d_inner, eps=1e-5)
        #self.norm = DBN(num_features=self.d_inner, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine)
        #self.norm = IterNorm(num_features=self.d_inner, num_groups=self.num_groups, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine)
        #self.norm = InstanceGroupItN(num_features=self.d_inner, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=0)
        #self.norm=ItNInstance(num_features=self.d_inner, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode),
        
        #None
        
        #self.norm = nn.LayerNorm(normalized_shape=self.d_inner,elementwise_affine=True)
        self.norm = RMSNormGated(hidden_size=self.d_inner)  
        
        
        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.dconv - 1):]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim), #if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            
            # y = y.permute(0,2,1)
            # y = self.norm(y)  # Apply GroupNorm here
            # y = y.permute(0,2,1)
            y = self.norm(y)  # Apply GroupNorm here
            
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
                
            out = self.out_proj(y)
        return out


    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Apply BatchNorm
        xBC = xBC.permute(0,2,1)
        xBC = self.norm(xBC.transpose(1, 2)).transpose(1, 2)
        xBC = xBC.permute(0,2,1)
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.norm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.norm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z, #if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.norm:
            y = y.permute(0,2,1)
            y = self.norm(y)
            y = y.permute(0,2,1)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
class BatchGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        """"""
        super(BatchGroupNorm, self).__init__()
        if num_channels > 0:
            assert num_features % num_channels == 0
            num_groups = num_features // num_channels
        if num_groups>num_features:
            num_groups=num_features
        assert num_features % num_groups == 0
        self.num_features = num_features
        self.num_groups = num_groups
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode
        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(self.num_groups))
        self.register_buffer('running_var', torch.ones(self.num_groups))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        training = self.mode > 0 or (self.mode == 0 and self.training)
        assert input.dim() == self.dim and input.size(1) == self.num_features
        sizes = input.size()
        reshaped = input.reshape(sizes[0] * sizes[1] // self.num_groups, self.num_groups, *sizes[2:self.dim])
        output = F.batch_norm(reshaped, self.running_mean, self.running_var, training=training, momentum=self.momentum,
                              eps=self.eps)
        output = output.view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)

class ScaleNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, dim=3, eps=1e-5, frozen=False, affine=True, *args, **kwargs):
        super(ScaleNorm, self).__init__()
        self.frozen = frozen
        self.num_features = num_features
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.shape = [1 for _ in range(dim)]
        self.shape[1] = self.num_features
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
        self.register_buffer('running_std', torch.zeros(self.num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
        self.running_std.fill_(1)

    def forward(self, input: torch.Tensor):
        assert input.size(1) == self.num_features and self.dim == input.dim()
        x = input.transpose(0, 1).contiguous().view(self.num_features, -1)
        if self.training and not self.frozen:
            std = x.std(-1, keepdim=True) + self.eps
            xn = x/std
            self.running_std = (1. - self.momentum) * self.running_std + self.momentum * std.data
        else:
            xn = x/self.running_std
        output = xn.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()
        if self.affine:
            output = output * self.weight
        return output

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, frozen={frozen}, affine={affine}'.format(**self.__dict__)

class DBN_Single(torch.nn.Module):
    def __init__(self, num_features, dim=3, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(DBN_Single, self).__init__()
        # assert dim == 4, 'DBN is not support 2D'
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_projection', torch.eye(num_features))


    def forward(self, X: torch.Tensor):
        x = X.transpose(0, 1).contiguous().view(self.num_features, -1)
        d, m = x.size()
        if self.training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            xc = x - mean
            # calculate covariance matrix
            sigma = torch.addmm(
                input=torch.eye(self.num_features).to(X), 
                mat1=xc, 
                mat2=xc.transpose(0, 1), 
                beta=self.eps, 
                alpha=1. / m
            )
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            u, eig, _ = sigma.svd()
           # eig, u = sigma.symeig(eigenvectors=True)
            scale = eig.rsqrt()
            #print(scale)
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm.data
        else:
            xc = x - self.running_mean
            wm = self.running_projection
        xn = wm.mm(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        return Xn

class DBN(torch.nn.Module):
    def __init__(self, num_features, num_channels=16, dim=3, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(DBN, self).__init__()
        # assert dim == 4, 'DBN is not support 2D'
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1 
        self.num_groups = num_groups
        self.DBN_Groups = torch.nn.ModuleList(
            [DBN_Single(num_features = self.num_channels, eps=eps, momentum=momentum) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.DBN_Groups.append(DBN_Single(num_features = num_channels_last, eps=eps, momentum=momentum))

      #  print('DBN by ZCA-------m_perGroup:' + str(self.num_channels) + '---nGroup:' + str(self.num_groups) + '---MM:' + str(self.momentum) + '---Affine:' + str(affine))
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.DBN_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, running_mean, running_wmat, num_features, T, eps, momentum, training):
        ctx.g = X.size(1) // num_features
        ctx.T = T  # 保存 T 到 ctx 中，以便在 backward 中使用
        x = X.transpose(0, 1).contiguous().view(num_features, -1)
        d, m = x.size()
        saved = []
        if training:
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            P = [None] * (T + 1)
            P[0] = torch.eye(d).to(X)
            Sigma = torch.addmm(torch.eye(num_features).to(X) * eps, xc, xc.transpose(0, 1), alpha=1. / m)
            rTr = (Sigma * P[0]).sum((0, 1), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(T):
                P[k + 1] = torch.addmm(P[k], torch.matrix_power(P[k], 3), Sigma_N, beta=1.5, alpha=-0.5)
            saved.extend(P)
            wm = P[T].mul_(rTr.sqrt())
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.mm(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_tensors
        xc = saved[0]
        rTr = saved[1]
        sn = saved[2].transpose(-2, -1)
        P = saved[3:]
        d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.mm(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]  # 使用在 forward 中保存的 ctx.T
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].mm(P[k - 1])
            g_sn += P2.mm(P[k - 1]).mm(g_P)
            g_tmp = g_P.mm(sn)
            g_P.addmm_(1.5, -0.5, g_tmp, P2)
            g_P.addmm_(1, -0.5, P2, g_tmp)
            g_P.addmm_(1, -0.5, P[k - 1].mm(g_tmp), P[k - 1])
        g_sn += g_P
        g_tr = ((-sn.mm(g_sn) + g_wm.transpose(-2, -1).mm(wm)) * P[0]).sum((0, 1), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        g_x = torch.addmm(wm.mm(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None




class IterNorm_Single(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=3, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm_Single, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_features))


    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_features, self.T,  self.eps, self.momentum, self.training)
        return X_hat

class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=3, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1
        self.num_groups = num_groups
        self.iterNorm_Groups = torch.nn.ModuleList(
            [IterNorm_Single(num_features = self.num_channels, eps=eps, momentum=momentum, T=T) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.iterNorm_Groups.append(IterNorm_Single(num_features = num_channels_last, eps=eps, momentum=momentum, T=T))
         
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)


    def reset_projection(self):
        for i in range(self.num_groups):
            self.iterNorm_Groups[i].running_mean.fill_(0)
            #self.iterNorm_Groups[i].running_wm = torch.eye(self.iterNorm_Groups[i].running_wm.size()[0]).to(self.iterNorm_Groups[i].running_wm)
            self.iterNorm_Groups[i].running_wm.fill_(0)

    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.iterNorm_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)

class InstanceGroupItN(nn.Module):
    def __init__(self, num_features, num_groups=32, T=5, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(InstanceGroupItN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        self.T = T
        if self.num_groups>num_features:
            self.num_groups=num_features
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

       # print('InstanceGroupItN --- num_groups=', self.num_groups, '--T=', self.T)
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, input: torch.Tensor):
        if input.numel() == 0: return input
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
        x = x.reshape(size[0], self.num_groups, -1)


        #x = input.view(size[0], self.num_groups, -1)

        IG, d, m = x.size()
        mean = x.mean(-1, keepdim=True)
        x_mean = x - mean
        P = [torch.Tensor([]) for _ in range(self.T+1)]
        sigma = x_mean.matmul(x_mean.transpose(1, 2)) / m

        P[0] = torch.eye(d).to(x).expand(sigma.shape)
        M_zero = sigma.clone().fill_(0)
        trace_inv = torch.addcmul(M_zero, sigma, P[0]).sum((1, 2), keepdim=True).reciprocal_()
        sigma_N=torch.addcmul(M_zero, sigma, trace_inv)
        for k in range(self.T):
            P[k+1] = torch.baddbmm(
                input=P[k], 
                batch1=self.matrix_power3(P[k]), 
                batch2=sigma_N, 
                beta=1.5, 
                alpha=-0.5
            )
        wm = torch.addcmul(M_zero, P[self.T], trace_inv.sqrt())
        y = wm.matmul(x_mean)
        output = y.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
        output = output.view_as(input)

        #output = y.view_as(input)

        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)

class ItNInstance(nn.Module):
    def __init__(self, num_features, num_groups=32, T=5, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(ItNInstance, self).__init__()
        if num_channels > num_features:
            num_channels = num_features
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        self.T = T
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

       # print('ItNInstance --- num_groups=', self.num_groups, '--T=', self.T)
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.reshape(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])

        x = x.reshape(size[0] * self.num_groups, size[1] // self.num_groups, -1)

        #print(x.size())
        IG, d, m = x.size()
        mean = x.mean(-1, keepdim=True)
        x_mean = x - mean
        P = [torch.Tensor([]) for _ in range(self.T+1)]
        sigma = x_mean.matmul(x_mean.transpose(1, 2)) / m

        P[0] = torch.eye(d).to(x).expand(sigma.shape)
        M_zero = sigma.clone().fill_(0)
        trace_inv = torch.addcmul(M_zero, sigma, P[0]).sum((1, 2), keepdim=True).reciprocal_()
        sigma_N=torch.addcmul(M_zero, sigma, trace_inv)
        for k in range(self.T):
            P[k + 1] = torch.baddbmm(P[k], self.matrix_power3(P[k]), sigma_N, beta=1.5, alpha=-0.5)
        wm = torch.addcmul(M_zero, P[self.T], trace_inv.sqrt())
        y = wm.matmul(x_mean)
        output = y.view(size[0], self.num_groups, size[1] // self.num_groups, *size[2:])
        output = output.view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps},' \
               ' momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)