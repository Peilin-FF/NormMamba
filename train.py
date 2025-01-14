import os
import random
import time
from functools import wraps
from typing import Callable, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim.ema import build_ema_optimizer
from src.utils.optim_groups import add_optimizer_hooks
from pytorch_lightning.callbacks import EarlyStopping

from src.models.functional.misc import apply_rand_init
from src.models.lm.model import LM_MODEL_LIST
import logging
logging.basicConfig(level=logging.DEBUG)

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# learning rate monitor:
from pytorch_lightning.callbacks import LearningRateMonitor

import os
import numpy as np
import torch.distributed as dist

def write_gradients_to_files(norm_weights, last_norm_weights, n_layers, file_path):
    # Only proceed if this is the main process (rank 0)
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # Ensure the main directory exists
    os.makedirs(file_path, exist_ok=True)
    
    for layer_index in range(n_layers):
        # Create a separate folder for each layer
        layer_folder = os.path.join(file_path, f'layer_{layer_index}')
        os.makedirs(layer_folder, exist_ok=True)
        
        # Generate file paths for norms
        l1_norm_file = os.path.join(layer_folder, 'L1_norm.txt')
        l2_norm_file = os.path.join(layer_folder, 'L2_norm.txt')
        L1_norm_diff_out_proj_weight_norm_file = os.path.join(layer_folder, 'L1_norm_diff_out_proj_weight_norm.txt')
        L2_norm_diff_out_proj_weight_norm_file = os.path.join(layer_folder, 'L2_norm_diff_out_proj_weight_norm.txt')
        ratio_L1_norm_file = os.path.join(layer_folder, 'ratio_L1_norm.txt')
        ratio_L2_norm_file = os.path.join(layer_folder, 'ratio_L2_norm.txt')

        # Retrieve current layer weights
        current_weights = norm_weights[layer_index][1].cpu().numpy()

        # Write L1 and L2 norms of current weights
        L1_norm = np.sum(np.abs(current_weights))
        L2_norm = np.sqrt(np.sum(np.square(current_weights)))
        with open(l1_norm_file, 'a') as file:
            file.write(f"{L1_norm:.6f}\n")
        with open(l2_norm_file, 'a') as file:
            file.write(f"{L2_norm:.6f}\n")

        # Handle last_norm_weights if they exist
        if last_norm_weights and last_norm_weights[layer_index] and last_norm_weights[layer_index][1] is not None:
            last_weights = last_norm_weights[layer_index][1].cpu().numpy()
            L1_norm_last = np.sum(np.abs(last_weights)) + 1e-6  # Avoid division by zero
            L2_norm_last = np.sqrt(np.sum(np.square(last_weights))) + 1e-6  # Avoid division by zero

            # Calculate differences and write them
            diff_weights = current_weights - last_weights
            L1_norm_diff_weights = np.sum(np.abs(diff_weights))
            L2_norm_diff_weights = np.sqrt(np.sum(np.square(diff_weights)))
            with open(L1_norm_diff_out_proj_weight_norm_file, 'a') as file:
                file.write(f"{L1_norm_diff_weights:.6f}\n")
            with open(L2_norm_diff_out_proj_weight_norm_file, 'a') as file:
                file.write(f"{L2_norm_diff_weights:.6f}\n")

            # Calculate ratio norms and write them
            ratio_L1 = np.sum(np.abs(diff_weights / L1_norm_last))
            ratio_L2 = np.sqrt(np.sum(np.square(diff_weights / L2_norm_last)))
            with open(ratio_L1_norm_file, 'a') as file:
                file.write(f"{ratio_L1:.6f}\n")
            with open(ratio_L2_norm_file, 'a') as file:
                file.write(f"{ratio_L2:.6f}\n")




# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        self.setup()  ## Added by KS

        ## log tensors during training for some datasets
        self.log_tensors_for_validation = False
        self.log_tensors_freq = -1

        if self.device.index == 0 or self.device.index is None:
            self.manual_checkpoints = config.train.manual_checkpoints
            self.ckpt_milestones = config.train.ckpt_milestones
            if isinstance(self.ckpt_milestones, int):
                self.ckpt_milestones = [self.ckpt_milestones]
        else:
            self.manual_checkpoints = False

        if self.manual_checkpoints:
            if self.device.index == 0 or self.device.index is None:
                self.setup_checkpoint_dir()

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()
            self.restrict_trainset()  # for training on a smaller sample

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # LMs need the source vocab if exists
        if self.hparams.model._name_ in LM_MODEL_LIST:
            if hasattr(self.dataset, 'vocab'):
                # a hack to bypass omegaconf
                self.hparams.model.src_tokenizer = ''

        # Instantiate model
        self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # LMs need the source vocab if exists
        if self.hparams.model._name_ in LM_MODEL_LIST:
            if not getattr(self.model, 'pretrained', True):
                pass
            elif hasattr(self.dataset, 'vocab'):
                # a hack to bypass omegaconf
                self.model.src_tokenizer = self.dataset.vocab
                self.model._insert_special_tokens()


        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        #extract the modules so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

        # Handle state logic
        self._initialize_state()

    def load_state_dict(self, state_dict, strict=True):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            # Modify the checkpoint['state_dict'] inside model_state_hook e.g. to inflate 2D convs to 3D convs
            state_dict = model_state_hook(self.model, state_dict)

        print("Custom load_state_dict function is running.")

        # note, it needs to return something from the normal function we overrided
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
                (n := self.hparams.train.state.n_context) is None
                or isinstance(n, int)
                and n >= 0
        )
        assert (
                (n := self.hparams.train.state.n_context_eval) is None
                or isinstance(n, int)
                and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def on_epoch_start(self):
        self._initialize_state()

    def forward(self, batch):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x, w = self.encoder(x, **z)  # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = self.model(x, **w, state=self._state)
        self._state = state
        x, w = self.decoder(x, state=state, **z)
        return x, y, w
    
                
    def step(self, x_t):
        x_t, *_ = self.encoder(x_t)  # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        # x_t = x_t[:, None, ...] # Dummy length
        # x_t, *_ = self.decoder(x_t, state=state)
        # x_t = x_t[:, 0, ...]
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):

        self._process_state(batch, batch_idx, train=(prefix == "train"))

        x, y, w = self.forward(batch)

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        if self.log_tensors_for_validation:
            inputs = batch[0].clone().cpu().detach()
            labels = y.clone().cpu().detach()
            preds = x.clone().cpu().detach()
            self.log_inputs_outputs(inputs=inputs, labels=labels, preds=preds, prefix=prefix)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics: these are accumulated and logged at the end of epochs
        self.task.torchmetrics(x, y, prefix)

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

        self.log_tensors_for_validation = True if self.log_tensors_freq > 0 else False

    def on_train_epoch_end(self):
        # Log training torchmetrics
        self.log_dict(
            {f"train/{k}": v for k, v in self.task.get_torchmetrics("train").items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

        self.log_tensors_for_validation = True if self.log_tensors_freq > 0 else False

    def on_validation_epoch_end(self):
        # Log all validation torchmetrics
        for name in self.val_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        if self.manual_checkpoints:
            self.manual_checkpointing()


    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def on_test_epoch_end(self):
        # Log all test torchmetrics
        for name in self.test_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # 继续原来的代码
        self.log_dict({
            "trainer/loss": loss,
            "trainer/epoch": self.current_epoch
        }, on_step=True, on_epoch=False, prog_bar=False, add_dataloader_idx=False, sync_dist=True)

        # 记录其他模块的指标
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)
                

        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, add_dataloader_idx=False, sync_dist=True)

        # 每隔28个全局训练步骤执行权重记录
        if self.global_step % 28 == 0:
            out_proj_weights = [(f'Layer {i} out_proj weights:', layer.layer.mamba.out_proj.weight.data.clone().detach())
                                for i, layer in enumerate(self.model.layers) if hasattr(layer, 'layer') and hasattr(layer.layer, 'mamba') and hasattr(layer.layer.mamba.out_proj, 'weight')]

            # 确保last_norm_weights存在
            if not hasattr(self, 'last_norm_weights'):
                self.last_norm_weights = [None] * len(out_proj_weights)  # 初始化last_norm_weights
                    
            folder_path = '/root/autodl-tmp/cacul_results/'
            
            # 调用函数写入权重信息
            write_gradients_to_files(out_proj_weights, self.last_norm_weights, len(out_proj_weights), file_path=folder_path)

            # 更新存储的上一次权重信息
            self.last_norm_weights = [(i, weight[1]) for i, weight in enumerate(out_proj_weights)]

        return loss







    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
                self.val_loader_names[dataloader_idx].endswith("/ema")
                and self.optimizers().optimizer.stepped
        )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):

        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        
        
        
        params = [p for p in all_params if not hasattr(p, "_optim")]

        # Construct optimizer, add EMA if necessary
        if self.hparams.train.ema > 0.0:
            optimizer = utils.instantiate(
                registry.optimizer,
                self.hparams.optimizer,
                params,
                wrap=build_ema_optimizer,
                polyak=self.hparams.train.ema,
            )
        else:
            optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)
            
       
        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        ### Layer Decay ###

        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers: num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                            self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)

        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (eg if test is duplicate)
        if self.hparams.train.get("remove_test_loader_in_eval", None) is not None:
            return val_loader_names, val_loaders
        # default behavior is to add test loaders in eval
        else:
            return val_loader_names + test_loader_names, val_loaders + test_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders

    def log_inputs_outputs(self, inputs, labels, preds, prefix):
        """Log outputs to disk"""

        if not self.log_tensors_for_validation:
            return
        elif self.current_epoch == 0:
            return

        if self.current_epoch % self.log_tensors_freq == 0:
            prefix = prefix + f'/epoch_{self.current_epoch}'
            os.makedirs(prefix, exist_ok=True)
            prefix = prefix + '/'
        else:
            return

        device = preds.device
        prefix = prefix + f'device_{str(device)}_'

        num_tensors = min([4, len(inputs)])
        inputs = inputs[:num_tensors]
        labels = labels[:num_tensors]
        preds = preds[:num_tensors]
        torch.save(inputs, prefix + f'inputs.pt')
        torch.save(labels, prefix + f'labels.pt')
        torch.save(preds, prefix + f'preds.pt')

        # save the loss in a dict
        info_dict = self.trainer.callback_metrics
        torch.save(info_dict, prefix + f'info_dict.pt')

        self.log_tensors_for_validation = False

    def manual_checkpointing(self):
        """Custom checkpointing in case auto-checkpointing fails - should be called at end of every validation epoch"""
        # move everything to cpu for saving
        device = self.device
        self.cpu()

        # save last epoch checkpoint
        last_ckpt_path = self.manual_ckpt_path + '/last.ckpt'
        self.trainer.save_checkpoint(last_ckpt_path)

        # check if monitored metric is best available
        best_ckpt_path = self.manual_ckpt_path + '/val/best.ckpt'
        metric_name = self.hparams.train.monitor
        metric_val = self.trainer.callback_metrics[self.hparams.train.monitor]


        if self.best_metric_val is None or self.trainer.current_epoch == 0:
            self.best_metric_val = metric_val
            self.trainer.save_checkpoint(best_ckpt_path)
            print(f'achieved best {metric_name} of {metric_val} - saving checkpoint to: {best_ckpt_path}')
        else:
            mode = self.hparams.train.mode  # min or max
            compare = lambda x, y: x <= y if mode == 'min' else x >= y

            if compare(metric_val, self.best_metric_val):
                self.best_metric_val = metric_val
                self.trainer.save_checkpoint(best_ckpt_path)
                print(f'\nachieved best {metric_name} of {metric_val} - saving checkpoint to: {best_ckpt_path}')

        # check if using checkpointing milestones
        if self.ckpt_milestones is not None:
            if self.trainer.current_epoch in self.ckpt_milestones:
                milestone_ckpt_path = self.manual_ckpt_path + f'/milestone_epoch{self.trainer.current_epoch}.ckpt'
                self.trainer.save_checkpoint(milestone_ckpt_path)
                print(f'achieved milestone {self.trainer.current_epoch} - saving checkpoint to: {milestone_ckpt_path}')

        torch.cuda.empty_cache()

        # return to device
        self.to(device)


    def setup_checkpoint_dir(self):
        """Setup checkpoint directory"""
        # setup the checkpoint directory
        ckpt_dir_path = os.getcwd() + '/manual_checkpoints'
        if not os.path.exists(ckpt_dir_path):
            os.mkdir(ckpt_dir_path)

        # create specific instance version due to wandb crashes
        v = 0
        version = '/version_{}'
        ckpt_version_dir_path = ckpt_dir_path + version.format(v)
        while os.path.exists(ckpt_version_dir_path):
            v += 1
            ckpt_version_dir_path = ckpt_dir_path + version.format(v)
        os.mkdir(ckpt_version_dir_path)

        self.manual_ckpt_path = ckpt_version_dir_path
        os.mkdir(ckpt_version_dir_path + '/val')  # for monitoring best val performance

        self.best_metric_val = None

    def restrict_trainset(self):
        """Restrict trainset to a subset of the data"""
        if self.hparams.train.get('limit_train_samples', None) is None:
            return

        # get the number of samples to keep
        dataset = self.dataset.dataset_train
        is_subset = True
        try:  # restrict manually - dataset.dataset_train is already a Subset object
            indices = dataset.indices
        except AttributeError:
            is_subset = False
            indices = [i for i in range(len(dataset))]

        num_samples = self.hparams.train.limit_train_samples
        if num_samples <= 1:
            num_samples = float(num_samples)
            num_samples = int(num_samples * len(indices))
        else:
            num_samples = int(num_samples)

        # get the indices to keep
        indices = np.random.RandomState(seed=getattr(self, "seed", 42)).permutation(indices)[
                  :num_samples].tolist()

        if is_subset:
            self.dataset.dataset_train.indices = indices
        else:
            self.dataset.dataset_train = torch.utils.data.Subset(self.dataset.dataset_train, indices)


### pytorch-lightning utils and entrypoint ###

def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # add learning rate monitor:
    lr_monitor = LearningRateMonitor(logging_interval='step')  # or 'epoch'
    callbacks.append(lr_monitor)

    # Configure ddp automatically
    if config.trainer.devices > 1:
        print("ddp automatically configured, more than 1 gpu used!")
        kwargs["strategy"] = pl.strategies.DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=False,
        )
        
    # 确保 accelerator 参数正确设置
    if "accelerator" not in kwargs:
        if config.trainer.devices > 0:
            kwargs["accelerator"] = "gpu"  # 使用 GPU
        else:
            kwargs["accelerator"] = "cpu"  # 使用 CPU


    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    kwargs.update(config.trainer)
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=100,
        verbose=True,
        mode='min'
    )
    #添加到 callbacks 列表中
    callbacks.append(early_stopping)
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **kwargs)
    return trainer



def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    ## modifications by me
    if config.dataset._name_.endswith('_lm') and config.train.log_tensors_freq > 0:
        model.log_tensors_freq = config.train.log_tensors_freq
        model.log_tensors_for_validation = True if config.train.log_tensors_freq > 0 else False
        if not os.path.exists('train'): os.mkdir('train')
        if not os.path.exists('val'):  os.mkdir('val')

    if 'rand_init' in config.train:
        if config.train.rand_init:
            apply_rand_init(model)

    if config.train.pretrained_model_path is not None:
        model = load_pretrained_model(config, model)
        
    #print(model)

    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    if config.train.ckpt is not None:  # for resuming training
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)
    if config.train.test:
        trainer.test(model)


def load_pretrained_model(config, model):
    ckpt_path = config.train.pretrained_model_path
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    try:
        model.load_state_dict(state_dict, strict=False)

    except Exception as e:
        print(f"\n\nLoading full state dict failed with error:\n'{e}'\n->Trying to load without decoder weights.")
        print(f"Loading from {ckpt_path}")

        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('decoder')}  # throw away decoder

        # in case linear encoder, mask embedding is the 2nd dim
        if config['encoder'] == 'linear':
            # check for additional mask dim
            if state_dict['encoder.0.weight'].shape[1] == model.encoder.state_dict()['0.weight'].shape[1] + 1:
                state_dict['encoder.0.weight'] = state_dict['encoder.0.weight'][:, :-1]  # last dim is mask embedding

            if state_dict['encoder.0.weight'].ndim == 1:
                state_dict['encoder.0.weight'] = state_dict['encoder.0.weight'].unsqueeze(-1)

        model.load_state_dict(state_dict, strict=False)  # strict False allows not loading decoder

    # validate state dict was loaded correctly
    for k, v in state_dict.items():
        if 'norm' in k or k.startswith('decoder'): continue  # ignore layer norms
        if v.dtype == torch.int64: continue  # ignore int64
        disc = torch.norm(v - model.state_dict()[k]).item()
        if disc > 1e-4:
            raise ValueError(f"State dict not loaded correctly at key {k}")
    print('State dict loaded successfully!\n\n')
    return model


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    
    config = utils.train.process_config(config)
    
    utils.train.print_config(config, resolve=True)

    train(config)


if __name__ == "__main__":
    main()
