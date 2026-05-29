# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ATC GPT pretraining entrypoint with an experimental DeepSpeed runtime wrapper.

This file intentionally keeps the integration local to this entrypoint. The
normal ATC-Megatron path is preserved unless ``--deepspeed`` is passed.
"""

import os
import time
from functools import partial

_PROGRAM_START_TIME = time.time()

import torch

import deepspeed
import pretrain_gpt as atc_gpt
from gpt_builders import gpt_builder
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import inprocess_restart, pretrain, set_startup_timestamps
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider


def _parser_has_option(parser, option):
    return any(option in action.option_strings for action in parser._actions)


def _add_deepspeed_wrapper_args(parser):
    if atc_gpt.has_nvidia_modelopt:
        parser = atc_gpt.add_modelopt_args(parser)

    parser = deepspeed.add_config_arguments(parser)

    # DeepSpeed launchers commonly pass one spelling or the other. ATC reads
    # LOCAL_RANK from the environment, but keeping the arg parseable makes this
    # entrypoint usable with the DeepSpeed launcher too.
    if not (
        _parser_has_option(parser, "--local_rank")
        or _parser_has_option(parser, "--local-rank")
    ):
        parser.add_argument(
            "--local_rank",
            "--local-rank",
            type=int,
            default=int(os.environ.get("LOCAL_RANK", "0")),
            help="Local rank passed by distributed launchers.",
        )

    group = parser.add_argument_group("ATC DeepSpeed wrapper")
    group.add_argument(
        "--deepspeed-wrapper-mode",
        choices=["model", "optimizer"],
        default=os.environ.get("ATC_DEEPSPEED_WRAPPER_MODE", "model"),
        help=(
            "DeepSpeed integration mode. 'model' wraps the ATC model in a "
            "DeepSpeedEngine while preserving the ATC optimizer. 'optimizer' "
            "also tries to route optimizer.step through DeepSpeed and is a "
            "diagnostic path for compatibility testing."
        ),
    )
    group.add_argument(
        "--deepspeed-zero-stage",
        type=int,
        choices=[0, 1, 2, 3],
        default=int(os.environ.get("ATC_DEEPSPEED_ZERO_STAGE", "0")),
        help="Zero stage used only when --deepspeed_config is not provided.",
    )
    return parser


class _DeepSpeedModelAdapter(torch.nn.Module):
    """Expose the ATC model-chunk surface while forwarding through DeepSpeed."""

    def __init__(self, engine, owns_optimizer=False):
        super().__init__()
        self.engine = engine
        self.module = engine.module
        self.owns_optimizer = owns_optimizer
        self._atc_deepspeed_engine = engine

    @property
    def force_all_reduce(self):
        return getattr(self.module, "force_all_reduce", False)

    @force_all_reduce.setter
    def force_all_reduce(self, value):
        setattr(self.module, "force_all_reduce", value)

    def forward(self, *args, **kwargs):
        return self.engine(*args, **kwargs)

    def zero_grad_buffer(self):
        if hasattr(self.module, "zero_grad_buffer"):
            return self.module.zero_grad_buffer()
        return None

    def finish_grad_sync(self, *args, **kwargs):
        if not self.owns_optimizer and hasattr(self.module, "finish_grad_sync"):
            return self.module.finish_grad_sync(*args, **kwargs)
        return None

    def start_grad_sync(self, *args, **kwargs):
        if not self.owns_optimizer and hasattr(self.module, "start_grad_sync"):
            return self.module.start_grad_sync(*args, **kwargs)
        return None

    def scale_gradients(self, scaling_factor):
        if not self.owns_optimizer and hasattr(self.module, "scale_gradients"):
            return self.module.scale_gradients(scaling_factor)
        for param in self.module.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                main_grad.mul_(scaling_factor)
            elif param.grad is not None:
                param.grad.mul_(scaling_factor)
        return None

    def broadcast_params(self):
        if hasattr(self.module, "broadcast_params"):
            return self.module.broadcast_params()
        return None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)


class _DeepSpeedOptimizerAdapter:
    """Expose the optimizer surface Megatron's loop expects for DeepSpeed."""

    def __init__(self, engine):
        self.engine = engine
        self.optimizer = engine.optimizer
        self.is_stub_optimizer = False

    @property
    def param_groups(self):
        if self.optimizer is not None and hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups
        inner_optimizer = getattr(self.optimizer, "optimizer", None)
        if inner_optimizer is not None and hasattr(inner_optimizer, "param_groups"):
            return inner_optimizer.param_groups
        return []

    def zero_grad(self, *args, **kwargs):
        self.engine.zero_grad()
        for param in self.engine.module.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                main_grad.zero_()

    def scale_loss(self, loss):
        # DeepSpeed applies its own loss scaling inside engine.backward().
        return loss

    def step(self, *args, **kwargs):
        for param in self.engine.module.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None and tuple(main_grad.shape) == tuple(param.shape):
                param.grad = main_grad
        self.engine.step(*args, **kwargs)
        update_successful = bool(getattr(self.engine, "_step_applied", True))
        grad_norm = getattr(self.engine, "_global_grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach()
        return update_successful, grad_norm, None

    def get_loss_scale(self):
        if hasattr(self.engine, "loss_scale"):
            value = self.engine.loss_scale()
        elif hasattr(self.engine, "optimizer") and hasattr(self.engine.optimizer, "loss_scale"):
            value = self.engine.optimizer.loss_scale
        else:
            value = 1.0
        return torch.tensor([float(value)], device=torch.cuda.current_device())

    def reload_model_params(self):
        return None

    def state_dict(self):
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            return self.optimizer.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self.optimizer is not None and hasattr(self.optimizer, "load_state_dict"):
            return self.optimizer.load_state_dict(state_dict)
        return None

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


class _DeepSpeedMPUAdapter:
    """Compatibility shim for DeepSpeed's older Megatron mpu interface."""

    def get_model_parallel_group(self):
        return mpu.get_model_parallel_group()

    def get_model_parallel_world_size(self):
        return (
            mpu.get_tensor_model_parallel_world_size()
            * mpu.get_pipeline_model_parallel_world_size()
        )

    def get_model_parallel_rank(self):
        return torch.distributed.get_rank(group=mpu.get_model_parallel_group())

    def get_tensor_model_parallel_group(self):
        return mpu.get_tensor_model_parallel_group()

    def get_tensor_model_parallel_world_size(self):
        return mpu.get_tensor_model_parallel_world_size()

    def get_tensor_model_parallel_rank(self):
        return mpu.get_tensor_model_parallel_rank()

    def get_tensor_model_parallel_src_rank(self):
        return mpu.get_tensor_model_parallel_src_rank()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def get_data_parallel_world_size(self):
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def get_data_parallel_group_ranks(self):
        if not torch.distributed.is_initialized():
            return [0]
        group = mpu.get_data_parallel_group()
        world_size = torch.distributed.get_world_size()
        ranks = []
        for rank in range(world_size):
            try:
                torch.distributed.get_group_rank(group, rank)
            except ValueError:
                continue
            ranks.append(rank)
        return ranks

    def get_sequence_parallel_world_size(self):
        return 1

    def get_sequence_parallel_rank(self):
        return 0

    def get_sequence_parallel_group(self):
        return mpu.get_tensor_model_parallel_group()

    def get_sequence_data_parallel_world_size(self):
        return mpu.get_data_parallel_world_size()

    def get_sequence_data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def get_sequence_data_parallel_group(self):
        return mpu.get_data_parallel_group()


def _default_deepspeed_config(args, include_optimizer=False):
    grad_accum = get_num_microbatches()
    data_parallel_size = mpu.get_data_parallel_world_size()
    config = {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": args.micro_batch_size * grad_accum * data_parallel_size,
        "zero_optimization": {"stage": args.deepspeed_zero_stage},
    }
    if include_optimizer:
        if args.optimizer != "adam":
            raise RuntimeError(
                "The default DeepSpeed optimizer config currently supports "
                "ATC --optimizer adam only. Pass --deepspeed_config for other "
                "DeepSpeed optimizer setups."
            )
        config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [args.adam_beta1, args.adam_beta2],
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            },
        }
        if args.clip_grad > 0:
            config["gradient_clipping"] = args.clip_grad
    if args.fp16:
        config["fp16"] = {"enabled": True}
    if args.bf16:
        config["bf16"] = {"enabled": True}
    return config


def _get_model_config(model):
    config = getattr(model, "config", None)
    if config is not None:
        return config
    module = getattr(model, "module", None)
    if module is not None:
        return getattr(module, "config", None)
    return None


def _configure_deepspeed_ownership(model, engine, owns_optimizer):
    config = _get_model_config(model)
    if config is None:
        raise RuntimeError(
            "DeepSpeed optimizer mode requires access to the model transformer "
            "config so the training schedule can route backward through the "
            "DeepSpeed engine."
        )

    config.deepspeed_engine = engine
    config.deepspeed_owns_backward = owns_optimizer
    config.deepspeed_owns_optimizer = owns_optimizer


def _install_main_grad_buffers_for_deepspeed(model):
    """Provide ATC fused linear backward kernels a gradient accumulation target."""

    for param in model.parameters():
        if param.requires_grad and getattr(param, "main_grad", None) is None:
            param.main_grad = torch.zeros_like(
                param.data,
                dtype=param.dtype,
                memory_format=torch.preserve_format,
            )


def _install_deepspeed_setup_wrapper():
    import megatron.training.training as training

    original_setup = training.setup_model_and_optimizer

    def setup_model_and_optimizer_with_deepspeed(*args, **kwargs):
        megatron_args = training.get_args()
        if not getattr(megatron_args, "deepspeed", False):
            return original_setup(*args, **kwargs)

        owns_optimizer = megatron_args.deepspeed_wrapper_mode == "optimizer"
        if owns_optimizer:
            original_skip_train = megatron_args.skip_train
            original_no_load_optim = megatron_args.no_load_optim
            megatron_args.skip_train = True
            megatron_args.no_load_optim = True
            try:
                model, _optimizer, _opt_param_scheduler = original_setup(*args, **kwargs)
            finally:
                megatron_args.skip_train = original_skip_train
                megatron_args.no_load_optim = original_no_load_optim
            optimizer, opt_param_scheduler = None, None
        else:
            model, optimizer, opt_param_scheduler = original_setup(*args, **kwargs)

        if len(model) != 1:
            raise RuntimeError(
                "The DeepSpeed wrapper currently supports a single ATC model chunk. "
                "Pipeline or virtual-pipeline DeepSpeed wrapping requires additional "
                "schedule/checkpointing support."
            )

        ds_config = (
            None
            if megatron_args.deepspeed_config
            else _default_deepspeed_config(megatron_args, include_optimizer=owns_optimizer)
        )
        ds_optimizer = None

        engine, _engine_optimizer, _loader, _engine_scheduler = deepspeed.initialize(
            args=megatron_args,
            model=model[0],
            optimizer=ds_optimizer,
            model_parameters=model[0].parameters(),
            lr_scheduler=None,
            mpu=_DeepSpeedMPUAdapter(),
            dist_init_required=False,
            config=ds_config,
        )
        if owns_optimizer and megatron_args.gradient_accumulation_fusion:
            _install_main_grad_buffers_for_deepspeed(model[0])
        _configure_deepspeed_ownership(model[0], engine, owns_optimizer)
        model = [_DeepSpeedModelAdapter(engine, owns_optimizer=owns_optimizer)]

        if owns_optimizer:
            optimizer = _DeepSpeedOptimizerAdapter(engine)
            if not megatron_args.skip_train:
                opt_param_scheduler = training.get_optimizer_param_scheduler(optimizer)

        return model, optimizer, opt_param_scheduler

    training.setup_model_and_optimizer = setup_model_and_optimizer_with_deepspeed


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    setattr(atc_gpt.train_valid_test_datasets_provider, "is_distributed", True)
    _install_deepspeed_setup_wrapper()

    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    parsed_args = parse_and_validate_args(
        extra_args_provider=_add_deepspeed_wrapper_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    full_config = pretrain_cfg_container_from_args(parsed_args)
    wrapped_pretrain(
        full_config,
        atc_gpt.train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        atc_gpt.forward_step,
        store=store,
        get_embedding_ranks=atc_gpt.get_embedding_ranks,
    )
