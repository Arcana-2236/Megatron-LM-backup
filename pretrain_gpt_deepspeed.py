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

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.module = engine.module
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
    """Small adapter for the optional DeepSpeed optimizer-step experiment."""

    def __init__(self, engine, fallback_optimizer):
        self.engine = engine
        self.fallback_optimizer = fallback_optimizer
        self.optimizer = engine.optimizer
        self.is_stub_optimizer = False

    @property
    def param_groups(self):
        if self.optimizer is not None and hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups
        return self.fallback_optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        self.engine.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.engine.step(*args, **kwargs)
        return True, None, None

    def get_loss_scale(self):
        if hasattr(self.engine, "loss_scale"):
            value = self.engine.loss_scale()
        elif hasattr(self.engine, "optimizer") and hasattr(self.engine.optimizer, "loss_scale"):
            value = self.engine.optimizer.loss_scale
        else:
            value = 1.0
        return torch.tensor([float(value)], device=torch.cuda.current_device())

    def reload_model_params(self):
        if hasattr(self.fallback_optimizer, "reload_model_params"):
            return self.fallback_optimizer.reload_model_params()
        return None

    def state_dict(self):
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            return self.optimizer.state_dict()
        return self.fallback_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if self.optimizer is not None and hasattr(self.optimizer, "load_state_dict"):
            return self.optimizer.load_state_dict(state_dict)
        return self.fallback_optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self.fallback_optimizer, name)


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


def _default_deepspeed_config(args):
    grad_accum = get_num_microbatches()
    data_parallel_size = mpu.get_data_parallel_world_size()
    config = {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": args.micro_batch_size * grad_accum * data_parallel_size,
        "zero_optimization": {"stage": args.deepspeed_zero_stage},
    }
    if args.fp16:
        config["fp16"] = {"enabled": True}
    if args.bf16:
        config["bf16"] = {"enabled": True}
    return config


def _extract_torch_optimizer(optimizer):
    if optimizer is None:
        return None
    torch_optimizer = getattr(optimizer, "optimizer", None)
    if isinstance(torch_optimizer, torch.optim.Optimizer):
        return torch_optimizer
    return None


def _install_deepspeed_setup_wrapper():
    import megatron.training.training as training

    original_setup = training.setup_model_and_optimizer

    def setup_model_and_optimizer_with_deepspeed(*args, **kwargs):
        model, optimizer, opt_param_scheduler = original_setup(*args, **kwargs)
        megatron_args = training.get_args()
        if not getattr(megatron_args, "deepspeed", False):
            return model, optimizer, opt_param_scheduler

        if len(model) != 1:
            raise RuntimeError(
                "The one-file DeepSpeed wrapper only supports a single ATC model chunk. "
                "Pipeline or virtual-pipeline DeepSpeed wrapping requires changes in "
                "Megatron training schedules/checkpointing outside pretrain_gpt_deepspeed.py."
            )

        ds_config = megatron_args.deepspeed_config or _default_deepspeed_config(megatron_args)
        ds_optimizer = None
        if megatron_args.deepspeed_wrapper_mode == "optimizer":
            ds_optimizer = _extract_torch_optimizer(optimizer)
            if ds_optimizer is None:
                raise RuntimeError(
                    "--deepspeed-wrapper-mode=optimizer requires the ATC optimizer to expose "
                    "a plain torch.optim.Optimizer. The current optimizer path does not, so "
                    "routing optimizer.step through DeepSpeed needs optimizer/framework changes "
                    "outside pretrain_gpt_deepspeed.py."
                )

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
        model = [_DeepSpeedModelAdapter(engine)]

        if megatron_args.deepspeed_wrapper_mode == "optimizer":
            optimizer = _DeepSpeedOptimizerAdapter(engine, optimizer)

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
