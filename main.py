# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         main.py
# Description:  Starts up a run
# ===========================================================================

import argparse
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import torch
import wandb

from runners.ScratchRunner import ScratchRunner
from utilities import Utilities as Utils

warnings.filterwarnings("ignore")
debug = "--debug" in sys.argv

defaults = dict(
    # System
    run_id=1,
    seed=None,

    # Setup
    dataset="cifar10",
    arch="ResNet18",
    batch_size=128,

    # Efficiency
    use_amp=None,

    # Optimizer
    optimizer="SGD",
    momentum=0.9,
    weight_decay=0.0001,
    client_early_stopping=None,
    server_early_stopping=None,

    # Strategy
    strategy="FedDistill",
    n_clients=10,

    # FL settings
    n_total_local_epochs=50,
    n_communications=50,
    n_server_epochs_per_round=1,
    server_lr="(Linear, 0.1)",
    client_lr="(Linear, 0.1)",
    restart_client_lr=None,
    reinit_server=None,
    warm_restarts=None,
    public_ds=None,
    public_ds_fraction=None,
    distill_temperature=1.0,

    # Attacks and defences
    defence=None,
    attack="CPA",
    n_byzantine_clients=2,
    filter_threshold=None,
    filter_quantile=None,
    memory_method=None,
    expweights=None,
    exp_stepsize=None,
    hips=None,
    sample_attack_frac=None,
    logit_gaussian_scale=1.0,

    # ManipulatingKD attack settings. "paper" implements the F-L/P-L objectives.
    manipkd_formulation="paper",
    manipkd_steps=30,
    manipkd_lagrange_lr=1.0,
    manipkd_history_rounds=5,
    manipkd_lr=0.05,
    manipkd_target="least_likely",
    manipkd_knowledge="full_local",
    manipkd_surrogate="mean",
    manipkd_constraints="l2,range,moments,entropy,outlier",
    manipkd_init="honest_mean",
    manipkd_lambda_l2=1.0,
    manipkd_lambda_range=1.0,
    manipkd_lambda_moments=0.25,
    manipkd_lambda_entropy=0.25,
    manipkd_lambda_outlier=1.0,
    manipkd_q_low=0.05,
    manipkd_q_high=0.95,
    manipkd_q_radius=0.95,
    manipkd_q_outlier=0.95,
    manipkd_radius_multiplier=1.25,
    manipkd_single_client_radius=0.5,
    manipkd_noise_std=0.01,
    manipkd_project=True,
    manipkd_log_every=0,
    manipkd_trim_count=None,
    manipkd_geomedian_iters=5,
    manipkd_krum_temperature=1.0,
    # Input-level backdoor attack settings
    backdoor_target_class=0,
    backdoor_source_class=None,
    backdoor_poison_rate=0.25,
    backdoor_upload_rate=1.0,
    backdoor_trigger_size=4,
    backdoor_trigger_value=1.0,
    backdoor_trigger_position="bottom_right",
    wanet_grid_strength=0.15,

    # Backdoor attack success rate evaluation
    evaluate_backdoor_asr=True,
    asr_eval_data="test",
    asr_eval_frequency=5,
    asr_batch_limit=None,

    # Probe audit settings
    attack_start_round=15,
    use_probe=True,
    probe_mode="replace",
    probe_type="scale",
    probe_score="hybrid",
    probe_base_count=20,
    probe_scales="1.0,1.2,1.5,2.0",
    probe_seed=0,
    probe_start_round=15,
    probe_target_class=None,
    probe_source_class=None,
    probe_blend_alpha=0.5,
    probe_blend_mode="random",
    probe_w_entropy=1.0,
    probe_w_confidence=1.0,
    probe_w_spc=1.0,
    probe_w_target_bias=0.0,
    probe_w_consensus=1.0,
    probe_w_label=1.0,
    probe_log_each_round=True,
    probe_clip_min=None,
    probe_clip_max=None,
    # Core-sigma filtering uses only within-client Soft-JS probe consistency.
    probe_core_sigma_audit=False,
    probe_core_sigma_filter=False,
    probe_core_sigma_threshold=5.0,

    # V1 raw Unet1D diffusion purifier settings
    use_diffusion_purifier=False,
    diffusion_ckpt=None,
    diffusion_target="yellow",
    diffusion_input_type="logits",
    diffusion_sampler="ddpm",
    diffusion_steps=10,
    diffusion_batch_size=512,
    diffusion_red_threshold=0.88,
    diffusion_yellow_low=0.65,
    diffusion_yellow_high=0.88,
    diffusion_entropy_calibration=False,
    diffusion_entropy_source="green_median",
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ["true", "1", "yes", "y"]:
        return True
    if value in ["false", "0", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def _coerce_cli_value(key, value, default):
    """Convert command-line strings to the type expected by the config."""
    if value in [None, "None", "none", "null", "NULL"]:
        return None

    int_keys = {
        "run_id", "seed", "batch_size", "n_clients", "n_total_local_epochs",
        "n_communications", "n_server_epochs_per_round",
        "n_byzantine_clients", "attack_start_round", "probe_base_count",
        "probe_seed", "probe_start_round", "probe_target_class",
        "probe_source_class", "diffusion_steps", "asr_eval_frequency",
        "asr_batch_limit",
        "diffusion_batch_size", "backdoor_target_class", "backdoor_source_class",
        "backdoor_trigger_size", "manipkd_steps", "manipkd_log_every", "manipkd_history_rounds",
        "manipkd_trim_count", "manipkd_geomedian_iters",
    }
    float_keys = {
        "momentum", "weight_decay", "public_ds_fraction",
        "filter_threshold", "filter_quantile", "expweights",
        "exp_stepsize", "sample_attack_frac", "distill_temperature",
        "backdoor_poison_rate", "backdoor_upload_rate", "backdoor_trigger_value",
        "logit_gaussian_scale",
        "wanet_grid_strength",
        "probe_blend_alpha", "probe_w_entropy", "probe_w_confidence", "probe_w_spc",
        "probe_w_target_bias", "probe_w_consensus", "probe_w_label",
        "probe_clip_min", "probe_clip_max", "diffusion_red_threshold",
        "diffusion_yellow_low", "diffusion_yellow_high",
        "probe_core_sigma_threshold",
        "manipkd_lr", "manipkd_lambda_l2", "manipkd_lambda_range",
        "manipkd_lambda_moments", "manipkd_lambda_entropy", "manipkd_lambda_outlier",
        "manipkd_q_low", "manipkd_q_high", "manipkd_q_radius", "manipkd_q_outlier",
        "manipkd_radius_multiplier", "manipkd_single_client_radius", "manipkd_noise_std", "manipkd_krum_temperature",
        "manipkd_lagrange_lr",
    }
    bool_keys = {
        "use_amp", "client_early_stopping", "server_early_stopping",
        "restart_client_lr", "reinit_server", "warm_restarts",
        "hips", "evaluate_backdoor_asr", "use_probe", "probe_log_each_round", "use_diffusion_purifier",
        "diffusion_entropy_calibration", "manipkd_project", "probe_core_sigma_audit",
        "probe_core_sigma_filter",
    }

    if key in int_keys:
        return int(value)
    if key in float_keys:
        return float(value)
    if key in bool_keys:
        return _parse_bool(value)
    return value


def apply_cli_overrides(defaults):
    """Allow commands like --attack CELMAX to override the default config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    for key in defaults.keys():
        parser.add_argument(f"--{key}", default=None)

    args, _ = parser.parse_known_args()
    for key, default in list(defaults.items()):
        value = getattr(args, key, None)
        if value is not None:
            defaults[key] = _coerce_cli_value(key, value, default)
    return defaults


defaults = apply_cli_overrides(defaults)
if not debug:
    defaults = Utils.fill_dict_with_none(defaults)

defaults["computer"] = socket.gethostname()

wandb.init(
    config=defaults,
    project="test-000",
    entity=None,
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)
n_gpus = torch.cuda.device_count()
if n_gpus > 0:
    config.update(dict(device="cuda:0"))
else:
    config.update(dict(device="cpu"))


@contextmanager
def tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write("Failed to clean up temp dir {}".format(path))


with tempdir() as tmp_dir:
    sys.stdout.write(f"Using config: {config}.\n")
    runner = ScratchRunner(config=config, tmp_dir=tmp_dir, debug=debug)

    runner.run()

    wandb_dir_path = wandb.run.dir
    wandb.join()

    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
