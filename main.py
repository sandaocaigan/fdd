# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         main.py
# Description:  Starts up a run
# ===========================================================================

import argparse
import getpass
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
import platform

warnings.filterwarnings('ignore')
debug = "--debug" in sys.argv

defaults = dict(
    # System
    run_id=1,

    # Setup
    dataset='cifar10',
    arch='ResNet18',
    batch_size=128,

    # Efficiency
    use_amp=None,  # Defaults to True 加速器

    # Optimizer
    optimizer='SGD',
    momentum=0.9,
    weight_decay=0.0001,
    client_early_stopping=None,
    server_early_stopping=None,

    # Strategy
    strategy='FedDistill',
    n_clients=10,

    # FL settings
    n_total_local_epochs=50,  # Number of local epochs per client (in total!)
    n_communications=50,  # Number of communications between server and clients
    n_server_epochs_per_round=1,  # How many epochs should the server model train per round 	每轮服务器用聚合后的公共预测训练几轮。
    server_lr='(Linear, 0.1)',
    client_lr='(Linear, 0.1)',
    restart_client_lr=None,  # If True, restart the learning rate after each communication是否每轮通信后重置客户端学习率。
    reinit_server=None,  # Reinitialize server model, optimizer, scheduler after up-communication.是否每轮重新初始化服务器模型/优化器。
    warm_restarts=None, # 学习率热重启。
    public_ds=None,  # If None, use the default public dataset, otherwise specify the name 公共数据集选择。
    public_ds_fraction=None,  # If None, use the default public dataset, otherwise take fraction of the train set for the public ds 公共数据集使用比例。

    # Attacks and defences
    defence=None,
    attack='CPA',
    n_byzantine_clients=2, # 恶意客户端数量
    filter_threshold=None, # PredictionFilter 防御的过滤阈值。
    filter_quantile=None, # 按分位数过滤异常客户端/预测的参数。
    memory_method=None,  # 'expweights', 'cumsum' or 'quantile' 带记忆的防御策略，比如 expweights。None 表示不用历史记忆。
    expweights=None, # 指数权重相关参数。通常配合 memory_method='expweights' 使用。
    exp_stepsize=None, # 指数权重更新步长。
    hips=None, # 特殊采样或高影响样本比例参数。
    sample_attack_frac=None,  # percentage of datapoints on which to choose hips byzantine pred 恶意客户端攻击多少比例的公共样本。

    # Probe audit settings
    attack_start_round=15,
    use_probe=True,
    probe_mode='replace',
    probe_type='scale', # 探针生成方式。
    probe_base_count=20,  # 选多少张原始公共图片作为探针基图。
    probe_scales='1.0,1.2,1.5,2.0', # 每张基图生成几个缩放版本
    probe_seed=0,  
    probe_start_round=15, # 探针审计开始轮数。
    probe_w_entropy=1.0, # 风险分数中 entropy（信息熵）项的权重。恶意客户端通常熵更低。
    probe_w_confidence=1.0, # 风险分数中 confidence（置信度）项的权重。恶意客户端通常置信度更高。
    probe_w_spc=1.0, # 	风险分数中 SPC（缩放预测一致性）项的权重。恶意客户端通常一致性更高。
    probe_log_each_round=True,
    probe_clip_min=None,
    probe_clip_max=None,
)



def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ['true', '1', 'yes', 'y']:
        return True
    if value in ['false', '0', 'no', 'n']:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def _coerce_cli_value(key, value, default):
    """Convert command-line strings to the type expected by the config."""
    if value in [None, 'None', 'none', 'null', 'NULL']:
        return None

    int_keys = {'run_id', 'batch_size', 'n_clients', 'n_total_local_epochs',
                'n_communications', 'n_server_epochs_per_round',
                'n_byzantine_clients', 'attack_start_round', 'probe_base_count',
                'probe_seed', 'probe_start_round'}
    float_keys = {'momentum', 'weight_decay', 'public_ds_fraction',
                  'filter_threshold', 'filter_quantile', 'expweights',
                  'exp_stepsize', 'hips', 'sample_attack_frac',
                  'probe_w_entropy', 'probe_w_confidence', 'probe_w_spc',
                  'probe_clip_min', 'probe_clip_max'}
    bool_keys = {'use_amp', 'client_early_stopping', 'server_early_stopping',
                 'restart_client_lr', 'reinit_server', 'warm_restarts',
                 'use_probe', 'probe_log_each_round'}

    if key in int_keys:
        return int(value)
    if key in float_keys:
        return float(value)
    if key in bool_keys:
        return _parse_bool(value)
    return value

# 读取终端命令行参数，并覆盖 defaults 里的默认配置。
def apply_cli_overrides(defaults):
    """Allow commands like --attack CELMAX to override the default config."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    for key in defaults.keys():
        parser.add_argument(f'--{key}', default=None)

    args, _ = parser.parse_known_args()
    for key, default in list(defaults.items()):
        value = getattr(args, key, None)
        if value is not None:
            defaults[key] = _coerce_cli_value(key, value, default)
    return defaults


defaults = apply_cli_overrides(defaults)
if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)
n_gpus = torch.cuda.device_count()
if n_gpus > 0:
    config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))


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
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:    
    sys.stdout.write(f"Using config: {config}.\n")
    runner = ScratchRunner(config=config, tmp_dir=tmp_dir, debug=debug)

    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)


