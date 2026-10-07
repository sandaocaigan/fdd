# FedDistill：联邦蒸馏中的攻击与主动 Probe 检测

本项目用于研究联邦蒸馏（FedDistill）场景下的客户端投毒、拜占庭攻击和防御方法。在原始 FedDistill 代码基础上，项目加入了主动 Probe 审计、SPC/Soft-SPC 指标、基于核心分布的 Soft-JS 过滤，以及面向 logit 上传的鲁棒聚合方法。

主要流程是：客户端使用私有数据本地训练，在公共数据上生成 logits 并上传；恶意客户端可以修改上传结果；服务器使用 Probe 审计客户端行为，必要时过滤异常上传，最后聚合 logits 并进行服务器蒸馏。

## 项目流程

```text
main.py

  -> ScratchRunner.run()

  -> 初始化随机种子、模型、数据和优化器

  -> 创建攻击与防御对象

  -> train_federated()

       -> 服务器模型下发

       -> 客户端私有数据本地训练

       -> 客户端在公共数据上生成 logits

       -> 攻击修改恶意客户端上传

       -> Probe 审计和 Soft-SPC 计算

       -> 可选的 5sigma Soft-JS 过滤

       -> Mean / TriMean / Krum / AFA 聚合

       -> 服务器使用聚合 logits 进行蒸馏

       -> 验证、测试并记录日志
```

当前默认策略是 `FedDistill`。客户端上传的是公共数据上的模型 logits，不是客户端模型参数。服务器根据聚合后的 logits 生成软标签，并在公共数据上训练服务器模型。

## 目录说明

- `main.py`：实验入口和全部默认超参数。
  
- `runners/BaseRunner.py`：初始化模型、客户端、服务器、私有数据、公共数据和 Probe。
  
- `runners/ScratchRunner.py`：执行通信轮、本地训练、攻击、Probe、过滤、防御和服务器蒸馏。
  
- `strategies.py`：定义 `FedAVG` 和 `FedDistill` 的流程。
  
- `actors.py`：定义 `Client`、`Server` 和公共 Actor 功能。
  
- `byzantine/attacks.py`：定义 logits 上传攻击和部分原始攻击。
  
- `byzantine/backdoor_attacks.py`：定义 `BadNets`、`WaNet` 和 `LabelConsistent`。
  
- `byzantine/defences.py`：定义 Mean、TriMean、Krum、AFA 等聚合防御。
  
- `probes/probe_dataset.py`：生成和包装 Probe 图片。
  
- `probes/probe_builder.py`：实现尺度、噪声和混合 Probe。
  
- `probes/probe_auditor.py`：计算 SPC、Soft-TV、Soft-JS、consensus 等审计指标。
  
- `models/`：定义不同数据集使用的模型结构。
  
- `config.py`：私有数据集、类别数、归一化和训练变换。
  
- `public_config.py`：公共数据集及其变换配置。
  
- `utilities.py`：数据集包装、聚合和训练辅助函数。
  
- `byzantine/cpa_info/`：CPA 攻击使用的预计算协方差文件。
  

## 环境

已验证的主要环境：

- Linux
  
- Python 3.10
  
- PyTorch 2.4.1
  
- torchvision 0.19.1
  
- CUDA 12.1 版 PyTorch
  
- `torchmetrics`、`wandb`、`numpy`、`scipy`、`matplotlib`、`Pillow`、`tqdm`
  

基础安装脚本：

```bash
bash install_feddistill_env.sh
```

扩散净化功能还需要对应的扩散模型依赖和有效检查点。普通攻击、Probe 和 logit 聚合实验不需要启用扩散净化器。

## 基本运行方式

在项目根目录运行：

```bash
conda activate /home/user/lichengyuan/.conda_envs/feddistill310

export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



mkdir -p logs



python -u main.py \

  --debug \

  --run_id 10001 \

  --seed 20260805 \

  --strategy FedDistill \

  --attack NoAttack \

  --defence NoDefence \

  --n_byzantine_clients 0 \

  --n_clients 30 \

  --n_total_local_epochs 70 \

  --n_communications 70 \

  --n_server_epochs_per_round 1 \

  --distill_temperature 1 \

  --use_probe false \

  --use_amp false \

  2>&1 | tee logs/10001_clear.log
```

使用 `NoAttack` 时应将 `n_byzantine_clients` 设置为 `0`。

## 攻击

常用攻击包括：

- `Gaussian`：向恶意客户端上传的 logits 添加高斯扰动。
  
- `Disruption`：在公共样本维度上打乱每个类别的 logits。
  
- `Shuffling`：改变最高类别和次高类别的 logit 排序。
  
- `LabelFlip`：恶意客户端在本地训练时翻转标签。
  
- `ManipulatingKD`：优化恶意软标签，同时满足攻击目标和隐蔽性约束。
  
- `CPA`、`CELMAX`：原项目中的预测攻击。
  
- `BadNets`、`WaNet`、`LabelConsistent`：输入级后门攻击。
  

## 防御

普通平均：

```bash
--defence NoDefence
```

TriMean：

```bash
--defence TriMean --trimean_trim_ratio 0.2
```

它对每个 logit 坐标去掉最低和最高的一部分客户端，再对剩余值求平均。

Krum 或 Multi-Krum：

```bash
--defence Krum --krum_f 3 --krum_num_selected 1
```

`krum_num_selected=1` 是 Krum；设置为大于 1 时是 Multi-Krum。Krum 需要预先给出恶意客户端数量上界。启用 Probe 过滤时，`krum_residual_f` 表示过滤后仍可能残留的恶意客户端上界。

AFA：

```bash
--defence AFA
```

AFA 根据客户端 logits 与聚合结果的相似度和历史可靠性进行筛选或加权。当前实现是适配 FedDistill logits 上传的版本，不应直接视为原始参数联邦学习中的 AFA 原代码复现。

## Probe、SPC 和 Soft-JS 过滤

启用 Probe：

```bash
--use_probe true \

--probe_mode replace \

--probe_type scale \

--probe_base_count 20 \

--probe_scales 1.0,1.2,1.5,2.0 \

--probe_seed 0 \

--probe_start_round 15
```

主要参数：

- `probe_base_count`：基础图片数量。
  
- `probe_scales`：每张基础图片生成的尺度列表。
  
- `probe_mode=replace`：用 Probe 变体替换公共数据中的对应位置。
  
- `probe_mode=audit_only`：保留普通公共图片，仅进行审计；单张图片独立成组时不能计算有效的组内尺度一致性 SPC。
  
- `probe_start_round`：开始审计的通信轮。
  

仅审计、不过滤：

```bash
--use_probe true --probe_core_sigma_audit true --probe_core_sigma_filter false
```

启用 Soft-JS 核心分布过滤：

```bash
--use_probe true --probe_core_sigma_filter true --probe_core_sigma_threshold 5.0
```

当前过滤逻辑以每轮 Soft-JS 数值最密集的约一半作为核心分布，将高于核心中心 `5sigma` 的客户端排除出本轮聚合。被排除的客户端仍会训练和生成上传结果，只是不参加当前轮聚合。

## 结果与日志

终端实时输出可以保存为日志：

```bash
2>&1 | tee logs/run_name.log
```

重点查看：

- `Evaluating server on public test data`：服务器测试结果。
  
- `[Probe Audit]`：每轮 Probe 指标。
  
- `[Probe Core-Sigma Filter]`：被过滤的客户端和保留数量。
  
- `[TriMean]`、`[Krum]`、`[AFA]`：防御聚合信息。
  

比较实验时应固定数据集、模型、客户端数量、恶意客户端数量、随机种子、攻击开始轮数、通信轮数和本地训练轮数。同时报告服务器准确率、被过滤客户端数量和良性误报数量。

## 当前限制

1. 客户端私有训练数据采用固定随机种子的均匀划分，默认不是 Non-IID 划分。
  
2. Probe 过滤阈值需要在多个随机种子和攻击比例下验证，不能只根据单次实验下结论。
  
3. Krum 和 AFA 是针对当前 logits 上传流程的适配实现，与参数上传联邦学习中的原始算法存在表示空间差异。
  

## 参考文献

本项目的基础代码来自：

```bibtex
@inproceedings{roux2025on,

  title={On the Byzantine-Resilience of Distillation-Based Federated Learning},

  author={Christophe Roux and Max Zimmer and Sebastian Pokutta},

  booktitle={The Thirteenth International Conference on Learning Representations},

  year={2025},

  url={https://openreview.net/forum?id=of6EuHT7de}

}
```