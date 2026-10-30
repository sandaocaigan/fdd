# FedDistill：联邦蒸馏中的攻击与主动 Probe 检测

本项目用于研究联邦蒸馏（FedDistill）场景下的客户端投毒、拜占庭攻击和防御方法。在原始 FedDistill 代码基础上，项目加入了主动 Probe 审计、SPC/Soft-SPC 指标、基于核心分布的 Soft-JS 过滤，以及面向 logit 上传的鲁棒聚合方法。

主要流程是：客户端使用私有数据本地训练，在公共数据上生成 logits 并上传；恶意客户端可以修改上传结果；服务器使用 Probe 审计客户端行为，必要时过滤异常上传，最后聚合 logits 并进行服务器蒸馏。

## PCFDLA 上传攻击（2026-10-26）

新增独立的 `--attack PCFDLA --pcfdla_peak 5`，保留原有 Shuffling。对每个恶意客户端自身正常 logits 排序，将第二高类别设为 `+S`，其余类别全部设为 `-S`。并列分数按类别编号稳定排序。只修改上传，不改变客户端模型或私有训练；普通公共图片和探针统一处理，不读取探针分组或公共集真实标签。

这是将 PCFDLA 的上传规则接入本项目公共图片 FedDistill 协议，并非复刻论文的整个 FedCache 系统。论文链接：https://doi.org/10.1007/s10791-025-09742-8 。`S=1/5/10` 是本项目试验档位，不是已核实的论文原始参数；第二高预测也不保证是真实错误类别。

`ssh/run_pcfdla_spc_matrix.sh` 恢复历史表格的主要条件：ResNet18 随机初始化、CIFAR-10 私有集 47500 张、验证集 2500 张、STL-10 无标签公共集 100000 张、旧版 FedDistill 权重下发、客户端不做公共集 KD；SGD、batch 128、温度 1、双方 Linear 0.1 学习率、每轮私有训练/服务器蒸馏均 1 epoch。正式实验为 30 客户端、9 恶意、70 轮、第 15 轮攻击、SPC 阈值 5。所有组使用相同探针配置并开启审计，只切换过滤，避免把图片变化与过滤效果混淆。

```bash
# 只看计划，不启动训练；正式运行需确保 Run ID 尚未使用。
bash ssh/run_pcfdla_spc_matrix.sh 11001 smoke plan
# 两轮、四客户端、一恶意，配对检查攻击及过滤链路，不代表最终效果。
bash ssh/run_pcfdla_spc_matrix.sh 11001 smoke run
# 八项正式实验：无攻击对照两项，以及 S=1/5/10 各自开关过滤。
bash ssh/run_pcfdla_spc_matrix.sh 11011 full plan
bash ssh/run_pcfdla_spc_matrix.sh 11011 full run
```

脚本自动检查数据完整性、运行 PCFDLA 单元测试，拒绝覆盖已有实验日志；运行明细和汇总保存在 `logs`。完整 STL-10 每轮需要大量推理，不能用短测试耗时直接估计正式实验总时间。先确认无防御攻击相对干净基线造成损失，再评估 SPC，不能预设它必然检测成功。

## PCFDLA S=10 与传统防御对比（2026-10-28）

新增 `ssh/run_pcfdla_traditional_matrix.sh`，只增加实验调度，不修改攻击、训练流程或现有聚合实现。六项实验依次为 TriMean、Multi-Krum、AFA 各自的无攻击和 PCFDLA S=10 配对组，默认编号 11021–11026。已有 11011、11012、11017、11018 分别作为 Mean/SPC 的无攻击/攻击参考，不重复训练。

共同条件与上一轮正式实验一致：随机初始化 ResNet18、FedDistill、CIFAR-10 私有 47500 张与 STL-10 公共 100000 张、30 客户端、70 轮、seed=20260805、第 15 轮启用攻击与探针；攻击组 9 恶意，无攻击组 0 恶意。全部新组保留相同探针图片配置与 SPC 审计，但关闭 SPC 过滤，检验的是传统聚合本身。此处的传统防御都作用于上传 logits，并非原始模型参数聚合系统。

防御参数预先固定：TriMean 两端各裁剪 30%（30 客户端时各 9 个；旧表为 20%，因此需要使用本次对应无攻击基线）；Multi-Krum 无攻击/攻击均 `f=9`、选择 3 个，禁用回退；AFA 为 xi=2、delta=0.5、Beta 先验 3/3、封禁阈值 0.95。优先比较各自干净基线的准确率损失，同时记录 Krum 恶意选中、AFA 恶意保留情况，不能仅用攻击组绝对准确率排名。

### 已完成结果（单随机种子）

六组实验均正常完成，运行时间约 6.2–6.6 小时/组。下表汇总传统防御对比，以及此前同配置的 Mean、SPC 参考组；准确率为 CIFAR-10 测试集结果。变化量按各防御自身的无攻击 server 基线计算。

| 防御 | 无攻击 Run | 攻击 Run | Server 无攻击 | Server 攻击后 | Server 变化 | Ensemble 无攻击 | Ensemble 攻击后 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean（无防御） | 11011 | 11017 | 71.79% | 29.94% | -41.85 pp | 71.70% | 29.83% |
| SPC | 11012 | 11018 | 71.14% | 72.06% | +0.92 pp | 71.57% | 72.30% |
| TriMean | 11021 | 11022 | 68.16% | 63.37% | -4.79 pp | 68.80% | 64.82% |
| Multi-Krum | 11023 | 11024 | 61.65% | 60.65% | -1.00 pp | 62.33% | 61.05% |
| AFA | 11025 | 11026 | 68.86% | 44.17% | -24.69 pp | 69.17% | 45.16% |

审计显示：Multi-Krum 在 56 个攻击轮次中共选出 168 个客户端，恶意客户端入选 0 次；TriMean 将 PCFDLA 相对无防御造成的 server 准确率损失从 41.85 pp 降至 4.79 pp。AFA 在攻击轮次保留恶意上传 330/504（65.48%），44/56 轮至少保留一个恶意上传，与其较大的准确率下降一致。SPC 在该攻击设置下过滤 504/504 个恶意上传，同时误过滤 14/1176 个正常上传（1.19%）。

这些是单个 seed 的结果，不代表多次运行的均值或统计显著性；SPC 准确率略高于清洁基线不能解释为攻击提升了模型。传统防御实验是在本项目的 logits 聚合路径上实现，不能据此声称复现了参数更新版 FedAvg 防御。TriMean 使用 ratio=0.3，而历史表格使用 0.2；本次使用自身配对的清洁基线进行比较。

```bash
bash ssh/run_pcfdla_traditional_matrix.sh 11021 clear_trimean plan
bash ssh/run_pcfdla_traditional_matrix.sh 11021 clear_trimean run
# 中断后用全新 BASE_ID，从未完成项重头训练；这不是 checkpoint 恢复。
# bash ssh/run_pcfdla_traditional_matrix.sh NEW_ID pcfdla_s10_multikrum plan
```

启动前运行 `test_pcfdla*.py`（攻击测试与三种防御兼容性测试），校验两种数据集，检查日志冲突和训练进程，并与现有 PCFDLA 调度共用锁。结果写入 `logs/*_pcfdla_traditional.summary.txt`；某项失败即停止，不覆盖已完成日志。正式参数未按本次攻击结果重新调优。

## 本版本更新（2026-10-10）

新增显式的 P-L Mean 代理模式，可用相同的 ManipulatingKD 上传分别测试 Krum、TriMean、AFA。攻击端用最近几轮广播结果构造 Mean 代理估计，服务器仍执行所选的真实防御；代理估计不等于对非线性聚合的精确反推。默认的 `strict_mean` 模式继续只支持均值聚合，已有实验命令的行为不变。

在版本 `e92bc17` 已完成 ManipulatingKD P-L 与 SPC 兼容的基础上，本版本新增了可选的服务器预训练初始化，并保留原有随机初始化方式。

- 新增 `--server_init_ckpt`：不传参数时保持随机初始化；传入检查点时加载预训练服务器权重。
- 预训练工具只使用固定清单中的 5000 张带标签 CIFAR-10 公共样本，不接触剩余 42500 张客户端私有样本。
- 固定的 2500 张验证样本用于选择最佳预训练轮次，CIFAR-10 测试集只在训练完成后评估一次。
- 检查点保存数据集、模型结构、类别数、公共样本数和清单 SHA256；联邦训练启动时逐项验证并严格加载权重。
- 禁止同时使用 `--server_init_ckpt` 和 `--reinit_server true`，避免通信轮重新初始化并丢弃预训练权重。
- 新增服务器检查点和公共数据隔离测试；默认无检查点的旧实验命令不受影响。

上一版本已完成以下 ManipulatingKD P-L 与 SPC 改动：

- P-L 历史记录不再使用过滤后列表的位置推断客户端身份，而是使用稳定的 `client_id`。
- SPC 过滤后，攻击历史会保存实际参与聚合的客户端数量、恶意上传和保留客户端 ID。
- 修正 P-L 对上一轮诚实客户端输出的重建公式，聚合分母改为本轮实际保留的客户端数量。
- 支持部分或全部恶意客户端被 SPC 删除，也能安全处理未保留诚实客户端的特殊情况。
- 解除 `ManipulatingKD + SPC` 的旧兼容性限制；Krum、TriMean、AFA 等非线性聚合须显式开启 Mean 代理估计。
- 新增 `tests/test_manipulatingkd_history.py`，覆盖无过滤、客户端位置变化、部分恶意客户端删除、全部恶意客户端删除等场景。
- 新增 `ssh/run_manipkd_spc_pair.sh` 和 `ssh/run_clear_spc_control.sh`，用于复现 P-L + SPC 及无攻击 + SPC 对照实验。

固定 CIFAR-10 公共集、`seed=20260805` 下的单次实验结果如下：

| 实验 | P-L 攻击 | SPC | Server 准确率 | Ensemble 准确率 |
|---|---:|---:|---:|---:|
| 清洁基线 | 否 | 否 | 37.03% | 36.54% |
| P-L 攻击 | 是 | 否 | 26.60% | 29.31% |
| 清洁 + SPC | 否 | 是 | 32.47% | 31.86% |
| P-L + SPC | 是 | 是 | 32.66% | 31.59% |

当前结果说明 P-L 攻击已经产生明显效果，SPC 基本消除了攻击带来的额外准确率损失。但无攻击条件下 SPC 的诚实上传误报率仍较高，且固定 5000 张公共数据的清洁基线偏低。表中结果全部来自随机初始化，不能与新增的预训练实验直接混合比较；加入预训练后应重新运行完整的清洁、攻击和防御对照组。

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

## 可选服务器预训练

该功能引入了一个明确的新实验假设：固定的 5000 张公共 CIFAR-10 图片具有真实标签，并允许服务器在联邦训练前使用这些标签进行监督预训练。论文或实验表格必须注明“带标签公共数据预训练”，不能把它与随机初始化结果视为同一设置。

先在项目根目录训练服务器初始化检查点：

```bash
/home/user/lichengyuan/.conda_envs/feddistill310/bin/python \
  tools/train_clean_teacher.py \
  --dataset cifar10 \
  --arch ResNet18 \
  --output checkpoints/cifar10_resnet18_public5000_seed20260805.pt \
  --public_split_manifest splits/cifar10_public_500x10_seed20260805.json \
  --public_samples_per_class 500 \
  --public_val_samples_per_class 250 \
  --public_split_seed 20260805 \
  --seed 20260805 \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --device cuda:0 \
  --use_amp
```

联邦训练时在原命令中加入：

```bash
--server_init_ckpt checkpoints/cifar10_resnet18_public5000_seed20260805.pt \
--reinit_server false
```

启动日志应出现 `Server initialization: loaded public-only checkpoint` 和检查点元数据。第一轮本地训练前，`FedDistill.before_local_training()` 会把该服务器权重下发给全部客户端。

不使用预训练时无需添加参数，或显式使用：

```bash
--server_init_ckpt none
```

此时日志显示 `Server initialization: random weights (no checkpoint)`，行为与旧版本一致。

## 攻击

常用攻击包括：

- `Gaussian`：向恶意客户端上传的 logits 添加高斯扰动。
  
- `Disruption`：在公共样本维度上打乱每个类别的 logits。
  
- `Shuffling`：改变最高类别和次高类别的 logit 排序。
  
- `LabelFlip`：恶意客户端在本地训练时翻转标签。
  
- `ManipulatingKD`：优化恶意软标签，同时满足攻击目标和隐蔽性约束。

- `CPA`、`CELMAX`：原项目中的预测攻击。

- `BadNets`、`WaNet`、`LabelConsistent`：输入级后门攻击。

论文式 P-L 测试 Krum、TriMean 或 AFA 时，沿用同一组数据、检查点和攻击参数，在原有命令中设置：

```bash
--attack ManipulatingKD \
--manipkd_formulation paper \
--manipkd_knowledge partial_local \
--manipkd_pl_estimator mean_surrogate \
--manipkd_history_rounds 5 \
--defence Krum
```

将最后一行换为 `--defence TriMean --trimean_trim_ratio 0.3` 或 `--defence AFA` 即可运行其他防御。论文在 30 个客户端、9 个恶意客户端的设置下，TriMean 每端去掉 9 个值，因此比例为 `0.3`。每种防御还需要同配置的 `NoAttack` 对照组；Krum 对照组应与攻击组使用相同的 `--krum_f 9`。默认 `strict_mean` 仍用于均值聚合的精确历史重建。日志中的 `[ManipulatingKD P-L proxy diagnostic]` 报告真实防御聚合与上传均值的差异；`[Krum selection audit]` 和 `[AFA retention audit]` 报告恶意上传的选中或保留情况。


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
