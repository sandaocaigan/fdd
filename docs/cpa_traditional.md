# CPA 与传统防御的对比

本次只增加实验脚本、集成测试与说明，不修改 CPA、SPC、聚合或训练实现。
沿用 12101–12104 的 CPA plain、logits 上传、CIFAR-10 / ResNet18 / STL-10 配置。

## 正式实验

| 编号 | 攻击 | 防御 |
| --- | --- | --- |
| 12301 | 无攻击 | TriMean |
| 12302 | CPA | TriMean |
| 12303 | 无攻击 | Multi-Krum |
| 12304 | CPA | Multi-Krum |
| 12305 | 无攻击 | AFA |
| 12306 | CPA | AFA |

所有组均独立从头训练 70 轮，不接着 11301 或 12101 checkpoint。
seed=20260805；30 客户端；攻击组 9 恶意，清洁组 0 恶意；第 15 轮开始攻击。
传统聚合从第 1 轮工作，因此前 14 轮是没有攻击的各自防御训练，
不同防御不共享完全相同的第 14 轮模型；AFA 历史也从第 1 轮积累。
同一防御的清洁/攻击组采用相同聚合超参数，每个运行重新初始化 AFA 状态。

防御参数预先固定，清洁与攻击组保持一致：

- TriMean：ratio=0.3，每个坐标两端各去掉 9 个客户端，平均中间 12 个。
- Multi-Krum：krum_f=9，选取 3 个客户端，nearest=19，不允许回退到 Mean。
  清洁组也使用 f=9，而不是根据清洁组真实恶意数改成 0。
- AFA：xi=2、delta_xi=0.5、先验 alpha=beta=3、block_threshold=0.95。

与 CPA Mean/SPC 四组相同：SGD、momentum=.9、weight_decay=.0001、batch=128、
server/client Linear .1 LR、每轮本地训练和服务器蒸馏各 1 epoch、T=1、
不预训练、不用 purifier/AMP/客户端公共 KD；CIFAR-10 私有 47500、验证 2500，
完整 STL-10 无标签公共集 100000。
CPA hips=false、sample_attack_frac=1、logit epsilon=1e-12；攻击者能观察诚实预测。

所有组保留相同探针输入和审计（20 基图 x 4 强度），从第 15 轮启用，
但 SPC 过滤始终关闭。探针的 core_flag 只用于审计，不表示该客户端被传统防御删除。
应读取 Krum selection audit、AFA retention audit 判断实际参与情况；
TriMean 是逐坐标裁剪，不能解释为整客户端删除。

已有参考结果不重跑：12101 无攻击 Mean=.7179；12102 无攻击 SPC=.7114；
12103 CPA Mean=.0442；12104 CPA SPC=.5306。
后续比较每种防御自身清洁到攻击的下降，以及攻击下最终准确率。
这批与历史参考都是单种子，不能当多种子均值；这些传统防御是 logits 空间适配。

## 上传（本地 PowerShell）

服务器已运行过 CPA 正式实验，因此只需上传新脚本和测试：

```powershell
Set-Location -LiteralPath 'C:\Users\ZhuanZ（无密码）\Desktop\fdd-main'
scp -P 50925 .\ssh\run_cpa_traditional_matrix.sh root@10.8.135.1:/home/user/lichengyuan/fdd-main/ssh/
scp -P 50925 .\tests\test_cpa_defences.py root@10.8.135.1:/home/user/lichengyuan/fdd-main/tests/
scp -P 50925 .\docs\cpa_traditional.md root@10.8.135.1:/home/user/lichengyuan/fdd-main/docs/
```

## 启动（服务器）

```bash
cd /home/user/lichengyuan/fdd-main
bash ssh/run_cpa_traditional_matrix.sh 12301 full plan
```

可以先做 6 客户端、1 恶意、2 轮的六组短测（攻击从第 1 轮开始）：

```bash
bash ssh/run_cpa_traditional_matrix.sh 12291 smoke run
```

短测使用 6 个客户端以满足 Multi-Krum n>2f+2 和选 3 个的要求，
正式实验仍为 30/9。短测也读取完整公共集，只验证链路而非收敛。

```bash
mkdir -p logs
nohup bash ssh/run_cpa_traditional_matrix.sh 12301 full run > logs/cpa_12301.launcher.log 2>&1 < /dev/null &
tail -f logs/cpa_12301.launcher.log
```

日志逐组写入 logs/12301_clear_trimean.log 等文件。
汇总为 logs/12301_12306_cpa_traditional_full.summary.txt。
脚本检查 CPA 回归/集成测试、数据完整性、已有日志编号、同类调度锁及训练进程；
任一组失败即停止，不覆盖旧日志。
默认解释器为 /home/user/lichengyuan/.conda_envs/feddistill310/bin/python，
可用 PYTHON 环境变量覆盖。本次准备工作没有上传或启动远程训练。
