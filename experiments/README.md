# 测量脚本

这几个脚本是本次提交配套的**测量**工具：它们的输出被逐字写进
`agent/conversation.json` 的 `experiments` 字段，用来支撑材料里"量过之后
才决定"的那些取舍。它们不是运行时的一部分，评测镜像不会拷贝这个目录
（`Dockerfile` 只拷贝 `minivectordb/`、`tools/`、`protocol_check/`、
`mysqltest/`）。

在仓库根目录运行：

```bash
python experiments/exp_distance_equivalence.py    # 距离计算：全量 1000 条查询逐条比对
python experiments/exp_ann_nprobe.py              # IVF：nprobe 与召回率、耗时的关系
python experiments/exp_checkpoint_policy.py       # 检查点：写入、崩溃恢复、磁盘占用
python experiments/exp_ann_real_engine.py         # 近似模式：真正建了索引时的召回率
```

| 脚本 | 回答的问题 | 结论 |
| --- | --- | --- |
| `exp_distance_equivalence.py` | 把距离计算换成更快的写法，返回的 TopK 会不会变？ | 冻结负载的**全部 1,000 条**查询上两种候选写法与现实现返回的主键集合完全一致；K 边界的最小距离间距 1.086e-05 比最大算术偏差 8.49e-08 大三个数量级。但两种候选都更慢，所以保留原实现。 |
| `exp_ann_nprobe.py` | 索引的 nprobe 买到了多少召回？ | nprobe=16（当前设置）Recall@10 只有 0.544，nprobe=64 才有 1.0，而后者比同一批数据的精确扫描慢约 49 倍。 |
| `exp_checkpoint_policy.py` | 每次提交写检查点，代价多大、换来什么？ | 写入慢 11.8%，换来崩溃恢复从 18.896 s 降到 0.017 s、数据目录从 13.15 MiB 降到 5.04 MiB。恢复上限 120 s，所以检查点必须每次提交都写。 |
| `exp_ann_real_engine.py` | 真正建了索引之后，近似模式的召回率是多少？ | 见 `agent/conversation.json` 的 `e36`；此前所有"ann 召回 1.0"都来自 50000 行以下的精确回退。 |

这些脚本只读仓库里的 `minivectordb/`，不修改任何实现；除了
`exp_checkpoint_policy.py` 会往 `E:\MiniVectorDB\runs\checkpoint_exp` 写临时
数据目录之外，不会在别处留下文件。
