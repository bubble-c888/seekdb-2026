# 本地补充用例

公开题包只给了 20 个用例，正式用例共 29 个，剩下的 9 个不公开输入和预期。
这里按赛题原文的「必须支持」与「必须报错」两条清单，自己补了一批用例，
覆盖公开用例没有走到的写法与组合：

| 用例 | 覆盖的要求 |
| --- | --- |
| `sql_null` | NULL 必须报错 |
| `sql_join` | JOIN 必须报错 |
| `sql_aggregate` | 聚合必须报错 |
| `sql_subquery` | 子查询必须报错 |
| `sql_foreign_key` | 外键必须报错 |
| `sql_nested_transaction` | 嵌套事务必须报错 |
| `sql_update_primary_key` | 主键更新必须报错 |
| `sql_limit_forms` | `LIMIT n`、`LIMIT offset, n`、`LIMIT n OFFSET m` |
| `sql_select_star` | `SELECT *` |
| `sql_insert_column_list` | 带列名的 INSERT |
| `sql_table_primary_key` | 表级 `PRIMARY KEY (col)` |
| `sql_string_order` | 字符串按码点序排序 |
| `sql_string_compare` | 字符串的大小比较 |
| `sql_case_insensitive` | 标识符不区分大小写 |
| `sql_vector_projection` | 向量列的投影格式 |
| `transaction_visibility` | 事务内能看到自己插入和删除的行 |
| `transaction_vector_visibility` | 事务内改过的向量按新值算距离 |
| `transaction_commit_visibility` | 提交后修改仍然可见 |

**预期结果全部是手写的**，不是从实现上抄的——否则这个用例只能证明
「实现和自己一致」，证明不了语义对不对。

用题包自带的检查器跑：

```bash
python3 protocol_check/run.py --image <镜像> \
  --package tests/local_cases --output protocol-results/local
```

或者在本机（Windows）上用仓库里的包装器：

```bash
python tools/run_checker.py --package tests/local_cases --output runs/local
```
