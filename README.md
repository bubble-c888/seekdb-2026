# MiniVectorDB

MiniVectorDB 是 2026 年 OceanBase 数据库大赛 **赛题一：完成一个轻量向量数据库** 的实现：
一个从零写起的单机、单进程、单客户端的向量数据库，支持题目要求的 SQL 子集、
持久化存储、事务与崩溃恢复，并提供精确（exact）与近似（ann）两种向量 TopK 检索。

存储引擎、SQL 解析与执行、向量索引、事务与恢复全部自行实现，没有使用 SQLite、
RocksDB 等现成存储或数据库引擎，也没有使用 Faiss / VSAG / HNSWlib 等向量索引库。
唯一的第三方依赖是 **numpy**（float32 运算与向量化扫描）。

---

## 1. 快速开始

### 1.1 构建镜像

```bash
docker build -t minivectordb .
```

镜像基于 `python:3.13-slim-bookworm`，numpy 的版本在构建阶段安装并锁定
（`numpy==2.5.3`），运行阶段不需要联网。

### 1.2 前台启动

```bash
docker run -i --rm -v /host/data:/data minivectordb
```

* 容器以前台方式运行数据库程序，**不监听任何端口**；
* 测试驱动通过进程管道连接容器的 stdin / stdout，因此使用 `-i`，
  **不要**使用 `-d` 或 `-t`；
* 数据库文件全部写入 `/data`（容器内的目录，可用 `-v` 挂载到宿主机的独立数据目录）；
* stderr 单独收集日志，不会混入 stdout 的评测结果。

本地直接运行（不需要容器）：

```bash
python -m minivectordb --data-dir ./data
```

### 1.3 一次完整的手工演练

```bash
$ printf '%s\n' \
  "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), content VARCHAR(16), embedding VECTOR(2));" \
  "INSERT INTO docs VALUES (3, '乙', '三', '[0, 2]'), (1, '甲', '一', '[0, 0]'), (2, '甲', '二', '[1, 0]');" \
  "SELECT id, content FROM docs WHERE category = '甲' ORDER BY id;" \
  "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 2;" \
  | python -m minivectordb --data-dir ./data
```

stdout（每条语句一个 JSON 对象，一行一个）：

```json
{}
{"affected_rows": 3}
{"columns": ["id", "content"], "rows": [[1, "一"], [2, "二"]]}
{"columns": ["id", "distance"], "rows": [[1, 0.0], [2, 1.0]]}
```

结束方式：关闭 stdin（EOF）、发送 `SIGTERM`/`SIGINT`，程序会回滚未提交事务、
落盘并以 0 退出。

---

## 2. 运行协议

程序从 stdin 逐行读取 SQL，每条语句在 stdout 上输出 **恰好一个 JSON 对象**，
输出后立即 flush，不缓冲。

### 2.1 语句边界

* 一行一条语句；语句可以跨多行（程序会等到语句完整后再执行）；
* 行尾的分号可有可无；同一行中出现多条以分号分隔的语句也支持；
* `--`、`#` 起始的行注释与 `/* ... */` 块注释会被忽略。

### 2.2 响应格式

| 语句 | 响应对象 |
| --- | --- |
| `CREATE TABLE` / `CREATE VECTOR INDEX` / `BEGIN` / `COMMIT` / `ROLLBACK` | `{}` |
| `INSERT` / `UPDATE` / `DELETE` | `{"affected_rows": 3}` |
| `SELECT` | `{"columns": ["id", "distance"], "rows": [[1, 0.0]]}` |
| 任何错误 | `{"error": "...", "code": 1064, "sqlstate": "42000"}` |

* `columns` 是列名数组，含别名，顺序与 `SELECT` 一致；
* `rows` 中的值：BIGINT → 整数，VARCHAR → 字符串，VECTOR → 浮点数组，
  距离 → 浮点数；
* 错误对象一定带 `error`（消息）、`code`（MySQL 兼容的错误码）与 `sqlstate`；
  语句失败不会让进程退出。

### 2.3 就绪标识

程序启动并完成恢复后会等待 `MINIVECTORDB_READY_DELAY_MS`（默认 400 毫秒）
等待第一条请求：

* 如果驱动先发请求，stdout 上 **不会出现** 任何多余的行；
* 如果驱动等待程序先说话，程序会在 stdout 上输出一行 `READY`。

也可以用环境变量或命令行显式指定：

```bash
MINIVECTORDB_READY_MARKER=stdout|stderr|both|none   # 就绪标识写到哪里
MINIVECTORDB_READY_TEXT=READY                        # 就绪标识的内容
MINIVECTORDB_READY_DELAY_MS=400                      # 等待首条请求的时间
python -m minivectordb --ready-marker stdout
```

### 2.4 精确 / 近似模式

默认是 `exact`。以下三种方式都可以切换到 `ann`：

```bash
python -m minivectordb --mode ann            # 命令行参数
MINIVECTORDB_MODE=ann python -m minivectordb  # 环境变量
```

```sql
-- 也可以在会话里切换
SET search_mode = 'ann';
```

`ann` 只影响向量 TopK 查询；**即使建了近似索引，`exact` 模式仍然返回精确结果**。

### 2.5 JSON 请求封装（可选）

如果驱动更习惯发送 JSON 请求，程序也接受这种形式：

```json
{"sql": "SELECT id FROM docs;", "mode": "ann", "id": 7}
```

响应仍然是普通的结果对象；如果请求里带 `id`，响应会原样带回。

---

## 3. SQL 支持范围

### 3.1 支持的语句

```sql
CREATE TABLE <name> ( <col> <type> [NOT NULL] [PRIMARY KEY] [, ...] );
CREATE TABLE <name> ( <col> <type> [, ...], PRIMARY KEY ( <col> ) );
INSERT INTO <name> [ ( <col> [, ...] ) ] VALUES ( <value> [, ...] ) [, ...];
SELECT <select_list> FROM <name> [WHERE <expr>] [ORDER BY <item> [ASC|DESC] [, ...]] [LIMIT <n>];
UPDATE <name> SET <col> = <value> [, ...] [WHERE <expr>];
DELETE FROM <name> [WHERE <expr>];
BEGIN;  COMMIT;  ROLLBACK;
CREATE VECTOR INDEX <name> ON <table>(<vector column>);
```

* 类型：`BIGINT`、`VARCHAR(n)`、`VECTOR(n)`；每表一个 BIGINT 主键，最多一个向量列；
* 比较运算符：`=`、`<`、`<=`、`>`、`>=`，条件之间用 `AND` 连接；
* 投影：普通列、`SELECT *`、`L2_DISTANCE(<vector column>, '<vector literal>') [AS alias]`；
* 排序：`ORDER BY <列名或别名>`，并列时按主键升序；`DESC` 只影响主排序键。

### 3.2 明确的语义约定

* **主键**唯一，重复插入报 `1062 duplicate entry`；
* **VARCHAR** 长度按 Unicode 字符数计算，字符串比较按 Unicode 码点顺序
  （例如 `'乙' < '甲'`），不做语言相关的排序；
* **向量**是固定维度的 float32 数组：维度不符、NaN、Infinity、
  超出 float32 表示范围的值都会被拒绝（`11000`）；
* **L2_DISTANCE** 是欧氏距离而不是距离平方：`[3,4]` 到 `[0,0]` 的距离是 `5`；
* 带过滤的 TopK **先过滤再取 TopK**，不会先在全表取 K 条再过滤；
* 结果不足 K 条时返回全部符合条件的记录；距离相同时按主键升序；
* `LIMIT 0` 返回空结果，`LIMIT` 大于记录数时返回全部记录；
* 不支持的语句（`NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、
  `OR`、`IN`、`BETWEEN`、`LIKE`、`DISTINCT`、嵌套事务、主键更新等）
  一律返回错误，绝不静默执行为别的语义。

### 3.3 错误码

| 场景 | code |
| --- | --- |
| 语法错误 | 1064 |
| 不支持的语法 / 功能 | 1235 |
| 表不存在 | 1146 |
| 列不存在 | 1054 |
| 主键重复 | 1062 |
| 值超出范围（向量维度、NaN、溢出） | 11000 |
| 字符串超长 | 1406 |
| 嵌套 `BEGIN` | 1225 |

---

## 4. 存储与事务设计

### 4.1 数据布局

每张表一个目录，列存布局：

```
/data/
  catalog.json                 # 表结构与检查点
  wal/wal-000001.log           # 预写日志（每个已提交事务一条记录）
  t/<table>/
    col_<name>.dat             # 每列一个定长文件，第 i 个槽位 = 第 i 行
    alive.bits                 # 存活行位图
    dict_<name>.dat/.idx       # VARCHAR 列的字典编码（码 -> 字符串）
    idx_<index>.cent           # 向量索引的聚类中心
    idx_<index>_list_<k>.dat   # 向量索引的倒排表
```

* 定长列（BIGINT / VECTOR）可以直接用 numpy 切片读取，
  因此过滤与距离计算都是向量化的；
* VARCHAR 使用字典编码：列文件里存的是整数码，
  码按 Unicode 码点顺序排序后即满足题目的排序语义；
* 删除的行只是把存活位图里的位置清掉，槽位在下一个事务里可以复用；
* 主键索引是内存中的开放寻址哈希表，启动时从主键列重建，
  因此索引永远不会和数据文件不一致。

### 4.2 事务与提交协议

事务把要做的修改先缓存在内存里（**不会**在提交前碰数据文件）：

1. `BEGIN` 开启事务，之后的语句都写进事务缓冲；
2. `COMMIT` 时：
   1. 把这次事务的**物理字节补丁**和新的元数据快照追加到预写日志，`fsync`；
      ——**这一步是提交点**；
   2. 把字节补丁写进数据文件并 `fsync`；
3. `ROLLBACK` 直接丢弃事务缓冲，数据文件从未被改动。

因为补丁是物理写入，重放是幂等的：

* 崩溃发生在第 1 步之前 → 事务完全没有生效；
* 崩溃发生在第 1 步之后 → 启动时重放日志，事务完整生效。

未显式 `BEGIN` 时，每条修改语句按自动提交处理，语义与显式事务一致。

### 4.3 崩溃恢复

启动时：

1. 读 `catalog.json`（上次检查点的目录）；
2. 从记录的日志位置开始重放所有已提交事务的字节补丁；
3. 重建内存中的主键索引和向量索引；
4. 写新的检查点。

日志末尾被截断的半条记录会被丢弃并截断掉 —— 这正对应
"已发出提交请求但未收到成功响应的事务，允许完整生效或完整不生效"。

### 4.4 向量索引

`CREATE VECTOR INDEX idx ON t(col)` 建立一个 IVF 风格的近似索引：

* 用 k-means（k=64）把向量聚成若干簇，簇心存在 `idx_<index>.cent`；
* 每个簇一个倒排文件，查询时按簇心距离由近及远取候选；
* 新增、更新、删除的行都会同步反映到倒排表里
  （更新会按新向量重新归簇，删除靠存活位图过滤）；
* `ann` 模式按候选集计算精确距离后取 TopK，带过滤时按需继续扩展候选集，
  保证过滤条件不会被放宽；
* `exact` 模式完全扫描向量列，忽略近似索引，结果永远精确。

---

## 5. 性能

`tools/benchmark.py` 按题目描述的负载做基准：10,000 条 128 维记录，
每批 100 条写入并提交，然后 1,000 次 TopK 查询（K=10，一半带等值过滤）。

```bash
python tools/benchmark.py --rows 10000 --queries 1000
```

实测（普通笔记本，Windows / Python 3.13 + numpy，两次独立运行结果一致）：

| 指标 | exact | ann |
| --- | --- | --- |
| 批量写入 10,000 行（128 维，每批 100 条提交） | 4.46 s | 4.36 s |
| 1,000 次 TopK 查询（K=10，一半带等值过滤） | 3.02 ms/次 | 3.08 ms/次 |
| 有过滤 / 无过滤平均 Recall@10 | 1.0 / 1.0 | 1.0 / 1.0 |
| 峰值常驻内存 | 59.4 MiB | 59.0 MiB |

> 峰值内存由操作系统接口（Linux `getrusage` / Windows `GetProcessMemoryInfo`）采集，
> 其中包含了基准脚本自身为计算真值而保留的 10,000×128 float32 矩阵；
> 数据库进程本身按 8192 行分块流式读取列文件，不会把全表读进内存
> （`tests.test_engine.ResourceTest` 用 4 KiB 的缓存预算验证过这一点）。

性能优化要点：

* 列存 + numpy 向量化：过滤、距离计算与 TopK 选取都在 C 层完成；
* 每块先算排序键（`np.lexsort`），只把块内最优的 K 行送进全局堆，
  避免逐行 Python 开销；
* 只有最终返回的 K 行才构造结果集；
* 存活位图常驻内存，避免每行一次磁盘读；
* 提交时按文件合并字节补丁，每个文件只打开一次；
* 热列带预算的内存缓存（`MINIVECTORDB_CACHE_BYTES`，默认 32 MiB），
  超预算时自动退回按块读取。

---

## 6. 本地自测

```bash
# 公开用例（模拟组织方的测试驱动）
python tools/run_case.py                 # 全部用例
python tools/run_case.py --case sql_crud # 单个用例

# 功能测试（事务、持久化、恢复、索引维护、错误处理）
python -m unittest tests.test_engine -v

# 解析器测试
python -m unittest tests.test_parser -v

# 基准
python tools/benchmark.py --mode exact
python tools/benchmark.py --mode ann
```

`tools/run_case.py` 为每个用例新建独立的数据目录，逐行发送 `.test` 文件里的 SQL，
把 stdout 的每个 JSON 对象与 `.result` 中同位置的断言比较
（浮点按 1e-5 的绝对与相对误差比较，未标记 `unordered` 的结果按顺序比较，
标记的按多重集合比较）。

> 这是我自己按公开用例语义重写的简化驱动，不是组织方的完整驱动；
> 公开包 README 也写明“进程通信的封装由配套测试驱动确定”。

公开用例覆盖建表、多行插入、查询、更新、删除、数值与字符串比较、`AND`、
升降序、并列主键排序、`LIMIT`、空表、二维向量与欧氏距离。
它们只是公开样例，不等于正式准入通过。

---

## 7. 已知限制

* 单机、单进程、单客户端；同一时刻最多一个显式事务，不支持并发控制；
* 不支持 `NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、嵌套事务与主键更新；
* 未提交事务的改动保存在内存里，超大事务会占用较多内存；
* 向量索引是 IVF 风格的近似索引，`exact` 模式始终精确，`ann` 模式的召回率
  取决于聚类数与探针数；
* 主键索引常驻内存（每行约 16 字节），超大规模数据下需要进一步做磁盘化。

---

## 8. 提交材料

* 源码：`minivectordb/`（引擎、解析器、协议层）与 `tools/`、`tests/`；
* 构建材料：`Dockerfile`（从源码构建，可直接启动评测容器）；
* Agent prompt 材料：`agent/conversation.json`（同一个提交的 commit 中）；
  仓库根目录的 `conversation.json` 是同一份内容的副本，两处都放置以适配不同的读取位置；
* 运行说明与设计记录：本文件。

提交时的分支为 `2026-competition-round2`，提测需要提供仓库 HTTPS 地址、
分支名与 40 位完整 commit SHA。
