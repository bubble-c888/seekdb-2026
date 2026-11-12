# MiniVectorDB

MiniVectorDB 是 2026 年 OceanBase 数据库大赛 **赛题一：完成一个轻量向量数据库** 的实现：
一个从零写起的单机、单进程、单客户端的向量数据库，支持题目要求的 SQL 子集、
持久化存储、事务与崩溃恢复，并提供精确（exact）与近似（ann）两种向量 TopK 检索。

存储引擎、SQL 解析与执行、向量索引、事务与恢复全部自行实现，没有使用 SQLite、
RocksDB 等现成存储或数据库引擎，也没有使用 Faiss / VSAG / HNSWlib 等向量索引库。
唯一的第三方依赖是 **numpy**（float32 运算与向量化扫描）。

通信协议按题包公布的 [STDIO_PROTOCOL.md](STDIO_PROTOCOL.md) 实现（JSON Lines），
自测直接使用题包附带的 `protocol_check/`。

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
* stderr 单独收集日志，不会混入 stdout 的协议报文。

评测时的容器参数（只读根文件系统、无网络、2 GiB 内存、128 个进程）都能满足：

```bash
docker run --rm -i --network=none --cpus=2 --memory=2g --memory-swap=2g \
  --pids-limit=128 --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --mount type=volume,src=minivectordb-data,dst=/data minivectordb
```

本地直接运行（不需要容器）：

```bash
python -m minivectordb --data-dir ./data
```

### 1.3 一次完整的手工演练

请求按 JSON Lines 发送，每行一个 JSON 对象：

```bash
$ printf '%s\n' \
  '{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), content VARCHAR(16), embedding VECTOR(2));","search_mode":"exact"}' \
  '{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (3, '"'"'乙'"'"', '"'"'三'"'"', '"'"'[0, 2]'"'"'), (1, '"'"'甲'"'"', '"'"'一'"'"', '"'"'[0, 0]'"'"'), (2, '"'"'甲'"'"', '"'"'二'"'"', '"'"'[1, 0]'"'"');","search_mode":"exact"}' \
  '{"id":3,"op":"execute","sql":"SELECT id, content FROM docs WHERE category = '"'"'甲'"'"' ORDER BY id;","search_mode":"exact"}' \
  '{"id":4,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding, '"'"'[0, 0]'"'"') AS distance FROM docs ORDER BY distance LIMIT 2;","search_mode":"exact"}' \
  '{"id":5,"op":"shutdown"}' \
  | python -m minivectordb --data-dir ./data
```

stdout（启动握手 + 每条请求一个响应对象，每行一个）：

```json
{"type": "ready"}
{"id": 1, "ok": true}
{"id": 2, "ok": true, "affected_rows": 3}
{"id": 3, "ok": true, "columns": ["id", "content"], "rows": [[1, "一"], [2, "二"]]}
{"id": 4, "ok": true, "columns": ["id", "distance"], "rows": [[1, 0.0], [2, 1.0]]}
{"id": 5, "ok": true}
```

关闭之后 stdout 不再有任何输出（包括空行），进程以退出码 0 结束。

---

## 2. 运行协议

协议细节以题包公布的 [STDIO_PROTOCOL.md](STDIO_PROTOCOL.md) 为准，本节只说明实现要点。

### 2.1 启动握手

程序打开 `/data`、完成恢复之后，**主动**向 stdout 输出一行并 flush：

```json
{"type": "ready"}
```

每次启动（包括恢复重启）都输出一次，且一定在等待第一条请求之前。

### 2.2 请求与响应

请求是 stdin 上的一行一个 JSON 对象：

```json
{"id": 1, "op": "execute", "sql": "SELECT id FROM docs;", "search_mode": "exact"}
```

| 字段 | 含义 |
| --- | --- |
| `id` | 请求编号，响应原样回传（JSON 类型不变，数字不会变成字符串） |
| `op` | `execute` 执行 SQL，`shutdown` 正常关闭 |
| `sql` | 要执行的 SQL 字符串 |
| `search_mode` | `exact` 或 `ann`，决定向量 TopK 的检索模式 |

每个请求恰好一个响应：

| 语句 | 响应对象 |
| --- | --- |
| 查询成功 | `{"id": 1, "ok": true, "columns": ["id"], "rows": [[1], [2]]}` |
| 修改成功 | `{"id": 2, "ok": true, "affected_rows": 3}` |
| 建表 / 事务控制等 | `{"id": 3, "ok": true}` |
| 失败 | `{"id": 4, "ok": false, "error": {"code": "duplicate_key", "message": "..."}}` |

* `ok` 一定是 JSON 布尔值，不用 `0` / `1` / 字符串代替；
* `error.code` 一定是非空字符串，`message` 是可读说明（判题不依赖措辞）；
* `columns` 与 `rows` 各列顺序对应，空结果集仍返回列名；
* BIGINT 用 JSON 整数输出，不经过浮点，不丢精度；
* 向量投影输出 JSON 数组，浮点值一定是有限数（不会出现 NaN / Infinity）；
* 语句失败不会终止进程，驱动可以继续发下一条请求。

### 2.3 正常关闭

判题器发送 `{"id": 4, "op": "shutdown"}` 后，程序回滚未提交事务、完成持久化，
回传 `{"id": 4, "ok": true}`，然后 flush 并以退出码 0 结束。
**关闭响应之后 stdout 不再输出任何内容。**

崩溃测试可能直接终止进程而不发送 shutdown，已确认提交的持久性不依赖正常关闭。

### 2.4 stdout 的其它约束

* stdout 上只有协议报文：启动日志、调试信息、错误堆栈全部写 stderr；
* 每个报文一个物理行，以 `\n` 结束，写完立即 flush；
* 不输出空行、提示符、Markdown 围栏、ANSI 颜色或额外的状态行；
* 不使用重复的对象键。

### 2.5 驱动指令

`--restart`、`--mode ann` 是**驱动指令**，不是 SQL：

* `--restart`：驱动先正常关闭程序、等它退出，再用同一个数据目录重新启动，
  程序要重新输出一次握手；
* `--mode ann`：驱动把后续请求的 `search_mode` 改成 `ann`，它本身不占一条响应。

程序另外接受 `--mode exact|ann`（命令行）、`MINIVECTORDB_MODE` 等环境变量，
用于第一条请求到达之前就确定模式；正常评测由请求里的 `search_mode` 决定。

`ann` 只影响向量 TopK 查询；**即使建了近似索引，`exact` 模式仍然返回精确结果**。

### 2.6 传输层的健壮性

* 请求按**字节**读取并按行切分，一个请求不会被文本层的解码切断；
* 输入开头的 UTF-8 BOM 会被丢弃，它属于传输方式而不是请求本身；
* stdout / stderr 强制使用 UTF-8，不受宿主控制台代码页影响。

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
* 排序：`ORDER BY <列名或别名>`，并列时按主键升序；`DESC` 只影响主排序键；
* 分页：`LIMIT <n>`、`LIMIT <offset>, <n>` 与 `LIMIT <n> OFFSET <offset>`；
* 标识符不区分大小写（表名、列名都可以任意大小写书写）。

### 3.2 明确的语义约定

* **主键**唯一，重复插入报 `duplicate_key`；
* **BIGINT** 是有符号 64 位整数，超出范围的值被拒绝（`value_out_of_range`），
  输出时用 JSON 整数，不经过浮点；
* **VARCHAR** 长度按 Unicode 字符数计算，字符串比较按 Unicode 码点顺序
  （例如 `'乙' < '甲'`），不做语言相关的排序；
* **向量**是固定维度的 float32 数组：维度不符、NaN、Infinity、
  超出 float32 表示范围的值都会被拒绝；
* **L2_DISTANCE** 是欧氏距离而不是距离平方：`[3,4]` 到 `[0,0]` 的距离是 `5`；
* 带过滤的 TopK **先过滤再取 TopK**，不会先在全表取 K 条再过滤；
* 结果不足 K 条时返回全部符合条件的记录；距离相同时按主键升序；
* `LIMIT 0` 返回空结果，`LIMIT` 大于记录数时返回全部记录；
* 不支持的语句（`NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、
  `CHECK` / `UNIQUE` 约束、`OR`、`IN`、`BETWEEN`、`LIKE`、`DISTINCT`、
  嵌套事务、主键更新等）一律返回错误，绝不静默执行为别的语义；
* 失败的语句不产生任何部分生效：语句要么整体生效，要么完全没发生，
  显式事务里之前的成功修改仍然可以提交或回滚。

### 3.3 错误语义名

响应里的 `error.code` 是题包使用的 **错误语义名**，数字错误码、SQLSTATE
与更详细的信息写在 stderr 上（`minivectordb: <name> (code ..., sqlstate ...): ...`）：

| 场景 | 语义名 | MySQL 兼容码 |
| --- | --- | --- |
| 语法错误 | `syntax_error` | 1064 |
| 不支持的语法 / 功能 | `unsupported_sql` | 1235 |
| 主键重复 | `duplicate_key` | 1062 |
| 向量维度不符 | `dimension_mismatch` | 11000 |
| 值超出范围（NaN、Infinity、float32 溢出、BIGINT 越界） | `value_out_of_range` | 11000 |
| 字符串超长 | `value_too_long` | 1406 |
| 表不存在 / 列不存在 | `table_not_found` / `column_not_found` | 1146 / 1054 |
| 表已存在 / 索引已存在 | `table_exists` / `index_exists` | 1050 / 1061 |

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
  字典正文留在磁盘上，内存里只保留每个码的偏移和一个哈希索引，
  文本按需读回，所以一列文本大于内存限额时也能写入和查询；
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
`UPDATE` 只重写被赋值的列（VARCHAR 列写的是 4 字节字典码），
因此大字段不会被整行复制。

事务内的查询能看到**本事务自己的修改**：插入的行立刻可查，删除的行立刻消失，
改过的列按新值参与过滤和排序，改过的向量按新向量算距离——
普通查询与向量检索的结果始终一致，`ROLLBACK` 之后一起恢复。

### 4.3 崩溃恢复

启动时：

1. 读 `catalog.json`（上次检查点的目录）；
2. 从记录的日志位置开始重放所有已提交事务的字节补丁；
3. 重建内存中的主键索引和向量索引；
4. 写新的检查点。

日志末尾被截断的半条记录会被丢弃并截断掉 —— 这正对应
"已发出提交请求但未收到成功响应的事务，允许完整生效或完整不生效"。

检查点跟着每次提交前进（日志位置写进目录文件），
所以崩溃之后只需要重放最后一次提交之后的部分日志：
大表的日志可以到 GB 级，如果每次都从头重放，恢复时间会超过启动的时限。
恢复过程中再次被杀掉，日志仍然可以继续重放。

### 4.4 向量索引

`CREATE VECTOR INDEX idx ON t(col)` 建立一个 IVF 风格的近似索引：

* 用 k-means（k=64）把向量聚成若干簇，簇心存在 `idx_<index>.cent`；
* 每个簇一个倒排文件，查询时按簇心距离由近及远取候选；
* 每个簇记录自己的半径（成员到簇心的最大距离），查询时用
  “查询点到簇心的距离 − 簇半径” 作为下界，一旦证明后续簇不可能更近就停止
  扩展候选，召回与耗时同时得到保证；
* 新增、更新、删除的行都会同步反映到倒排表里
  （更新会按新向量重新归簇，删除靠存活位图过滤）；
* `ann` 模式按候选集计算精确距离后取 TopK，带过滤时按需继续扩展候选集，
  保证过滤条件不会被放宽；
* `exact` 模式完全扫描向量列，忽略近似索引，结果永远精确；
* 数据量小于 `MINIVECTORDB_ANN_EXACT_LIMIT`（默认 50000 行）时，
  `ann` 模式也走精确扫描，因此公开样例里 `ann` 与 `exact` 的结果完全一致。

---

## 5. 性能

`tools/benchmark.py` 按题目描述的负载做基准：10,000 条 128 维记录，
每批 100 条写入并提交，然后 1,000 次 TopK 查询（K=10，一半带等值过滤）。

```bash
python tools/benchmark.py --rows 10000 --queries 1000
```

评测环境下的口径是**容器内**跑（平台用 cgroup v2 统计耗时、磁盘 I/O 与
峰值内存），因此下表的数字来自容器：Linux 容器、512 MiB 内存上限、
数据写在容器内的卷上，驱动按题包公布的 JSONL 协议与程序交互
（因此这些数字包含协议层的解析与响应开销）。

| 指标（容器内，10,000 行 × 128 维） | exact | ann |
| --- | --- | --- |
| 批量写入 10,000 行（每批 100 条提交） | 7.27 s | 7.15 s |
| 1,000 次 TopK 查询（K=10，一半带等值过滤） | 4.43 ms/次 | 4.51 ms/次 |
| 有过滤 / 无过滤平均 Recall@10 | 1.0 / 1.0 | 1.0 / 1.0 |
| 峰值内存（cgroup memory.peak） | 47.9 MiB | 48.4 MiB |
| 数据目录体积 | 13.2 MiB | 13.2 MiB |

> 上表的 Recall 由驱动侧按题面口径暴力计算真值得到：过滤查询的真值
> 在**过滤后**的集合里算，分母是 `min(K, N)`。真值与被测程序都用 float32
> 存储向量，浮点比较容差 1e-5。

> 写入里的大头是协议层：每批 100 行的 SQL 要先被 JSON 解码、再被解析。
> 直接调用引擎（不经协议层）时，同样的负载写入是 4.3 s。

### 5.1 超过内存限额的数据

平台曾反馈过一次 `JSON response timeout`。按题面“能处理超过内存限额的数据”
这条要求复现了一个独立场景：17,000 行 × 131,072 字符的列
（逻辑数据 2,228,224,000 字节），容器仍然用 `--memory=2g --cpus=2
--read-only --network=none`。实测（容器内）：

| 操作 | 耗时 |
| --- | --- |
| 建表 | 0.004 s |
| 17,000 行写入（每批 20 条） | 345.3 s（含驱动侧构造 2.23 GB 文本的开销，单条语句均在 1 s 内） |
| 过滤扫描 `WHERE g = 2 ORDER BY id DESC LIMIT 3` | 0.003 s |
| 建向量索引 | 0.547 s |
| **全表 `UPDATE large SET g = 4`** | **0.115 s** |
| 显式事务里的全表 UPDATE 与 COMMIT | 0.044 s / 0.068 s |
| 强杀（SIGKILL）后重启并查询 | 0.002 s，数据完整 |
| 全表 DELETE | 0.202 s |

整个过程中容器峰值内存 41 MiB：这一列的取值全部相同，字典编码把它折叠成
一个条目，列文件里每行只有 4 字节的整数码。`UPDATE` 只重写被赋值的列
（这里是一列 BIGINT），不复制文本列，所以全表 UPDATE 不随文本体积变慢。

同一场景换成**每行互不相同**的文本（字典里没有可折叠的条目）时，
字典正文会写到 2.23 GB 的字典文件里，但容器峰值内存仍只有 108 MiB：
字典正文留在磁盘上，内存里只有每个码的偏移和一个哈希索引，
文本按需读回，写入和查回都正常。
（改造之前，这一列会把字典整个读进内存，5000 行就撞到 2 GiB 上限、
进程被杀掉；这正是“先量再改”的一次实例。）

性能优化要点：

* 列存 + numpy 向量化：过滤、距离计算与 TopK 选取都在 C 层完成；
* 协议层按行读请求，不做重复的全量解析，避免大语句被反复扫描；
* 距离计算分块写入复用的缓冲区，结果与朴素写法逐位相同，
  省掉每次查询的临时数组分配；
* 每块先算排序键（`np.lexsort`），只把块内最优的 K 行送进全局堆，
  避免逐行 Python 开销；
* 只有最终返回的 K 行才构造结果集；
* 存活位图常驻内存，避免每行一次磁盘读；
* 提交时按文件合并字节补丁，每个文件只打开一次；
* 热列带预算的内存缓存（`MINIVECTORDB_CACHE_BYTES`，默认 32 MiB），
  超预算时自动退回按块读取。

> 数据库进程本身按 8192 行分块流式读取列文件，不会把全表读进内存
> （`tests.test_engine.ResourceTest` 用 4 KiB 的缓存预算验证过这一点），
> 因此数据量大于内存限额时仍然可以工作。

---

## 6. 本地自测

### 6.1 用题包自带的检查工具

题包的 `protocol_check/run.py` 就是判题器同源的实现：它按正式顺序等待握手、
发送请求、读取响应，支持公开用例的重启与模式切换。

```bash
docker build -t minivectordb:local .
python3 protocol_check/run.py --image minivectordb:local --handshake-only --output protocol-results/handshake
python3 protocol_check/run.py --image minivectordb:local --output protocol-results/public
```

> `protocol_check/run.py` 使用 `os.read` 与 `selectors`，只能在 POSIX 系统上运行；
> 在 Windows 上用 `tools/run_checker.py` 包装后在容器里跑。

### 6.2 公开用例（本地驱动）

```bash
# 20 个用例 / 95 条 SQL，断言直接复用题包的 protocol_check/_assertions.py
python tools/run_case.py                  # 全部用例
python tools/run_case.py --case sql_crud  # 单个用例
python tools/run_case.py --verbose        # 打印每条语句的请求与响应
```

`tools/run_case.py` 为每个用例新建独立的数据目录，把 `.test` 文件里的 SQL
逐条包装成 `execute` 请求发给程序，把响应与 `.result` 中同位置的断言比较
（浮点按 1e-5 的绝对与相对误差比较，未标记 `unordered` 的结果按顺序比较，
标记的按多重集合比较）。`.test` 里的 `--restart` 与 `--mode ann` 按驱动
指令处理：重启复用同一个数据目录，模式切换只改变后续请求的 `search_mode`。

### 6.3 单元测试

```bash
# 功能测试（事务、持久化、恢复、索引维护、错误处理）
python -m unittest tests.test_engine -v
# 边界与协议测试
python -m unittest tests.test_edge_cases -v
# 不支持的语句（NULL / JOIN / 聚合 / 子查询 / ALTER / 外键 / 嵌套事务 / 主键更新）
python -m unittest tests.test_unsupported -v
# VARCHAR 字典（磁盘驻留、取值读回、码点序）
python -m unittest tests.test_dictionary -v
# 题面子集里公开用例没用到、但正式用例可能用的写法（分页、投影、条件）
python -m unittest tests.test_sql_forms -v
# 事务内看得到自己的修改（普通查询与向量检索）
python -m unittest tests.test_transaction_visibility -v
# 解析器测试
python -m unittest tests.test_parser -v
# 本地驱动自身的冒烟测试
python -m unittest tests.test_harness_smoke -v
# 基准
python tools/benchmark.py --mode exact
```

公开用例覆盖建表、多行插入、查询、更新、删除、数值与字符串比较、`AND`、
升降序、并列主键排序、`LIMIT`、空表、二维向量与欧氏距离、持久化与重启、
事务提交/回滚、失败语句的原子性、关闭时回滚、近似索引维护与模式切换。
它们只是公开样例，不等于正式准入通过。

---

## 7. 已知限制

* 单机、单进程、单客户端；同一时刻最多一个显式事务，不支持并发控制；
* 不支持 `NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、`CHECK` / `UNIQUE`
  约束、嵌套事务与主键更新；
* 不支持 `SELECT` 不带 `FROM` 的常量投影（`SELECT 1`），它按不支持的语法报错；
* 未提交事务的改动保存在内存里，超大事务会占用较多内存；
* VARCHAR 列使用字典编码，字典正文留在磁盘上：内存里只有偏移和哈希索引，
  所以每行一个 128 KB 的**互不相同**文本（逻辑数据 2.23 GB）也能在 2 GiB 的
  容器里写入并查回，实测容器峰值内存 108 MiB；
  但对 VARCHAR 列做**范围比较或排序**时，为了得到全局顺序仍会把字典里的
  值读进内存，超大字典下这一条路径吃内存；
* 向量索引是 IVF 风格的近似索引，`exact` 模式始终精确，`ann` 模式的召回率
  取决于聚类数与探针数；
* 主键索引常驻内存（每行约 16 字节），超大规模数据下需要进一步做磁盘化。

---

## 8. 仓库里的另一条实现线（参考）

这次开发是两条线并行推进的，仓库里保留了两份实现：

| | 提交的实现 | 参考实现 |
| --- | --- | --- |
| 代码 | `minivectordb/`（列存 + numpy） | `main.py` / `engine.py` / `storage.py` / `vector_index.py`（磁盘行存，纯标准库） |
| 入口 | `python -m minivectordb` | `python main.py --data-dir /data` |
| Dockerfile | **是**，评测构建的就是它 | 否 |
| 说明 | 本文件 | `MINIVECTORDB.md` |

参考实现是同一场开发里并行推进的另一条路线：每行一个文件、字段级稀疏补丁、
重做日志与坐标投影索引，只依赖 Python 标准库。它的价值不在速度
（同样 10,000 行 × 128 维的写入是 38.77 秒，本实现是 7.27 秒），
而在于它记录了平台给出的具体失败原因（JSON response timeout），
并复现出了两个具体场景：15 MB 的一次性 INSERT，以及 17,000 行 × 128 KB 文本
在全表 UPDATE 时的超时。

`check_bulk.py`、`check_scale.py`、`check_container.py` 使用题包的官方传输层
并针对**镜像**运行，因此它们检验的是提交的实现；
`check_recovery.py`、`check_patch_recovery.py`、`check_windows.py` 直接运行
`main.py`，检验的是参考实现。

---

## 9. 提交材料

* 源码：`minivectordb/`（引擎、解析器、协议层）与 `tools/`、`tests/`；
* 构建材料：`Dockerfile`（从源码构建，可直接启动评测容器）；
* 题包原件：`STDIO_PROTOCOL.md`、`protocol_check/`、`mysqltest/`（公开用例与断言）；
* Agent prompt 材料：`agent/conversation.json`（同一个提交的 commit 中）；
  仓库根目录的 `conversation.json` 是同一份内容的副本，两处都放置以适配不同的读取位置；
* 运行说明与设计记录：本文件。

提交时的分支为 `2026-competition-round2`，提测需要提供仓库 HTTPS 地址、
分支名与 40 位完整 commit SHA。
