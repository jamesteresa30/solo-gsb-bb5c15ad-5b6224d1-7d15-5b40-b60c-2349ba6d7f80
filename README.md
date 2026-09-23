# 自适应测验选题 API（FastAPI + SQLite）

基于 **2PL IRT（项目反应理论）** 的计算机自适应测验（CAT）后端：

- 题库保存每题的**难度 `b`、区分度 `a`、曝光次数 `exposure_count`**；
- 考生提交答案后，服务端用**一次 Newton-Raphson 更新**估计能力值 `θ`；
- 下一题在满足**曝光上限**的候选题中，按 **Fisher 信息量最大**选择，信息量并列时取**最小（稳定的）题目 ID**；
- `response_id` 为客户端幂等键：**重复提交返回首次结果，不重复更新能力值、不重复计曝光**；
- 支持可选测验时限 `duration_seconds`（60～86400 秒）：创建时按 UTC 记录截止时间 `expires_at`，到期后取题/作答原子结束会话并返回 `409`；
- 支持可选**内容标签最低配额** `tag_minimums`：键为题库现有非空 `content_tag`、值为正整数、配额总和不超过 `max_items`；剩余答题位只够补齐配额时仅从欠额标签选题，欠额标签题量不足则以 `tag_quota_unreachable` 结束且不额外计曝光；
- 提供会话进度查询与完整答题记录查询；
- 自带 30 题确定性演示题库，容器启动后即可调用。

## 目录结构

```
.
├── app/
│   ├── main.py        # FastAPI 路由：会话 / 选题 / 作答 / 进度 / 记录
│   ├── irt.py         # 2PL 概率、Fisher 信息量、单次 Newton 更新
│   ├── database.py    # SQLite 连接、建表、初始化
│   ├── models.py      # Pydantic 请求/响应模型
│   └── seed.py        # 确定性演示题库（ITEM-001 ~ ITEM-030）
├── examples/demo.py   # 仅用标准库的端到端示例（含幂等演示）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 快速开始

### 方式一：Docker Compose（推荐）

```bash
docker compose up --build
```

- API： http://localhost:8000
- 交互式文档（Swagger UI）： http://localhost:8000/docs
- 健康检查： http://localhost:8000/health
- SQLite 数据持久化在命名卷 `quiz-data`（容器内 `/data/quiz.db`）

### 方式二：本地直接运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

首次启动自动建表并灌入 30 道演示题。

### 一键跑示例

服务启动后，另开终端：

```bash
python examples/demo.py
```

脚本会：创建带 600 秒时限的会话（演示 `duration_seconds` 越界校验与 `expires_at`）→ 循环「取题 → 作答」5 轮 → 演示重复 `response_id` 幂等、重复 `/next` 不重复计曝光 → 打印进度与答题记录。

## 核心算法

### 2PL 模型与信息量

```
P(θ) = 1 / (1 + exp(-a(θ - b)))          # 答对概率
I(θ) = a² · P(θ) · (1 - P(θ))            # Fisher 信息量
```

### 作答后一次 Newton 更新

对单条二元作答 `u`（1 对 / 0 错），对数似然关于 θ 的一、二阶导数为：

```
g  = a(u - P(θ))
h  = -a²·P·(1-P) = -I(θ)
θ' = θ - g/h = θ + a(u - P) / I(θ)       # 仅一步
```

为数值稳定：分母有 `1e-4` 下限，θ 被截断在 `[-6, 6]`。注意题目要求的是
**一次** Newton 更新（非迭代至收敛），实现见 `app/irt.py: newton_step`。

### 选题规则（`POST /api/sessions/{id}/next`）

候选题必须同时满足：

1. `exposure_count < exposure_limit`（**全局**曝光上限，所有会话共享）；
2. 该题在本会话中尚未作答（不重复出题）。

候选题按 `(-信息量, 题目 ID)` 排序取第一题 —— 信息量最大优先，**完全并列时
ID 更小的题胜出**，因此选择结果确定可复现。题目被选中的同一事务内
`exposure_count += 1`。

- 若存在「已取出但未作答」的挂起题，重复调 `/next` 返回**同一题且不重复加曝光**；
- 已答满 `max_items` → 会话结束，`finish_reason = max_items`；
- 没有满足曝光上限的候选题 → 结束，`finish_reason = no_available_items`。

### 内容标签最低配额（可选 `tag_minimums`）

- 创建会话时可传 `tag_minimums`：一个 `{content_tag: 最少答到的题数}` 映射。
  - 键必须是题库中**现有且非空**的 `content_tag`（演示题库标签：`算术 / 代数 / 几何 / 概率统计 / 数论`）；
  - 值必须是**正整数**（≥1）；
  - 所有配额之和**不得超过 `max_items`**；不传或传 `{}` 表示不设配额，行为与旧版完全一致。
  - 校验不通过返回 **HTTP 422**，错误信息会指明原因与题库现有标签。
- 配额随会话持久化（`sessions.tag_minimums`，JSON），旧数据库启动时自动 `ALTER TABLE` 加列迁移，
  并在创建响应、`GET /api/sessions/{id}`、`/progress`、`/next` 中以 `tag_minimums` 返回。
- 每次取题都依据**已写入的作答记录**统计各标签已答数，计算缺口
  `gap(tag) = max(0, 配额 - 已答数)`，并在 `/next` 响应中以 `tag_gaps` 返回。
- 选题仍在原有「未作答 + 曝光上限」候选集内进行：
  - **有冗余位时**（`max_items - 已答数 > 缺口总数`）：照常从**全部候选**中按信息量/ID 选题；
  - **剩余答题位恰等于缺口总数**时：后续每题都必须用于补配额，候选池**收窄为欠额标签的题**，
    仍按相同的信息量最大、ID 最小规则选择；响应 `note` 会说明本次被收窄到哪些欠额标签。
- 进入收窄选择时，若某欠额标签满足曝光/未作答限制的可用题数**已不足以补齐自身缺口**，
  立即结束会话，`finish_reason = tag_quota_unreachable`（`note` 列出各标签还缺几题），
  **不选题、不增加任何曝光**。
- 挂起题（已取未答）的重复 `/next` 幂等行为不变；进度接口额外返回 `tag_answered`
  （每个配额标签在本会话已答的题数）。

### 测验时限（可选 `duration_seconds`）

- 创建会话时可传 `duration_seconds`，取值 **60～86400 秒**；不传则无时限（行为与旧版一致）。
- 服务端在创建时按 **UTC** 计算并持久化截止时间 `expires_at = created_at + duration_seconds`
  （ISO-8601，秒级），在创建响应以及 `GET /api/sessions/{id}`、`/progress` 中一并返回。
- `POST /next` 与 `POST /responses` 在**写事务内最先检查**截止时间：一旦 `now(UTC) >= expires_at`，
  即在同一事务内原子地把会话置为 `finished`、`finish_reason = time_expired`、清空挂起题，
  并以 **HTTP 409** 拒绝本次操作；**不会**增加任何曝光计数，也**不会**写入答题记录。
  未到期时完全保留原有行为（包括挂起题不重复计曝光、`response_id` 幂等）。
- 只读的会话/进度/记录查询不触发自动结束；客户端可据 `expires_at` 自行倒计时。

### 幂等作答（`POST /api/sessions/{id}/responses`）

- 客户端为每次作答生成唯一 `response_id`（建议 UUID），与请求体一起提交；
- 服务端首次收到时评分（提交的选项键与题库 `answer` 比对）、做一次 Newton
  更新、清空挂起题、写入不可变记录；
- **同一 `response_id` 再次提交：以 `200` 返回首次存储的结果（首次为 `201`），不再更新 θ、不再计曝光。**
  若该 `response_id` 对应的是其他会话/题目，返回 `409`。

所有写操作使用 `BEGIN IMMEDIATE` 事务，SQLite WAL 模式下并发请求串行化，
避免曝光计数与能力更新丢失；`responses(id)` 主键与 `UNIQUE(session_id, item_id)`
在数据库层面兜底并发竞争。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | `/health` | 健康检查、题库题量 |
| GET  | `/api/items` | 题库列表（含曝光次数，分页 `limit/offset`） |
| GET  | `/api/items/{item_id}` | 单题详情（管理用，含正确答案） |
| POST | `/api/sessions` | 创建测验会话 |
| GET  | `/api/sessions/{id}` | 会话状态 |
| GET  | `/api/sessions/{id}/progress` | **会话进度**（θ、已答数、挂起题、已答题目） |
| GET  | `/api/sessions/{id}/responses` | **答题记录**（逐题评分与 θ 轨迹） |
| POST | `/api/sessions/{id}/next` | 自适应选下一题 |
| POST | `/api/sessions/{id}/responses` | 提交答案（幂等，一次 Newton 更新） |

完整字段与可交互试用见 http://localhost:8000/docs。

## curl 调用示例

```bash
# 1) 健康检查
curl -s http://localhost:8000/health
# {"status":"ok","bank_size":30}

# 2) 创建会话（初始 θ=0，最多 10 题，单题曝光上限 5）
curl -s -X POST http://localhost:8000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"initial_theta": 0.0, "max_items": 10, "exposure_limit": 5}'
# 记下返回的 "id"，导出 SID=...

# 2b) 也可以在创建时指定测验时限（秒，60~86400）；
#     返回中会带 UTC 截止时间 "expires_at"
curl -s -X POST http://localhost:8000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"initial_theta": 0.0, "max_items": 10, "duration_seconds": 600}'
# {"id":"...", "duration_seconds":600,
#  "expires_at":"2026-09-23T08:10:00+00:00", "status":"active", ...}

# 3) 选下一题
curl -s -X POST http://localhost:8000/api/sessions/$SID/next
# item.id / item.stem / item.options / information / exposure_count ...
# 若已过 expires_at：HTTP 409，
# {"detail":"测验时限已到（UTC 截止时间 ...），会话已结束：finish_reason=time_expired"}

# 4) 提交答案（response_id 为客户端幂等键，务必唯一）
curl -s -X POST http://localhost:8000/api/sessions/$SID/responses \
  -H 'Content-Type: application/json' \
  -d '{"response_id": "'$(uuidgen)'", "item_id": "ITEM-xxx", "response": "A"}'
# 返回 correct、score、theta_before、theta_after、p_correct ...
# 到期后提交同样返回 409，且不写入答题记录、不增加曝光

# 5) 用完全相同的请求体重放一次 -> 返回首次结果，theta_after 不变

# 6) 查询会话（含 expires_at）、进度与答题记录
curl -s http://localhost:8000/api/sessions/$SID
curl -s http://localhost:8000/api/sessions/$SID/progress
curl -s http://localhost:8000/api/sessions/$SID/responses
```

重复步骤 3-4 即可完成整场测验；当 `status` 变为 `finished` 时，可从
`finish_reason` 得知结束原因（`max_items` / `no_available_items` /
`time_expired` / `tag_quota_unreachable`）。

### `tag_minimums` 调用示例

```bash
# 创建会话：4 题中至少答 2 道「代数」、1 道「几何」（配额和 3 ≤ max_items 4）
curl -s -X POST http://localhost:8000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"max_items": 4, "exposure_limit": 5,
       "tag_minimums": {"代数": 2, "几何": 1}}'
# {"id":"...", "max_items":4, "tag_minimums":{"代数":2,"几何":1}, "status":"active", ...}
# 记下返回的 "id"，导出 SID=...

# 非法配额会被 422 拒绝：
#   {"代数": 1}  配额之和超过 max_items（如 {"代数":3,"几何":2} 而 max_items=4）
#   值不是正整数（0 / 负数 / 小数）
#   键不是题库现有非空 content_tag（如 {"不存在的标签": 1}）

# 取题：响应中带 tag_gaps（当前各标签缺口）。有冗余位时从全部候选题中选；
# 当剩余答题位 == 缺口总数时，仅从欠额标签选题并在 note 中说明：
curl -s -X POST http://localhost:8000/api/sessions/$SID/next
# {"item": {"id": "ITEM-002", "content_tag": "代数", ...},
#  "tag_minimums": {"代数": 2, "几何": 1},
#  "tag_gaps": {"代数": 2, "几何": 1},
#  "note": "剩余答题位仅够补齐标签配额：本次仅从欠额标签 ['代数', '几何'] 中按信息量与 ID 规则选题"}

# 若欠额标签的可用题数不足以补齐配额：
# {"status": "finished", "item": null,
#  "finish_reason": "tag_quota_unreachable",
#  "note": "欠额标签的可用题数已不足以补齐配额（几何 缺 1），会话结束且不额外增加曝光"}

# 进度接口返回配额与各标签已答数：
curl -s http://localhost:8000/api/sessions/$SID/progress
# {"answered_count": 3, "max_items": 4,
#  "tag_minimums": {"代数": 2, "几何": 1},
#  "tag_answered": {"代数": 2, "几何": 1}, ...}
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QUIZ_DB_PATH` | `data/quiz.db`（本地）/ `/data/quiz.db`（镜像） | SQLite 文件路径 |
| `DEFAULT_EXPOSURE_LIMIT` | `5` | 创建会话未指定曝光上限时的默认值 |

## 说明与边界

- 模型为 2PL（无猜测参数）；演示题为简单算术题，重点展示选题与能力更新机制。
- 曝光计数是**跨会话全局**的：用于模拟真实 CAT 的曝光控制（曝光上限可按会话放宽/收紧）。
- SQLite + WAL 适合单机/演示与中小流量；写事务已串行化，水平扩展时可替换连接层。
