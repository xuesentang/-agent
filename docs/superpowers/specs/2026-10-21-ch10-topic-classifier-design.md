# CH10 设计：微调主题分类器 + 飞轮主题分布

> 需求原文：`各章提示词.md` 629-696 行。3 条验收。
> 前置事实：模型已在 `D:\python-kefu-agent\finetune-lab` 训完（RoBERTa-wwm-ext 全参微调，
> 冻结集 2614/150/150，测试 micro-F1 0.9687，ONNX 409MB，阈值 0.20）。

## 0. 本章的边界（用户 2026-10-21 拍板）

| 决定 | 选的是 | 代价 |
|---|---|---|
| 数据集 | 沿用 lab 冻结集与已训模型 | 语料是 legacy 合成语料而非飞轮池真实问题；切分 ≈86:5:5 不是 80:10:10 —— 在 README 与 dev-notes 如实写明 |
| 标注 | 补 LLM 预标 + 人工抽审清单 | 只跑在池内真实问题与新增数据上，**不碰冻结训练/测试集**，已测出的 0.9687 不受影响 |
| 训练脚本 | train / export / prepare 三个脚本搬进仓库 | 仓库多 ~500 行训练代码，需要 torch+transformers（本机已装） |

因此本章分两条线，互不干扰：

- **训练线（离线、可复现）**：`scripts/train_topic_model.py` / `export_topic_onnx.py` 用仓库内冻结数据重训重导，评测脚本对齐 lab 的数字。
- **接入线（本章主体）**：推理服务 + 批量归类 + 入库 + 分布页，全部在旁路，主链路一行不碰。

## D1 模型资产不进 Git，用脚本拉 + SHA256 校验

ONNX 409MB、tokenizer 439KB → 落 `models/topic_classifier/`（`.gitignore` 已忽略 `/models/`）。
`scripts/fetch_topic_model.py --from <lab 目录>` 拷贝并校验 `SHA256SUMS.txt` 里的哈希，
校验不过直接报错退出——半截文件比没有文件更难查。

仓库只提交小而关键的东西：`labels.json`（1KB）、`taxonomy.py`、冻结数据 750KB、
评测报告（几 KB）。

## D2 推理依赖惰性导入，缺依赖/缺模型只影响旁路

`onnxruntime` / `tokenizers` 在 `classifier.py` 里函数内 import。缺包或模型文件不在 →
抛 `TopicModelUnavailable`，服务层捕获后返回「跳过 + 原因」，日志一行。
理由：这是旁路功能，装不上依赖不该让客服服务起不来（需求明说实时主链路不调它）。

## D3 17 类术语表是代码里的唯一来源

`app/ml/topic/taxonomy.py` 存 17 类顺序、每类一句边界说明、近邻对照（修归保修维修、
退归退换货；运费管钱、物流管货；价保补差价、优惠活动是券和满减）。
训练标签顺序、预标 prompt、前端图表文案全从这里取，不许第二份。

`labels.json` 保留但只作模型契约（与 ONNX 输出位次对齐），启动时用 taxonomy 校验一次，
不一致直接抛——顺序错一位，全表标错。

## D4 批量归类幂等，按批跑，不做实时

- `topic_classifications.uk_question_id` 唯一键：同一问题重复归类只更新标签，不新增行。
- 批大小 `TOPIC_BATCH_SIZE`（默认 50）：「攒够一批归一次」由脚本/定时任务控制。
- 只挑「池里还没归过类」的问题（LEFT JOIN 或 NOT IN），避免每跑一次全表重算。
- 归类失败（模型缺失）不写库，返回统计里的 `skipped` 原因。

## D5 数据四步产出 v2，用来补数据链路，不重训已交付模型

`scripts/prepare_topic_data.py` 四步：清洗（脱敏手机号/订单号/邮箱、空白与全角规范化、
去重）→ 归并术语表（越界标签归「其他」并计数）→ 按主标签分层 80/10/10 →
数据增强（同义词替换 + 句式微调，**只扩训练集**）。产 `manifest.json`（行数/标签数/sha256）。

v2 与训练用的 v1 是两份东西，README 写清楚：**测试分数来自 v1，v2 只证明四步跑得通**。

## D6 评测报告用 ONNX 跑冻结测试集

`scripts/eval_topic_model.py`：加载 ONNX + 冻结 `test.jsonl` → 阈值 0.20 → 每类
P/R/F1/support + 每类混淆矩阵（tp/fp/fn/tn）→ `reports/ch10_topic_eval.md` + json。
微 F1 与 lab 的 0.9687 对齐容差 ±0.002，超了说明模型资产或阈值接错了，直接报错。

## D7 预标走 LLM，抽审出清单，人工修订可回写

`scripts/prelabel_topics.py`：
1. 取池内问题（默认全部未归类的）+ 可选 LLM 造数补足（`--synth N`，照术语表造）；
2. 一条一次调用，要求只输出 JSON `{"labels": [...]}`，标签必须落在 17 类内；
3. 写 `reports/ch10_prelabel/<时间>.jsonl`，并按比例抽 `--review N` 条生成
   `reports/ch10_prelabel/review_<时间>.md`（原话 + 预标 + 空栏留给人工改）；
4. `--apply <reviewed.jsonl>` 把人工核定后的标签写回，统计预标准确率。

抽审是抽查不是全量：需求原话「大模型照术语表预标、人工抽审这条折中路线」。

## D8 前端独立页 `/admin/topics`（CH09 未做，先独立）

CH09 飞轮后台没做，本章不假装有它：主题分布页单独一页，页内两栏——上排 17 类横向条形图
（按问题量），下排最近归类的问题列表（原话 + 标签 + 时间）。README 注明「CH09 补上飞轮后台后并入」。

两个只读接口：
- `GET /api/v1/topics/distribution` → `{total, labels:[{label,count}], sources:[...]}`；
- `GET /api/v1/topics/questions?label=尺码&limit=50` → 带原话与标签的列表。

## D9 阈值与开关进配置，不硬编码

`TOPIC_THRESHOLD=0.20`（lab 在验证集上扫出来的）、`TOPIC_BATCH_SIZE=50`、
`TOPIC_MODEL_DIR=./models/topic_classifier`、`TOPIC_ENABLED=false`（默认关，验收时开）。
阈值写死会把「换模型要改代码」这件事埋进细节里。

## D10 测试策略

能单测的都单测（repo / taxonomy / classifier 用假 session 与假 tokenizer / service / 路由）；
不能单测的三类用实跑替代：
- 训练与导出：拿 lab 的现成产物对齐（评测脚本本身就是校验）；
- 预标 Prompt：跑池内 39 条真实问题，人工看标签质量（`reports/ch10_prelabel/`）；
- 前端图表：`node --check` + 手工过一遍页面。

## 验收对应

1. 每类 F1 + 混淆矩阵 → `reports/ch10_topic_eval.md`（D6）
2. 一批真实问题跑归类、分布页看到统计 → `scripts/classify_topics.py` + `/admin/topics`（D4/D8）
3. 多诉求句子同时命中多类 → `inference` 实测「买大了想退」→ 退换货 + 尺码（D2 的输出形态）

## 做完了但不属于本章的东西

- CH09 的飞轮后台页面、review_queue 表、eval_runs 表：没做就是没做，本章不补。
  主题分布页先独立，将来并入。
