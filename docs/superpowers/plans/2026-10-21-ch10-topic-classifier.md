# CH10 实施计划（T1-T8）

TDD 口径：能单测的先写测试；训练/预标/前端按需求「拿样例或 eval 跑一遍验证」。

## T1 表与数据层
- `app/models/topic_classification.py`：id / question_id(uniq) / labels(JSON) / classified_at。
- `app/models/__init__.py` 注册。
- `app/repositories/topic_repo.py`：`upsert_many`（同 question_id 更新标签）、
  `list_classified_ids`、`list_unclassified`（池子 LEFT JOIN）、`distribution`、`list_by_label`。
- `scripts/sql/ch10_schema.sql`（照 `各章建表.md` 605-640）。
- 测试 `tests/test_topic_repo.py`。

## T2 术语表 + 推理服务
- `app/ml/topic/taxonomy.py`：17 类 + 边界说明 + 近邻对照 + `validate_labels()`。
- `app/ml/topic/labels.json`：从 lab 拷，作模型契约。
- `app/ml/topic/classifier.py`：`TopicClassifier`（惰性 import、批推理、阈值、sigmoid）、
  `TopicModelUnavailable`、`load_classifier()` 单例。
- 测试 `tests/test_topic_taxonomy.py`、`tests/test_topic_classifier.py`（假 session / 假 tokenizer，
  真模型缺失时跳过集成用例）。

## T3 模型资产 + 训练导出脚本 + 评测
- `scripts/fetch_topic_model.py`：从 lab 拷 onnx+tokenizer，校验 SHA256。
- 冻结数据 750KB 进 `app/ml/topic/data/`（含 manifest）。
- `scripts/train_topic_model.py` / `scripts/export_topic_onnx.py`（由 lab 脚本适配仓库路径）。
- `scripts/eval_topic_model.py` → `reports/ch10_topic_eval.md`（每类 P/R/F1 + 混淆矩阵）。
- 实跑：微 F1 与 0.9687 对齐（±0.002）。

## T4 批量归类服务 + CLI
- `app/services/topic_service.py`：`classify_pending(session_factory, batch_size)` → 统计 dict。
- `scripts/classify_topics.py --batch 50 [--dry-run]`。
- 测试 `tests/test_topic_service.py`（假分类器，验幂等与跳过）。

## T5 LLM 预标 + 抽审
- `scripts/prelabel_topics.py`：预标 → jsonl + review md；`--apply` 回写与准确率。
- 实跑池内真实问题，产物落 `reports/ch10_prelabel/`。

## T6 数据四步脚本
- `scripts/prepare_topic_data.py`：清洗/术语表/80-10-10/增强 → v2 + manifest。
- 测试 `tests/test_prepare_topic_data.py`（纯函数：清洗、分层、增强只动训练集）。

## T7 前端主题分布页 + 只读接口
- `GET /api/v1/topics/distribution`、`GET /api/v1/topics/questions`。
- `app/static/admin_topics.html` + `admin_topics.js`（条形图 + 最近归类列表）。
- 测试 `tests/test_topic_routes.py`；`node --check`。

## T8 收尾
- README CH10 段（功能、验收命令、配置项、已知限制）。
- dev-notes/ch10.md 各阶段追记；能力差异 CH10 节。
- 全量单测 → commit → push。
