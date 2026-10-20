# CH04 评估真实化执行计划（E0-E7）

对应 spec：`docs/superpowers/specs/2026-10-20-ch04-eval-realism-design.md`

| 阶段 | 内容 | 产出 | 验证 |
|---|---|---|---|
| E0 | 冻结基线 + 更正小库结论 | `tests/eval/ch04_cases_baseline.jsonl`、`reports/ch04_retrieval_baseline.md`、dev-notes 更正段 | 基线副本与原文件字节一致 |
| E1 | 事实骨架（12 篇新文档） | `scripts/kb_blueprint.py` | 骨架自检：型号/数值/版本冲突点齐全且互不矛盾 |
| E2 | 扩写生成器 | `scripts/expand_knowledge.py` → `data/knowledge_docs/*.md` | 每篇关键事实逐字回检；总字数 ≥120k |
| E3 | 灌库与量级验证 | `data/devstack_milvus.db` 重建 | 块数 ≥350；Top-10 占全库 ≤3% |
| E4 | 新评估集 | `tests/eval/ch04_cases_v2.jsonl`（含噪声变体） | 每桶题数、D 桶 far/near 各 ≥6；`expected_section` 全部能在库里找到 |
| E5 | 评估脚本改造 | `tests/eval/run_rag_eval.py`、`app/config.py` | 单测覆盖新口径（subtype/双口径/judge 回落） |
| E6 | 跑评估 + 人工抽检 | `reports/ch04_eval_v2.md` | 验收标准 1-7 |
| E7 | 文档与提交 | dev-notes / README / 能力差异.md | 与基线逐项对比说明 |

顺序不可跳：E3 的块数不达标，E6 的指标就没有意义；E2 的陷阱写错，E4 的 near 题就没有 ground truth。
