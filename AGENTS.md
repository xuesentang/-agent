# AGENTS.md — 电商智能客服 Agent

> 本文件面向后续所有协作者（人或 AI agent）。动手前先读本文件。
> 与 README / dev-notes 冲突时，以最新章的 dev-notes 为准，并回头修订本文件。

## 1. 项目定位

- 电商智能客服 Agent，按教程分章交付；最终形态是求职作品（Agent 开发实习投递）。
- 这意味着：每一行代码都要能当面向面试官讲清「为什么这么设计、换方案会怎样」。禁止自己都讲不清的魔法代码。
- 每章交付物 = 能跑的代码 + 通过的测试 + 更新的 README + dev-notes 留痕 + 面试能讲的设计取舍。

## 2. 数据规则（2026-10-15 立，最高优先级）

- 本项目使用一个客服领域专用、经过多次检验的真实数据集（用户提供中），**替代各章提示词中原本设计的模拟数据 / 假数据 / 随机生成数据**。
- 提示词是需求意图的表达：其中**数据来源相关描述**（「模拟数据」「生成 N 条测试对话」「假订单」等）一律以真实数据集为准；提示词中的**功能需求、接口设计、流程、验收标准仍然有效**，不因换数据而漂移。
- 能用真实数据的场景不用模拟数据；仅为单测 / 边界构造保留最小化 fixture，且 fixture 不进主数据流。
- 数据入库前必须：① 检查个人信息 / 敏感字段，脱敏或剔除；② 记录数据集名称、来源、许可证——面试时要能说清数据来历与处理方式。
- 数据文件不进 git（.gitignore 或外置目录 + 导入脚本）；大文件用「导入脚本 + 采样」，禁止整体提交。

## 3. 技术栈与选型纪律

- 延续 CH01：Python 3.13 + FastAPI + LangChain（langchain-openai ChatOpenAI）+ pydantic-settings + pytest。
- 模型接入统一 OpenAI 协议：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 全走 `.env`，代码零硬编码。当前默认 GLM-5.3-Flash（智谱，OpenAI 兼容）。
- 结构化输出默认 `function_calling`；回退链 `function_calling` → `json_mode` → **停下来问用户**，不自行换方案。
- 引入新依赖或新 API 用法前：先查 Context7 / 官方文档确认当前版本用法，查证结论写进 spec 或 dev-notes。
- requirements.txt 已含 langgraph / langgraph-checkpoint / langgraph-prebuilt / pandas 等，后续章节预计用到；新增依赖先确认用途再装。
- 数据库方言以各章提示词附带的建表语句为准（待建表语句到位后确认并回填本节）。
- 环境事实：项目解释器为系统 Python 3.13（`C:\Users\28707\AppData\Local\Programs\Python\Python313\python.exe`），`.env` 已含有效 `LLM_API_KEY`。沿用该环境，不另建 venv、不迁 WSL。

## 4. 每章工作流（固定）

1. 读该章提示词与教程整理稿 → 复述目标与验收标准 → 有歧义先问。
2. brainstorming → spec（`docs/superpowers/specs/`）→ plan（`docs/superpowers/plans/`）→ 执行（TDD：可单测走 pytest；Prompt 质量 / 抽取准确率用标注样例 eval，≥90% 通过算过）→ code review → dev-notes 留痕（随阶段追加，禁止收尾一次性补记）。
3. 单章内连续执行不中途停。**仅以下三种情况停下问用户**：① 重要问题；② 难以回退的选型决定（数据结构 / 对外接口 / 依赖选型 / 目录结构）；③ 需要用户提供的外部资源（API key、数据集文件）。其余用合理默认值推进。
4. 每章结束更新 README.md（功能表 / 验收命令 / 已知限制）。

## 5. Git 与安全纪律

- 只在本地提交；push 到远端（github.com/xuesentang/-agent）由用户自行执行。禁止 force push、禁止 rebase 公共历史。
- 禁止提交：`.env`、密钥 token、含个人信息的数据、大文件。
- 删除文件、改环境或密钥、数据库迁移、破坏性 git 操作、对外发布前需明确确认；本文件已确认的同类操作不重复问。

## 6. 当前进度

- **CH01 纯对话：已完成。** 36 项单测全绿；真实 eval 12/13 = 92%；本地 HEAD = `1c4d693`（未推送，远端 main 为旧项目历史，覆盖需用户自行 force push）。
- 待办：CH02 起按章提示词开发；数据集到位后按第 2 节规则处理（入库前脱敏检查、记录来源与许可证、导入脚本 + 采样）。
