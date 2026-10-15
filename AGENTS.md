# AGENTS.md — 电商智能客服 Agent

> 本文件面向后续所有协作者（人或 AI agent）。动手前先读本文件。
> 与 README / dev-notes 冲突时，以最新章的 dev-notes 为准，并回头修订本文件。
>
> **状态：v0.2（2026-10-16）。** 章级提示词 / 建表语句已到位并完成 CH02，信息不全的初版判断已按实际修订；后续每章交付后同步更新 §6。本文件是活文档，不是定稿。

## 1. 项目定位

- 电商智能客服 Agent，按教程分章交付；最终形态是求职作品（Agent 开发实习投递）。
- 这意味着：每一行代码都要能当面向面试官讲清「为什么这么设计、换方案会怎样」。禁止自己都讲不清的魔法代码。
- 每章交付物 = 能跑的代码 + 通过的测试 + 更新的 README + dev-notes 留痕 + 面试能讲的设计取舍。

## 2. 数据规则（2026-10-15 立，最高优先级）

- **各章数据以该章提示词为准，开发时由 AI 直接生成，不引入外部数据集。**（2026-10-15 用户决策：原计划引入的客服真实数据集经检查不够全，弃用；回到提示词原始方案。）
- 生成标准两个字：**逼真**。格式真实（订单号 / 手机号 / 日期 / 金额符合真实业务格式与量级）、分布合理（正常 case 为主、边界与异常 case 按需覆盖）、表达多样（不同用户说话风格、不同商品类目、不同售后类型）。
- 不同用途的数据分开设计，不互相凑合：RAG 知识库数据（FAQ / 政策文档，按文档体裁写）、工具系统查单数据（订单库表记录，按数据库记录体裁写）、对话与 eval 集（按对话体裁写）。字段与格式以各章提示词附带的建表语句 / 结构定义为准。
- 数据由脚本 / seed 生成，可复现，不手敲随机数；生成脚本进 git，生成的数据文件属项目资产可进 git，但文件头或 README 须注明「AI 合成数据，非真实用户数据」。
- 面试口径：数据为 AI 合成，设计依据是业务格式与分布合理性；不存在真实个人信息合规风险，也不宣称使用真实数据集。

## 3. 技术栈与选型纪律

- 延续 CH01：Python 3.13 + FastAPI + LangChain（langchain-openai ChatOpenAI）+ pydantic-settings + pytest。
- 模型接入统一 OpenAI 协议：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 全走 `.env`，代码零硬编码。当前默认 GLM-5.3-Flash（智谱，OpenAI 兼容）。
- 结构化输出（`with_structured_output`，CH01 售后信息抽取用）默认 `json_mode`（`llm_structured_method` 可配）：GLM-5.3-Flash 实测走 function_calling 不产 tool_calls、只在 content 输出 JSON；回退链 `function_calling` → `json_mode` → **停下来问用户**，不自行换方案。与 CH02 起的 agent 工具循环（`bind_tools` + `tool_calls`）是两套机制，勿混述——后者 GLM 实测正常产 tool_calls。
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

- **每章完结：本地 commit + push 到 origin/main（github.com/xuesentang/-agent）**——2026-10-15 用户明确授权「每开发完一章向 github 提交一次」。章中中间提交只入本地。
- 禁止 force push、禁止 rebase 公共历史。
- 禁止提交：`.env`、密钥 token、含个人信息的数据、大文件。
- 删除文件、改环境或密钥、数据库迁移、破坏性 git 操作、对外发布前需明确确认；本文件已确认的同类操作不重复问。

## 6. 当前进度

- **CH01 纯对话：已完成。** 36 项单测全绿；真实 eval 12/13 = 92%。
- **CH02 工具调用 + 数据库 + SSE 聊天：已完成（本章收尾提交待推送）。** 88 项单测全绿（含判分器单测）；工具选择 eval 13/13 连跑两轮；真实 MySQL 8.4（Docker，主机 3307）四表 + 30 条 FAQ 种子全链路验收过；三条章验收标准全过（订单 1001 物流 / 退货政策 / 「邮费是多少」漏召回）。code review 1C + 3I + 10M 已修（见 dev-notes/ch02.md）。
- git：本地 main 领先 origin/main 7 个提交（origin/main = `eade56a`），快进推送即可，无需 force。
- 待办：CH03 起按章提示词开发；无外部数据集依赖，各章数据由 AI 按提示词生成（脚本化、可复现、注明合成数据）。
