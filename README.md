# MewCode Lite

[English version → README.en.md](README.en.md)

一个**不用任何 Agent 框架、手写实现**的终端编码 Agent（最小可用版）。
你在终端里给它一句任务，它会自己读文件、写文件、搜代码、跑命令，一步步把任务做完。

> 状态：学习型项目 v0.1。目标是把 Agent 最核心的机制亲手实现一遍并讲清楚，
> 而不是做一个大而全的产品。已实现的功能见下表，未实现的在「局限与路线图」里如实列出。

## 已实现

| 机制 | 实现位置 | 说明 |
|---|---|---|
| ReAct 式主循环 | `mewcode/agent.py` | 模型回复 → 有 tool_calls 就执行 → 结果回传 → 再问模型，直到给出最终文本 |
| 工具层（4 个工具） | `mewcode/tools.py` | `read_file` / `write_file` / `grep_search` / `run_shell`，均为 OpenAI 风格 function schema |
| 流式 tool_calls 拼接 | `mewcode/llm.py` `assemble_stream` | 流式响应里一个工具调用会被拆成很多片段（arguments 是字符串碎片，按 index 分组），这里把碎片拼回完整调用 |
| 结构化错误反馈 | `mewcode/agent.py` `_execute` | 工具出错（参数 JSON 坏了、文件不存在、命令被拦）不崩溃，统一回传 `{"ok": false, "error": ...}`，让模型自己改参数重试 |
| 最大步数限制 | `Agent(max_steps=...)` | 防止模型无限打转烧 token |
| 重复调用检测 | `Agent(repeat_limit=...)` | 同一工具 + 同一参数反复调用超过上限，判定陷入死循环并停止 |
| 安全边界 | `mewcode/tools.py` | 文件工具被限制在工作区内（拒绝 `../` 逃逸）；shell 有危险命令黑名单；非只读命令必须人工确认 |
| 模型客户端抽象 | `mewcode/llm.py` `LLMClient` | 主循环只依赖协议接口；实现了 OpenAI 兼容客户端（仅用标准库）和离线用的脚本化 FakeLLM |

## 快速开始

需要 Python 3.10+。运行库本身**零第三方依赖**（只用标准库）；pytest 只在跑测试时需要。

```bash
git clone https://github.com/yang-cloud-0/mewcode.git
cd mewcode
python3 -m venv .venv && source .venv/bin/activate
pip install pytest        # 仅测试需要
```

### 1. 离线 demo（不需要 API key、不联网）

```bash
python -m mewcode --demo
```

demo 用脚本化的假模型驱动**真实的**主循环和工具，在临时目录里创建 `hello.py` 并真的执行它。

### 2. 跑测试

```bash
pytest -q
# 19 passed
```

测试覆盖：工具读写与路径逃逸拦截、grep、shell 安全策略、流式 tool_calls 分片拼接、
主循环的错误反馈恢复、最大步数与重复调用检测。

### 3. 接真实模型

兼容任何 OpenAI 格式的 chat-completions 接口（OpenAI / DeepSeek / 通义等）：

```bash
export MEWCODE_API_KEY=你的key
export MEWCODE_MODEL=gpt-4o-mini            # 可改成你要用的模型
export MEWCODE_BASE_URL=https://api.openai.com/v1
python -m mewcode "在当前目录创建一个冒泡排序脚本并运行验证"
```

非只读的 shell 命令执行前会在终端里要求你确认（y/N）。

## 工作原理（一分钟版）

```
用户任务
  │
  ▼
┌───────────────┐   tool_calls    ┌───────────────┐
│  Agent 主循环  │ ──────────────▶ │   工具层       │
│ (agent.py)    │ ◀────────────── │ (tools.py)    │
└───────────────┘  {"ok": ...}    └───────────────┘
  │  ▲
  │  │ 流式分片在 llm.py 里拼装成完整的回复 / tool_calls
  ▼  │
  模型 API（OpenAI 兼容）
```

一句话：**模型负责“想下一步做什么”，主循环负责“执行并把结果如实喂回去”，
工具层负责“真干活并守住安全边界”。**

## 局限与路线图

如实说明，目前**没有**实现：

- [ ] Anthropic 等其他协议的接入（当前只有 OpenAI 兼容协议）
- [ ] 上下文压缩 / token 预算管理（长任务会撞上下文上限）
- [ ] Todo 规划工具与多文件任务的计划模式
- [ ] 评测集（用固定任务集量化通过率 / 平均步数）
- [ ] 会话持久化与断点续跑

下一步计划：先补一个小评测集（修 bug / 加函数 / 补测试三类任务），
再做上下文压缩；租房合同审查类 RAG Agent 会作为独立的第二个项目。

## 目录结构

```
mewcode/
├── mewcode/
│   ├── agent.py    # 主循环：步数限制、重复检测、结构化错误反馈
│   ├── llm.py      # 客户端抽象、流式 tool_calls 拼接、FakeLLM
│   ├── tools.py    # 4 个工具 + 工作区 confinement + shell 安全策略
│   └── cli.py      # 命令行入口与离线 demo
├── tests/          # 19 项 pytest 测试
├── README.md       # 本文件
└── README.en.md    # English version
```

## License

MIT
