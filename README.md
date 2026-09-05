# agent-harness

一个面向 **SSE 流式 LLM Agent 服务**的通用黑盒测试框架，把功能评测、性能压测、回归比较和 CI 质量门禁放在同一套命令行工具中。

框架不依赖具体 Agent SDK、编排框架或业务领域。你只需要用 YAML 描述自己的 HTTP 请求结构、SSE 事件格式和测试期望，无需修改 agent-harness 源码。

> 仓库中的服务地址、城市、工具名和用例均为虚构示例，不包含真实客户或设备数据。

## 能做什么

- 功能评测：意图、工具调用、关键词、正则和参考答案数值校验
- 开放问题评分：可选 LLM-as-Judge、多维评分和一致性检查
- 性能压测：并发、持续时间、吞吐、TTFT、延迟分位数和流中断率
- 稳定性分析：重复采样、flaky 检测和 Wilson 置信区间
- 回归分析：基线比较、McNemar/paired-t 显著性检验
- 可追溯性：保存完整 trace，并对回答、工具调用和耗时进行回放 diff
- CI 门禁：按照最低分、通过率和最大回归数阻止不合格发布

## 安装

需要 Python 3.11 或 3.12。

```bash
git clone https://github.com/AlexJ1ang/agent-harness.git
cd agent-harness
python -m venv .venv
```

激活虚拟环境：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

安装：

```bash
python -m pip install -e .

# 参与开发时安装测试、lint 和类型检查工具
python -m pip install -e ".[dev]"
```

## 五分钟接入自己的 Agent

### 1. 复制示例配置

```bash
cp examples/config.yaml config.local.yaml
```

Windows PowerShell：

```powershell
Copy-Item examples/config.yaml config.local.yaml
```

`config.local.yaml` 已被 Git 忽略，可以安全地放置本地地址或认证请求头。仍建议优先使用短期令牌，并避免把任何密钥写入可提交文件。

### 2. 映射 HTTP 请求

假设你的接口接收：

```json
{
  "message": "用户问题",
  "conversation_id": "动态会话 ID",
  "metadata": {"locale": "zh-CN"}
}
```

把 `target.request_adapter.body_template` 改成：

```yaml
target:
  base_url: "http://localhost:8000"
  chat_path: "/api/agent/stream"
  method: POST
  request_adapter:
    body_template:
      message: "{question}"
      conversation_id: "{session_id}"
      metadata: "{context}"
```

可用占位符：

| 占位符 | 来源 |
|---|---|
| `{question}` | 当前测试问题 |
| `{case_id}` | 用例 ID |
| `{context}` | 当前用例的完整 context 对象 |
| `{session_id}` | 每次执行自动生成 |
| `{user_id}` | 根据用例 ID 自动生成 |
| `{任意字段}` | 从当前用例的 `context` 中读取同名字段 |

模板可以包含嵌套对象和数组。若接口需要固定请求头，可写入 `extra_headers`：

```yaml
request_adapter:
  extra_headers:
    X-Client: "agent-harness"
```

认证头请只放在被忽略的 `config.local.yaml` 中，不要提交真实令牌。

也可以用环境变量覆盖常用连接项：

```bash
HARNESS_BASE_URL=http://localhost:9000
HARNESS_CHAT_PATH=/v2/chat/stream
HARNESS_JUDGE_API_KEY=your-key
HARNESS_JUDGE_API_BASE=https://api.openai.com/v1
HARNESS_JUDGE_MODEL=gpt-4o-mini
```

### 3. 映射 SSE 协议

框架期望标准 SSE 文本流，但事件名和 JSON 字段可以自由映射。例如服务返回：

```text
event: token
data: {"text":"Hello"}

event: function_call
data: {"function":"get_weather","args":{"city":"Example City"},"id":"call-1"}

event: finished
data: {"trace_id":"trace-1"}
```

对应配置：

```yaml
target:
  sse_mapping:
    answer_event: "token"
    answer_content_field: "text"
    tool_call_event: "function_call"
    tool_name_field: "function"
    tool_args_field: "args"
    tool_call_id_field: "id"
    done_event: "finished"
    error_event: "error"
    error_message_field: "message"
```

没有结束事件的流会被标记为 `stream_interrupted`。未映射的事件仍会进入 trace，但不参与对应指标计算。

### 4. 为自己的项目编写测试集

复制 `examples/suites/smoke.yaml`，然后替换问题、上下文和期望：

```yaml
id: my-agent-regression
name: My Agent Regression Suite

cases:
  - id: weather-001
    question: "What is the weather in Example City today?"
    context:
      locale: "en-US"
    expectations:
      expected_intents:
        main: "weather_query"
      expected_tools:
        - "get_weather"
      expected_keywords:
        - "Example City"
      expected_patterns:
        - "-?\\d+\\s*°C"
      is_open_ended: false
```

期望字段：

| 字段 | 用途 |
|---|---|
| `expected_intents` | 检查状态事件中识别出的意图 |
| `expected_tools` | 检查工具是否被调用，不要求调用顺序 |
| `expected_keywords` | 检查最终回答是否包含关键词 |
| `expected_patterns` | 用正则检查最终回答 |
| `reference_answer` | 检查参考答案中的数值是否被回答覆盖，也供 Judge 参考 |
| `is_open_ended` | 为 `true` 时启用 LLM-as-Judge |

测试集支持 YAML、JSON 和 CSV。业务数据应使用虚构值，真实回归用例请放在私有仓库或被 Git 忽略的文件中。

### 5. 运行评测

```bash
agent-harness eval \
  --config config.local.yaml \
  --suite examples/suites/smoke.yaml
```

默认生成 JSON、HTML、trace 和基线：

```text
harness_output/
├── report_<run_id>.json
├── report_<run_id>.html
├── baselines/latest.json
├── traces/<run_id>/<case_id>.json
└── judge_cache/
```

`harness_output` 可能包含完整问题、回答、请求体和工具参数，因此默认被 Git 忽略。

## LLM-as-Judge

确定性的行为优先使用规则评分；只有无法通过规则准确判断的开放式回答才建议启用 Judge。

```yaml
eval:
  scorers_enabled: [intent, tool, fact, llm_judge]
  judge:
    enabled: true
    model: "gpt-4o-mini"
    pass_threshold: 3.0
    cot_enabled: true
    consistency_check: false
```

设置密钥后运行：

```bash
export HARNESS_JUDGE_API_KEY="your-key"
agent-harness eval -c config.local.yaml -s suites/regression.yaml
```

PowerShell 使用 `$env:HARNESS_JUDGE_API_KEY = "your-key"`。Judge API 需兼容 OpenAI Chat Completions 接口。

## 性能压测

```bash
agent-harness load \
  --config config.local.yaml \
  --suite examples/suites/smoke.yaml \
  --concurrency 10 \
  --duration 60
```

请只对你拥有或明确获准测试的服务执行压测，并先在测试环境确认容量限制。

## 回放与报告

```bash
# 查看已有基线和 trace run
agent-harness list

# 回放单个用例并生成回答、工具和耗时 diff
agent-harness replay -c config.local.yaml -s suites/regression.yaml \
  --run-id <run_id> --case-id <case_id>

# 从 JSON 重新生成 HTML
agent-harness report -i harness_output/report_<run_id>.json

# 预览将被清理的旧 trace；加 --yes 才实际删除
agent-harness prune --keep 20
```

## CI 质量门禁

```bash
agent-harness eval \
  -c config.local.yaml \
  -s suites/regression.yaml \
  --gate \
  --gate-min-score 0.70 \
  --gate-min-pass-rate 0.90 \
  --gate-max-regressions 0
```

任一门禁条件失败时命令返回非零退出码。仓库中的 `.github/workflows/ci.yml` 用于验证框架自身；把 Agent 回归评测加入你的项目 CI 时，还需要启动被测服务并安全注入配置和密钥。

## 开发

```bash
pytest
ruff check .
mypy src
```

## 数据安全

- 不要提交 `harness_output`、本地配置、API Key 或访问令牌。
- trace 会保存完整请求、SSE 事件、回答和工具参数；按敏感业务数据管理。
- 对公开示例使用虚构的人名、组织、设备、账户和服务地址。
- 在把真实生产数据用于 Judge 前，确认数据处理与合规要求。

## License

Apache License 2.0，详见 [LICENSE](LICENSE)。
