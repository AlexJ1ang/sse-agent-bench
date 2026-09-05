# Contributing

感谢你为 agent-harness 做贡献。

## 本地开发

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

请为行为变化补充测试，避免在测试、日志、trace 或示例中提交真实客户数据、设备编号、访问令牌和内部接口地址。

提交 Pull Request 前请确保测试、lint 和类型检查全部通过，并在描述中说明变更动机与验证方式。
