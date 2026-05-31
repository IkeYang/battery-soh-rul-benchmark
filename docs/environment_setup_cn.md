# 中国大陆环境配置说明

本题使用 CPU 传统机器学习，不需要 GPU。

推荐 Python 版本：`3.11`。

依赖锁定在 `agent-start/requirements.txt`：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r agent-start/requirements.txt
```

如果使用 SE-Bench harness 构建镜像，不要在任务 JSON 中硬编码镜像源。通过环境变量统一注入：

```bash
export SEBENCH_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export SEBENCH_APT_MIRROR_URL=http://mirrors.aliyun.com/debian
```

如果下载 GitHub Release 或外部资产速度很慢，可在宿主机配置代理后通过 harness 环境变量传入：

```bash
export SEBENCH_HTTP_PROXY=http://127.0.0.1:7890
export SEBENCH_HTTPS_PROXY=http://127.0.0.1:7890
```

不要把代理、API key 或 Git token 写入公开任务文件。

远端真实 agent 验证还需要有效的 Codex/OpenAI/SeedEdge/OpenRouter 凭证或可复用的 Codex 登录态。只用公开 `models` 端点返回 200 不能证明 key 可用；应以需要认证的端点或一次最小 `codex exec` 作为准入检查。

本仓提供一个不打印密钥的准入检查：

```bash
python tools/long_run_preflight.py --require-live-auth
```

如果输出 `ok: false`，不要启动 30m/1h/2h/8h 长跑；先修复 `/root/.env` 中的 `SEBENCH_AGENT_API_KEY`、`SEBENCH_AGENT_API_BASE_URL`，或完成 `/root/.codex/auth.json` 登录态配置。
