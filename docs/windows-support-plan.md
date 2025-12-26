# Windows 11 Native Support Plan

This document captures a phased plan and concrete repository changes required to run `exo-explore/exo` natively on Windows 11 (without WSL). The intent is to unblock Windows developers first and then enable production-grade inference.

## Goal Definition

- **Phase 1 (fastest path):** Windows can run Master + Dashboard and run a Worker in a "no-MLX" mode. Clusters start, nodes discover each other, REST API and dashboard function. Inference can be dummy or local-only via a Windows-friendly backend.
- **Phase 2 (full path):** Windows runs real inference and can eventually participate in multi-node sharding.

## 1) Enable Windows in tooling and dependencies

### Allow Windows in `uv`
Add Windows to the supported environments:

```toml
[tool.uv]
environments = [
  "sys_platform == 'darwin'",
  "sys_platform == 'linux'",
  "sys_platform == 'win32'",
]
```

### Gate MLX-only dependencies
Keep shared dependencies cross-platform and restrict MLX packages to Apple Silicon macOS:

```toml
[project]
dependencies = [
  # cross-platform deps
  "anyio==…",
  "fastapi==…",
  "hypercorn==…",
  "pydantic==…",
  "psutil==…",
  # …

  # Apple Silicon-only
  "mlx==… ; sys_platform == 'darwin' and platform_machine == 'arm64'",
  "mlx-lm==… ; sys_platform == 'darwin' and platform_machine == 'arm64'",
]
```

### Windows extras group
Provide a Windows install extra; two viable options:

- **Recommended starter (GGUF):**
  ```toml
  [project.optional-dependencies]
  windows = [
    "llama-cpp-python==… ; sys_platform == 'win32'",
  ]
  ```
- **Alternative (PyTorch/DirectML):**
  ```toml
  [project.optional-dependencies]
  windows = [
    "torch==… ; sys_platform == 'win32'",
    "transformers==…",
    "accelerate==…",
  ]
  ```

After updating, regenerate locks to include Windows wheels.

## 2) Guard platform-incompatible features

### `uvloop`
Windows users have reported install/runtime failures with `uvloop`. Make it optional:

```python
import sys

if sys.platform != "win32":
    import uvloop
    uvloop.install()
```

or remove `uvloop` entirely and rely on Hypercorn/AnyIO defaults.

### macOS-only monitoring
The worker includes macOS-specific metrics collection. Either add a Windows implementation (e.g., `windowsmon.py` using `psutil`/WMI) or make monitoring optional so Windows degrades gracefully instead of failing at import.

## 3) Add a Windows inference backend

Inference is currently MLX-centric. Introduce an engine abstraction and a Windows-friendly backend so workers can serve tokens.

### Engine interface
Create a lightweight interface for inference engines (e.g., `src/exo/worker/engines/base.py`):

```python
from typing import Protocol, AsyncIterator
from exo.shared.types.chunks import TokenChunk

class InferenceEngine(Protocol):
    async def load(self, model_id: str, **kwargs) -> None: ...
    async def generate(self, prompt: str, **kwargs) -> AsyncIterator[TokenChunk]: ...
```

Refactor the runner lifecycle to depend on this interface and keep the existing MLX engine under `engines/mlx/engine.py`.

### Llama.cpp engine (Windows-friendly)
Add `engines/llamacpp/engine.py` using `llama-cpp-python` (GGUF):

```python
from llama_cpp import Llama
from exo.shared.types.chunks import TokenChunk

class LlamaCppEngine:
    def __init__(self):
        self.llm = None

    async def load(self, model_path: str, n_ctx: int = 4096, **kwargs):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, **kwargs)

    async def generate(self, prompt: str, max_tokens: int = 512, **kwargs):
        for out in self.llm(prompt, max_tokens=max_tokens, stream=True, **kwargs):
            txt = out["choices"][0]["text"]
            yield TokenChunk(text=txt, finish_reason=None, model="llama.cpp")
```

Model handling: prefer GGUF artifacts for Windows; MLX remains for Apple Silicon. Linux can choose either path based on roadmap.

## 4) Networking and discovery on Windows

Windows Firewall often blocks UDP-based discovery. Document setup for allowing EXO ports (examples):

```powershell
New-NetFirewallRule -DisplayName "EXO UDP Discovery 5678" `
  -Direction Inbound -Protocol UDP -LocalPort 5678 -Action Allow

New-NetFirewallRule -DisplayName "EXO API 52415" `
  -Direction Inbound -Protocol TCP -LocalPort 52415 -Action Allow
```

Adjust ports to match runtime configuration.

## 5) Windows-native packaging

Once source installs cleanly, provide a native installer and executable:

- **PyInstaller build:** ensure entrypoints call `multiprocessing.freeze_support()` on Windows, bundle dashboard static assets, and include `exo_pyo3_bindings` DLLs.
- **Installer:** package as MSIX (Windows 11 friendly) or via WiX/Inno Setup.
- **Service option:** wrap the binary with WinSW or implement a Windows service using `pywin32`.

## 6) Deferred work: distributed inference on Windows

Full sharded inference currently depends on MLX distributed backends. A Windows port will need a non-MLX distributed path (e.g., custom TCP/gRPC comms or a PyTorch/llama.cpp-aware runtime). Initial Windows support should target single-node inference and add multi-node support later.

## Suggested implementation order

1. Enable Windows builds in `uv` and gate MLX dependencies; guard/remove `uvloop`; make macOS-specific monitoring optional.
2. Allow a Windows worker to join a cluster with dummy or no-op inference.
3. Add the `llama.cpp` engine for local Windows inference.
4. Add Windows packaging CI (PyInstaller + installer).
5. Tackle distributed inference for the Windows engine as a separate milestone.
