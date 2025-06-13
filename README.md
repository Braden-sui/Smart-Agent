# Simulated Mind

An agentic AI framework designed for self-reflection, safe self-modification, and lifelong learning.

---

## 1. Current Architecture (MVP, ✅ implemented)

| Layer                    | Key Components                                                      | Purpose                                                                                                           |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Logging / Transparency** | `logging.journal.Journal`                                           | Structured event trace across *all* subsystems.                                                                   |
| **Memory**               | `memory.dao.MemoryDAO` + `memory.mem0_client.Mem0Client`            | Multi-level key/value store with tag search & knowledge-graph capabilities via live **mem0 backend** (SDK integration complete) or in-memory fallback. |
| **Core Agents**          | `core.base_agent.BaseAgent`, `core.meta_agent.MetaAgent`, `core.sub_agent.SubAgent` | Perceive-Decide-Act loop, sub-agent orchestration, learning trigger. All log to Journal.                          |
| **Planning & Logic**     | `core.planner.Planner`, `core.logic_engine.LogicEngine`, `templates/planner_rules.py`, `templates/graphs/` | **Recursive & Graph-Based** goal decomposition. Supports YAML-defined logic graphs via `LogicEngine`. Produces standardized `Goal` objects. Robust self-tests passing. |
| **Learning**             | `learning.loop.LearningLoop`                                        | Periodic post-mortem review events stored to memory.                                                              |
| **Introspection**        | `introspection.introspector.Introspector`                           | AST snapshots & insertion-point discovery for self-analysis.                                                      |
| **Self-Modification**    | `modification.modifier.CodeModifier`                                | Safely applies code patches with backup, rollback & event hooks.                                                  |
| **Safety**               | `safety.guard.SafetyGuard`                                          | ➡️ Syntax-check via `ast.parse`<br>➡️ SHA-256 hash-chain for each file<br>(future: lint & unit-test gating).      |
| **Tests**                | `tests/`                                                            | Smoke test for agent lifecycle and safety/rollback scenarios (all **green**).                                     |

Execution flow:

1. **MetaAgent** receives an event → may **spawn SubAgents**.
2. **SubAgent** consults **Planner** (which uses **LogicEngine** for graph-based plans or falls back to templates/memory), storing/retrieving via **MemoryDAO**.
3. Periodic **LearningLoop** persists reviews.
4. When self-modification is requested, **Introspector** locates insertion points → **CodeModifier** proposes a patch → **SafetyGuard** validates → Journal logs outcome.

---

## 2. Target Architecture (🔭 roadmap)

| Gap                         | Planned Upgrade                                                                                                     |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Live Mem0 Integration**   | ✅ Connect to live mem0 backend (SDK). Support multi-user, multi-session graphs. Robust fallback to in-memory store. |
| **Planner Integration**     | 🟡 Core integration with `SubAgent` and `Goal` objects complete. Next: Advanced memory use (e.g., plan/graph recall, scoring). `TaskManager` TBD. |
| **LearningLoop**            | Automated behaviour mining, performance metrics, self-improvement patch generation.                                 |
| **SafetyGuard**             | Full linting (`ruff`, `mypy`), run selective unit-tests, dynamic sandbox, cryptographic signing for multi-writer chains. |
| **CodeModifier**            | Parallel patch queues, speculative execution sandboxes, automatic rollback on failed metrics.                         |
| **Task / Goal Management**  | Concrete `TaskManager`, `Goal` dataclass, contradiction detection, emotion model integration.                       |
| **Environment & World Sim** | Plug-in world adapters for interactive simulations or real APIs.                                                      |
| **CI & Audit**              | GitHub Actions matrix running lint + tests + hash-chain verification.                                               |
| **Documentation**           | Auto-generated API docs (Sphinx / mkdocs-material), architecture diagrams.                                          |

---

## 3. Environment setup & running tests

1. **Create the conda env once** (or update if it already exists):
    ```powershell
    conda env create -f environment.yml  # or `conda env update -f ...`
    ```
2. The repo contains a `.conda_env` marker (generated on checkout).  Add the auto-activate snippet shown below to your **PowerShell $PROFILE** so every shell that starts inside this repo (or `cd`’s into it) automatically activates the env.

    ```powershell
    # --- Auto-activate conda env when .conda_env file present ---
    function Invoke-AutoConda {
        $marker = Get-ChildItem -Path (Get-Location) -Name ".conda_env" -ErrorAction SilentlyContinue
        if ($marker) {
            $envName = Get-Content ".conda_env" | Select-Object -First 1
            if ($env:CONDA_DEFAULT_ENV -ne $envName) {
                & conda activate $envName 2>$null
            }
        }
    }
    Invoke-AutoConda
    Register-EngineEvent PowerShell.OnLocationChanged -Action { Invoke-AutoConda } | Out-Null
    ```
3. **Run tests** (now that the env is auto-activated):
    ```powershell
    pytest -q
    ```

---

**Status**: Core planning (recursive & graph-based via LogicEngine) and Mem0 integration stable. All 47 tests green ✅. Next: Deeper integration or Learning Loop enhancement.
