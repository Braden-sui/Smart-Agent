# Planner Improvement To-Do List

A concise, data-driven roadmap to evolve the logic module into a state-of-the-art (SOTA) reasoning engine while keeping code lean and transparent.

---

## 0. Core Principles

- **Less code, more data** – behaviour lives in editable graphs/templates rather than nested Python.
- **Declarative reasoning** – small set of primitives (`SEQ`, `PAR`, `COND`, `CALL`, `EVAL`, `STORE`).
- **Memory-first** – fetch & learn rules from mem0, minimise hard-coded logic.
- **Self-introspective** – runtime trace, hot-reload, guarded self-modification.
- **Auditability & safety** – every step logged; SafetyGuard gate for risky patches.

---

## 1. Milestone Checklist

### M1 – Primitive Extraction  

✅ Define and unit-test minimal primitives (≤15 LOC each).  
✅ Create registry for hot-swappable primitives.

### M2 – Graph-Based Planner  

✅ Represent templates as YAML/JSON logic graphs.  
✅ Implement graph loader → node objects.  
✅ Replace old recursion with graph traversal executor.

### M3 – Memory-Augmented Rules  

🟡 On plan request, search mem0 for matching graph (fallback → local template).  
⬜ Store generated graphs back to mem0 with usage metrics.

### M4 – Introspection & Hot-Reload  

⬜ Expose `GET_LOGIC_GRAPH` / `PATCH_GRAPH` via Introspector.  
⬜ Add CLI (or REST) helper for live editing.

### M5 – Trace & Explainability  

⬜ Emit `logic.trace` events (node, primitive, in/out, duration).  
⬜ Implement `explain(goal_id)` to reconstruct reasoning chain.

### M6 – Safety & Rollback  

⬜ Sandbox test every graph patch (unit tests + mypy).  
⬜ Version graphs in mem0; enable 1-click rollback.

### M7 – Stress & Regression  

⬜ Build goal corpus (simple → multi-domain).  
⬜ Fuzz primitives for edge cases.  
⬜ Keep core executor ≤400 LOC; refactor if exceeded.

---

## 2. Stretch / SOTA Enhancements

| Idea | Benefit | Effort |
|------|---------|--------|
| **LLM-Assisted Graph Synthesis** – auto-draft new logic graphs from natural-language goals. | Rapid expansion of planner capability without human templates. | ★★★ |
| **Dynamic Tool Selection** via function-calling LLM during `CALL` primitive. | On-the-fly integration with external APIs/tools. | ★★ |
| **MCTS / Planning-as-Search** overlay for long-horizon goals. | Better global optimisation vs simple traversal. | ★★★★ |
| **Reinforcement Learning of Primitive Weights** (success-rate-based). | Adaptive prioritisation of alternative sub-plans. | ★★★ |
| **Neural Symbolic Hybrid** – embed small LM to decide `COND` branches when rule unclear. | Graceful handling of ambiguity while retaining symbolic trace. | ★★ |
| **Multi-Agent Plan Negotiation** – share graphs between SubAgents and converge via voting. | Collective intelligence, task parallelism. | ★★★ |

---

## 3. Definition of Done

- All M1-M7 tasks checked.  
- Core executor ≤ **400 LOC**.  
- 95 % unit-test coverage for primitives & loader.  
- Average plan generation < **50 ms**.  
- New rule integration ≤ **5 min, zero code changes**.  
- Trace/explain outputs human-readable and passes audit review.

---

> Keep it **simple, observable, and editable** – today’s clarity enables tomorrow’s super-intelligence.
