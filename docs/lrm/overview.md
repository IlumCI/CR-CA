---
title: LRM Overview
---

# Large Reasoning Model (LRM)

CR-CA’s LRM mode is a **ReAct + self‑critique** reasoning architecture that:

- Runs **multi‑cycle Reason → Act → Observe** loops.
- Performs **end‑of‑cycle critique** to revise plans.
- Enforces **strict tool gating** (no numeric causal outputs unless crca_core approves).
- Writes to structured memory **only on finalize**.

## What it is
- A **reasoning kernel** that orchestrates tools and structured memory.
- A **training pipeline** that finetunes a small base model on ReAct traces.

## Full finetune (Qwen2.5-7B-Instruct)
- See `docs/lrm/finetune_full.md` for the 16GB GPU setup, dataset build steps, and export workflows.

## What it is not
- Not an LLM that can output causal numbers without locked specs.
- Not a replacement for `crca_core` scientific validation.

## Key modules
- `crca_reasoning/godclass.py` (ReActGodClass)
- `crca_reasoning/react_controller.py`
- `crca_reasoning/critique.py`
- `crca_reasoning/rationale.py`
- `crca_reasoning/memory.py`
- `crca_reasoning/tool_router.py`

