---
language: en
license: apache-2.0
tags:
  - reasoning
  - react
  - causal-inference
  - tool-use
---

# CR-CA LRM (draft model card)

## Model summary
This release pairs a **ReAct + self‑critique reasoning kernel** with a full-finetuned model.
The system is designed to **orchestrate tools** and **enforce refusal‑first causal outputs**.

**Base model:** Qwen2.5-7B-Instruct  
**Finetune type:** full finetune (no LoRA adapters)  
**Context length:** 8k

## Intended use
- Planning and reasoning over structured causal tasks.
- Tool‑augmented reasoning with strict gating.

## Limitations
- Numeric causal outputs are refused unless validated by `crca_core`.
- Reasoning traces are **non‑scientific** metadata and must not be treated as proofs.

## Training data
Hybrid dataset:
- Internal ReAct traces (reason/act/observe/critique) with refusal labels.
- Public reasoning datasets (see dataset builder config).

## Evaluation
Mixed evaluation:
- Refusal accuracy.
- ReAct cycle convergence.
- Causal identification correctness on synthetic SCMs.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "YOUR_HF_REPO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Reason step-by-step and propose a tool call."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

