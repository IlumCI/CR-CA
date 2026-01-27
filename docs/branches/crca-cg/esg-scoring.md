# ESG Scoring

ESG (Environmental, Social, and Governance) scoring system for corporate evaluation.

## Overview

ESG scoring evaluates companies based on environmental, social, and governance factors.

## Mathematical Foundation

ESG score is computed as:

$$\text{ESG} = w_E \cdot S_E + w_S \cdot S_S + w_G \cdot S_G$$

Where:
- $S_E, S_S, S_G$: Environmental, Social, Governance sub-scores
- $w_E, w_S, w_G$: Weights (typically $w_E + w_S + w_G = 1$)

## Usage

```python
from branches.crca_cg.corposwarm import ESGScorer

scorer = ESGScorer()
score = scorer.compute_esg_score(company_data)
```

## Next Steps

- [Multi-Agent](multi-agent.md) - Multi-agent orchestration
