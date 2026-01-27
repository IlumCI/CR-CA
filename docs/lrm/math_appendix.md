---
title: LRM Math Appendix
---

# LRM Math Appendix (Pearl + ML/CS)

## 1. Structural Causal Models (SCM)
Let \(V=\{V_1,\dots,V_n\}\) be endogenous variables and \(U=\{U_1,\dots,U_n\}\) exogenous noise.
An SCM is defined by structural equations:
\[
V_i \leftarrow f_i(\mathrm{Pa}(V_i), U_i).
\]
If the SCM is **Markovian**, then the joint distribution factorizes:
\[
P(V) = \prod_i P(V_i \mid \mathrm{Pa}(V_i)).
\]

## 2. Interventions and do-calculus
The intervention \(do(X=x)\) replaces the structural equation for \(X\) with a constant \(x\).
The interventional distribution is \(P(Y \mid do(X=x))\).
The do-calculus rules (Pearl, 2009) allow transformation between interventional and observational expressions
under graphical conditions (d-separation in manipulated graphs).

## 3. Counterfactuals (Abduction–Action–Prediction)
Given evidence \(E=e\), the counterfactual \(Y_x\) is computed by:
1. **Abduction**: infer \(U\) from \(E=e\).
2. **Action**: replace equations with intervention \(do(X=x)\).
3. **Prediction**: forward-simulate \(Y\) using inferred \(U\).

## 4. Identification (Pearl + Shpitser–Pearl)

### 4.1 Backdoor criterion
**Definition**: A set \(Z\) satisfies backdoor w.r.t. \((X,Y)\) if it blocks all backdoor paths from \(X\) to \(Y\) in \(G_{\bar{X}}\).

**Estimand**:
\[
P(Y\mid do(X)) = \sum_Z P(Y\mid X,Z)P(Z).
\]

**Proof sketch**: Apply do-calculus Rule 2 on \(G_{\bar{X}}\) where \(Z\) d-separates \(X\) and \(Y\).

### 4.2 Frontdoor criterion
**Definition**: \(M\) is a frontdoor mediator if:
1. All directed paths \(X\to Y\) go through \(M\);
2. No backdoor path from \(X\) to \(M\);
3. All backdoor paths from \(M\) to \(Y\) are blocked by \(X\).

**Estimand**:
\[
P(Y\mid do(X)) = \sum_M P(M\mid X)\sum_{X'} P(Y\mid M,X')P(X').
\]

**Proof sketch**: Use Rule 3 to exchange intervention with observation for the \(M\to Y\) part and Rule 2 for the \(X\to M\) part.

### 4.3 Instrumental Variables (Linear)
Assume linear SCM:
\[
X = \pi Z + U_X,\quad Y = \beta X + U_Y,
\]
with exclusion \(Z \not\to Y\) except through \(X\), independence \(Z \perp U_Y\), and relevance \(\pi \neq 0\).

Then:
\[
\beta_{IV} = \frac{\mathrm{Cov}(Z,Y)}{\mathrm{Cov}(Z,X)}.
\]

### 4.4 ID Algorithm (Shpitser–Pearl)
The ID algorithm recursively identifies \(P(Y\mid do(X))\) in semi-Markovian graphs.
Key steps:
1. If \(Y\subseteq X\), return 1.
2. If \(X=\emptyset\), return \(P(Y)\).
3. Decompose into C-components and recurse.

Correctness and completeness: Shpitser & Pearl (JMLR 2008).

## 5. Linear-Gaussian SCM Inference
For linear-Gaussian SCM:
\[
V = B^T V + U,\quad U\sim \mathcal{N}(0,\Sigma).
\]
Let \(A = I - B^T\). Then:
\[
U = AV,\quad V = A^{-1}U,\quad \Sigma_V = A^{-1}\Sigma A^{-T}.
\]

**Partial observation**: If \(V = (V_o, V_m)\) with observed \(V_o\), then
\[
\mathbb{E}[V_m\mid V_o] = \Sigma_{mo}\Sigma_{oo}^{-1}V_o.
\]
Thus:
\[
\mathbb{E}[U\mid V_o] = A\mathbb{E}[V\mid V_o].
\]

## 6. Discovery algorithms (assumptions)
### PC
Correct under causal sufficiency + faithfulness; returns CPDAG.

### FCI
Handles latent confounding; returns PAG; correctness relies on ancestral graph Markov properties.

### GES
Score-based search; consistent under correct score + large sample assumptions.

### PCMCI
Time-lagged conditional independence; requires stationarity and lag sufficiency.

## 7. ReAct with critique (formal)
Define a gating function \(g(a)\in\{\text{allow},\text{refuse}\}\) for actions.
Define critique score:
\[
C_t = \sum_i w_i \phi_i(\text{missing prereqs}, \text{refusal risk}, \text{spec mismatch}).
\]
Policy: if \(C_t\ge \tau\), revise; else finalize.

## References
- Pearl, *Causality* (2009)
- Shpitser & Pearl (JMLR 2008)
- Peters, Janzing, Schölkopf (2017)
- Runge et al. (2019) PCMCI
