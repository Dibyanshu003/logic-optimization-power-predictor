# Logic Optimization Power Predictor

**One-line summary:** Predict **logic-optimization power (QoR)** from a **netlist graph** and an **ABC recipe** using a **GCN + CNN** model.  
> We operate at the AIG/netlist level (ABC).

---

## Monte-Carlo (MC) Data Generation 
1) **Sample a 20-op recipe** from a fixed **13-op set**  
   `{refactor, rewrite, resub, balance, rfz, rwz, rsz, resyn2, st, share, c2rs, strash, drwsat2}`  
   (we typically set `op0 = strash` for stability).
2) **Apply to the design** in ABC, then run `ps -p`.
3) **Parse labels:** `AND`, `LEV`, `POWER` and compute `AND×LEV = AND*LEV`.
4) **Record a row:** `op0..op19, AND, LEV, POWER, AND×LEV`.  
(Repeat many times to create a diverse pretraining set; optionally mutate previously good sequences for explore+exploit.)

---

## Full Process (What We Did)

### 1) Problem Framing
- **Goal:** Quickly estimate **post-optimization power** (a QoR axis) for a given **netlist** and **recipe** without running full flows each time.
- **Inputs:**  
  - **Netlist** as an **AIG** (e.g., `.bench`) — post-RTL, pre-mapping representation.  
  - **Recipe**: a sequence of **20 ABC operations** chosen from a **13-op vocabulary**.
- **Output:** A scalar **POWER** prediction (mW scale) corresponding to ABC’s `ps -p`.

### 2) Operation Vocabulary (13 Ops, Fixed IDs)
We align Monte-Carlo and fine-tuning to the same mapping.

### 3) Monte-Carlo Pretraining Data
- **Sampling:** Generate many 20-op recipes. We usually fix `op0 = strash`, then sample the remaining 19 ops uniformly.  
- **Explore + Exploit (optional):** Keep a small memory of “good” sequences and **mutate** one position with some probability to search promising regions.
- **Labeling:** For each recipe, apply it in ABC on the target design and record `AND`, `LEV`, `POWER` from `ps - p`. Derive `AND×LEV`.
- **Dataset format:**  
  `op0,op1,…,op19, AND, LEV, POWER, AND×LEV`.

### 4) Model: GCN + CNN (Approach 2)
- **Graph branch (GCN):**  
  - Input: AIG as a graph (nodes = gates, edges = connections).  
  - Layers: `GCNConv → ReLU → GCNConv → ReLU → global mean pool` → **64-d** netlist embedding.
- **Sequence branch (CNN):**  
  - Input: 20 op IDs → **embedding (dim 16)** → **1D Conv (32, k=3)** → **adaptive max-pool** → **32-d** recipe embedding.
- **Fusion head:** concat(64, 32) → `FC(96→64, ReLU) → Linear(64→1)` → **POWER**.
- **Loss:** MSE.

### 5) Zero-Shot Pretraining
- **Data:** MC dataset across one or more base designs.  
- **Objective:** Teach the model general relationships between (netlist, recipe) and POWER.  
- **Outcome:** A pretrained checkpoint capturing broad patterns.

### 6) Few-Shot Fine-Tuning (New Design)
- **Select K recipes** (e.g., via clustering recipe embeddings to pick diverse representatives; typical K ≈ 50–250).  
- **Label K recipes** with ABC (`ps - p`) on the new design → `op0..op19, power`.  
- **Fine-tune** the pretrained model on this small set (lower LR, fewer epochs).  
- **Result:** The model adapts to design-specific characteristics with minimal labeling cost.

### 7) Inference / Use
- For any new recipe on that design, feed **(netlist graph, 20-op recipe IDs)** → get **POWER** prediction instantly, enabling rapid recipe ranking before running expensive tool flows.

---

## Results (from our experiments)
- **Small (max_orig):** ~**5%** error  
- **Medium (iir_orig):** ~**1.1%** error  
- **Large (aes_orig):** ~**4%** error  
*(Measured against ABC `ps - p`; numbers summarize the core outcome of Approach 2.)*

---

## Notes & Clarifications
- **Scope:** We predict **logic-optimization power** (pre-mapping, ABC AIG level), not sign-off post-P&R power.   
- **Consistency:** The **13-op mapping** must be identical in **Monte-Carlo generation** and **fine-tuning**.  
- **Why it works:** MC pretraining gives diverse coverage; few-shot fine-tuning specializes the model to a new design with very few fresh labels.

---

## Acknowledgments
- Course context: **CS533 — Machine Learning for EDA (IIT Guwahati)**  
- Tools: **ABC** for ground truth (`ps - p`), **PyTorch / PyTorch Geometric** for modeling.

