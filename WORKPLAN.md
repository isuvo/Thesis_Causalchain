# Work Plan for Causal Vulnerability Detection

## Core Goals

- **GNN Backbone** — learn structural + semantic representations from PDG.
- **Attention-driven Causal Chains** — construct ordered vulnerability chains sink → root.
- **Root Cause Tracing** — explicitly link a vulnerability to its origin.
- **Cross-Class Propagation** — model should capture when vulnerabilities propagate across code classes/modules.
- **Probability Outputs** — all predictions and chain links should be probability-weighted for interpretability.

---

## Step 1. Data Preparation

**What to do:**
- Clean and normalize datasets (**DiverseVul** now, **BigVul** later).
- Build PDG graphs with **Joern** + slicing (nodes = statements, edges = data/control deps).
- Label: **vulnerable (1)** vs **safe (0)**.

**What to expect:**
- JSON graphs with line-contents, PDG edges, target.

---

## Step 2. Embedding & Graph Construction

**What to do:**
- Train **Word2Vec** embeddings on tokens.
- Convert each graph into **PyTorch Geometric** `Data(x, edge_index, y)`.
  - `x`: average word vectors per statement.
  - `edge_index`: control/data dependencies.

**What to expect:**
- Node embeddings + graph structures ready for GNN input.

---

## Step 3. Model (Core Deliverables)

**What to do:**
- Build a GNN encoder (**GCN layers**).
- Add **attention module**: PDG-masked self-attention computes importance between connected nodes.
- Add **probability head**:
  - Graph-level **softmax** → \(P(\text{vulnerable})\) vs \(P(\text{safe})\).
  - Node-level **attention weights** → probability a node contributes to a vulnerability.

**What to expect:**
- **Model outputs:**
  - Vulnerability probability for the graph.
  - Attention matrix (probabilities per edge/node).
  - Contextualized node embeddings for tracing.

---

## Step 4. Training

**What to do:**
- Train with **weighted cross-entropy** to handle imbalance.
- Use probability outputs from **softmax** to monitor \(P_{\text{vul}}\).
- Early stopping + checkpoints.

**What to expect:**
- Best model saved with stable accuracy.
- Probability distribution over **vulnerable** vs **safe** graphs.

---

## Step 5. Causal Chain Construction

**What to do:**
- At inference, identify the **sink node** (highest probability of vulnerability).
- **Backtrack** along attention edges (probability-weighted) to build a causal chain.
- **Export**: sink → root chain with probability scores at each hop.

**What to expect:**
- Human-readable causal maps like:

```
[Sink: strcpy()] P=0.92
   ← [length calc missing] P=0.78
   ← [input read] P=0.65
```

---

## Step 6. Cross-Class Propagation (Multi-Class in Thesis Sense)

**What to do:**
- Extend model to keep track of **code class/module IDs** for each node.
- Ensure attention traces can **span across classes**.
- When vulnerability originates in **Class A** and flows into **Class B**, output a **propagation map**.

**What to expect:**
- Chains showing propagation across modules, e.g.:
```
ClassA: read_input() → ClassB: process_data() → sink: memcpy()
```
- Each hop **weighted with probability**.

---

## Step 7. Testing & Demonstration

**What to do:**
- Run model on **test set**.
- Export predicted **vulnerability probability** + **causal chains**.
- Collect examples for **thesis appendix**.

**What to expect:**
- JSON/CSV outputs:
```json
{
  "function": "foo",
  "P_vulnerable": 0.91,
  "chain": [
    {"line": "user_input()", "P": 0.66, "class": "InputHandler"},
    {"line": "len_check()", "P": 0.74, "class": "Validator"},
    {"line": "strcpy()", "P": 0.92, "class": "Processor"}
  ]
}
```

---

## Notes

**Probability is central:**
- Graph-level classification uses **softmax**.
- Node-to-node contributions use **attention probabilities**.
- Chains are built by following the **highest probability paths** backward to the root.

> **Evaluation metrics (CCS, CFAM) → next stage.**
