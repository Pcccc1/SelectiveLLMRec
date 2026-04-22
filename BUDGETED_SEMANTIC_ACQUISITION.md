# Budgeted Semantic Acquisition for Graph Recommendation

## 1. Core Framing

The method should not be framed as "calling the LLM for fewer nodes to save cost."
That framing is too weak.

The stronger formulation is:

> Under a limited semantic acquisition budget, how can we allocate expensive semantic supervision to the nodes that will yield the largest recommendation benefit?

In this framing:

- LLM is an expensive semantic acquisition engine, not the main model.
- Routing is a budget allocation module, not a heuristic filter.
- Fusion is a conservative injection mechanism, not semantic replacement.


## 2. Recommended Scope

The first workable version should only enhance the **item side**.

Reasons:

- Item text is more stable than user text.
- Item-side semantic acquisition is cacheable.
- Item routing is easier to justify.
- User-side summarization and clustering add too much noise and complexity.


## 3. Method Overview

The method should be organized as a 3-stage framework:

1. **Collaborative Backbone Pretraining**
2. **Budgeted Semantic Acquisition**
3. **Conservative Semantic Injection**

High-level pipeline:

```text
Interaction Graph
    -> LightGCN pretraining
    -> item value estimation under budget
    -> select top-B items
    -> acquire expensive semantics only for selected items
    -> semantic projection + conservative fusion
    -> ranking fine-tuning
```


## 4. Stage A: Collaborative Backbone

Train a pure collaborative filtering backbone first, e.g. LightGCN.

Outputs:

- user structural embeddings
- item structural embeddings

Purpose:

- provide the base recommender
- provide routing signals for semantic budget allocation
- provide a stable embedding space for later semantic alignment

This stage should remain close to the current `pretrain.py` pipeline.


## 5. Stage B: Budgeted Semantic Acquisition

### 5.1 Budget Definition

Let:

- `I` be the item set
- `rho in (0, 1]` be the semantic acquisition budget ratio
- `B = floor(rho * |I|)` be the number of items allowed to use expensive semantics

Only `B` items are allowed to receive LLM-based semantic acquisition.


### 5.2 Item Value Estimation

Do not define routing as "importance."
Define it as **expected marginal utility of semantic acquisition**.

Recommended item value score:

```text
v_i = a * U_i + b * D_i + c * E_i - d * H_i
```

Where:

- `U_i`: structural uncertainty
- `D_i`: graph-semantic discrepancy proxy
- `E_i`: exposure or recommendation impact potential
- `H_i`: redundancy or head-item penalty

Interpretation:

- `U_i` identifies items whose structural embeddings are unstable or underdetermined
- `D_i` identifies items where collaborative representation may miss semantic information
- `E_i` ensures selected items are likely to affect ranking quality
- `H_i` avoids wasting budget on items that are already too easy or too dominant

### 5.3 Practical Signals

Based on the current project, the following can be reused or extended:

- propagation-layer embedding variance or drift
- gap between ID embedding and final GNN embedding
- degree / PageRank / exposure frequency
- a penalty for overly popular items

The current `Node_value_Evaluator` can be treated as a prototype, but it should be reinterpreted as a **budget utility estimator**, not just a node-importance scorer.


## 6. Stage C: Selective Semantic Acquisition

For the selected item set `S_B`:

- acquire expensive semantic annotations only for these items
- cache the results
- encode them into semantic embeddings

Important design rule:

The LLM should not be presented as a free-form text generator whose output is later blindly fused.
It should be presented as a **sparse semantic supervision provider**.

Recommended output style:

- short semantic summary
- structured tags
- category-style descriptors
- target preference descriptors

Avoid long, open-ended generated paragraphs whenever possible.

Preferred acquisition flow:

```text
raw item profile
    -> structured / concise LLM semantic annotation
    -> semantic encoder
    -> semantic embedding s_i
```

This is better than:

```text
raw text
    -> long generated description
    -> embedding model
    -> fusion
```

because the latter behaves too much like expensive paraphrasing.


## 7. Conservative Semantic Injection

This is the most important modeling constraint.

For non-selected items, the model should strictly fall back to the original GNN embedding:

```text
z_i = g_i
```

For selected items:

```text
s_i_proj = P(s_i)
delta_i = MLP([g_i ; s_i_proj])
alpha_i = sigmoid(w^T [g_i ; s_i_proj])
z_i = g_i + m_i * alpha_i * delta_i
```

Where:

- `g_i`: GNN item embedding
- `s_i`: acquired semantic embedding
- `P(.)`: semantic projector into the GNN space
- `m_i in {0,1}`: budget mask
- `alpha_i`: adaptive fusion strength

### Design Requirements

- non-selected items must not be semantically perturbed
- selected items should receive only residual corrections
- the semantic branch must not overwrite collaborative signal
- fusion should start near the identity mapping

### Initialization Recommendations

- initialize the last fusion layer close to zero
- initialize fusion strength to a small value
- treat semantic injection as a small correction, not a replacement


## 8. Learning Objective

Do not rely on ranking loss alone.

Recommended objective:

```text
L = L_rank + lambda_1 * L_align + lambda_2 * L_cons + lambda_3 * L_reg
```

Where:

- `L_rank`: pairwise ranking loss such as BPR
- `L_align`: semantic-graph alignment loss on selected items
- `L_cons`: consistency regularization to keep fused embeddings close to original GNN embeddings
- `L_reg`: parameter regularization

### Recommended Forms

`L_rank`

- BPR is acceptable and aligns with the current project

`L_align`

- compute only on selected items
- use cosine loss, MSE, or InfoNCE
- align projected semantic embeddings with structural embeddings

`L_cons`

- constrain fusion magnitude, e.g. `||z_i - g_i||^2`
- prevents semantic branch from destroying the pretrained collaborative space


## 9. Training Strategy

Do not train everything end-to-end from scratch.

Recommended training schedule:

### Stage A

- pretrain LightGCN backbone

### Stage B

- compute item value scores offline
- select top-B items under budget
- run semantic acquisition offline
- encode and cache semantic embeddings

### Stage C

- initialize the semantic projector and fusion module
- freeze backbone at first
- train projector / gate / fusion module
- then unfreeze backbone with a small learning rate for joint fine-tuning

This schedule is much more stable than direct end-to-end fusion.


## 10. Inference

At inference time:

- user embeddings come from the collaborative backbone
- selected items use fused embeddings
- non-selected items use original GNN embeddings
- full ranking is then computed as usual

This preserves the budgeted nature of the method:

- expensive semantic acquisition is sparse
- semantic overhead is precomputed and bounded


## 11. How This Maps to the Current Project

### Reusable Components

- `pretrain.py` as the collaborative backbone stage
- `utils/item_node_value_evaluation.py` as the starting point for item utility scoring
- existing item profiles and item semantic embedding files as provisional semantic inputs
- `LightGCN_retrain` as the second-stage training entry point

### Components That Must Be Changed

- remove user-side semantic fusion from the first version
- route only items
- make fusion strictly mask-controlled
- ensure non-selected items remain unchanged
- stop truncating semantic embeddings
- add explicit alignment and consistency losses
- make the semantic acquisition output more structured and concise


## 12. What the Contribution Should Be

The contribution should not be written as:

- "we reduce LLM calls"
- "we select important nodes"
- "we fuse LLM embeddings into GNN embeddings"

The stronger contribution statement is:

1. We formulate recommendation enhancement under a limited semantic acquisition budget.
2. We propose a utility-based routing mechanism to allocate expensive semantic supervision only to the most beneficial items.
3. We introduce a conservative semantic injection strategy that preserves collaborative structure while absorbing sparse semantic signals.


## 13. Minimal Viable Version

The first implementable version should be:

- item-only
- budget ratios: `1%`, `5%`, `10%`, `20%`
- offline selection of top-B items
- structured semantic acquisition for selected items only
- semantic projector into GNN space
- mask-controlled residual fusion
- BPR + alignment + consistency training objective

This is the most realistic path to a stable and defensible method.


## 14. One-Sentence Summary

> We study graph recommendation under a limited semantic acquisition budget, where expensive LLM-derived semantics are selectively allocated to the items with the highest expected marginal utility, and then injected into a collaborative backbone through conservative, alignment-guided residual fusion.
