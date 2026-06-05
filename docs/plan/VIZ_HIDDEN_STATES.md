# 3D Hidden State Path Visualization

## Core question to answer visually

During AR generation (given warmup tokens), how does the model's hidden state
"navigate" through the memory encoded in the KV slots?

Specifically: the KV slots form a manifold in hidden space (one point per source position).
The Y tokens trace a path through or near that manifold as they decode.
This path should explain the phase-shift seen in extrapolation.

---

## Views to build (in order of interest)

### View 1 — Slot Manifold + Y Trajectory (main)

**What it shows**: where the slot representations live in hidden space,
and where the Y token attends during generation.

**Data**:
- Run one forward pass on a structured sequence (e.g. `up_counter`, seg=128).
- Collect hidden states of all N slot tokens at the final transformer layer: `h_slot[i]` (N × d).
- Collect hidden states of all Y tokens at the final layer: `h_Y[k]` (seg_len × d).
- Collect attention weights from each Y[k] to each slot[i]: `attn[k, i]` (seg_len × N).

**Reduce to 3D**:
- Stack all (N + seg_len) hidden states.
- Fit PCA on combined matrix → take top 3 components.
- Project slots → N points; project Y tokens → seg_len points.

**Plot**:
- **Slot manifold**: N points colored by source position (0=blue → N=red).
  For `up_counter`, these should form a smooth curve in hidden space.
- **Y trajectory**: seg_len points connected by lines, colored by generation step.
  Shows the "reading path" through the slot manifold.
- **Attention arrows** (sparse): for each Y[k], draw a vector from Y[k]'s position
  toward the centroid of its top-3 attended slots. Shows where the model is "looking."
- **Warmup marker**: first point of Y trajectory (warmup token) highlighted differently.

**What to look for**:
- Does the Y trajectory follow the slot manifold (recall model reads slots in order)?
- At extrapolation: does the Y path leave the manifold? In what direction?
- Phase shift: does the warmup token project onto the "wrong" part of the slot manifold?

---

### View 2 — Layer-by-Layer Token Refinement

**What it shows**: how a single token's representation changes as it passes through layers.

**Data**: same forward pass; collect hidden states of Y[0] (warmup token)
and Y[-1] (last generated token) at each layer 0..L.

**Plot**: two 3D paths, one per token. Each node = one layer.
Color = layer index (early=cool, late=warm).

**What to look for**:
- Does the warmup token's representation converge to a stable point by the final layer?
- Does the last generated token follow a similar trajectory shape?
- For the phase-shifted extrapolation case: does the warmup path end up near
  the beginning of the slot manifold instead of near the end?

---

### View 3 — Multi-sequence Slot Manifold Comparison

**What it shows**: how different source sequences form different manifold shapes,
and whether the manifolds share common structure.

**Data**: run 4 sequences through the encoder:
- `up_counter` (arithmetic step=1)
- `geometric` (multiplicative ratio)
- `palindrome` (symmetric)
- random uniform (unstructured)

**Plot**: 4 manifold curves in the same 3D space (after joint PCA).
Color = sequence type. Thickness = local curvature (how much the manifold bends).

**What to look for**:
- Do structured sequences form smooth, low-curvature curves?
- Does random data form a chaotic cloud?
- Do `up_counter` and `down_counter` form parallel / anti-parallel curves?
- Is the manifold geometry learned (i.e., structured sequences cluster away from random)?

---

### View 4 — Attention Weight Heatmap Surface (3D)

**What it shows**: the full attention routing from Y to slots as a 3D surface.

**Data**: `attn[k, i]` — attention weight from Y token k to slot i.
Shape: (seg_len, N). This is a matrix; render as a 3D surface.

**Plot**: surface where X=Y-step, Y=slot-index, Z=attention-weight.
Color = Z (attention intensity).

**What to look for**:
- Diagonal ridge = model reads slot i when generating Y[i] (perfect positional routing).
- Off-diagonal mass = model is not reading in order.
- For extrapolation warmup: does the ridge shift (phase offset visible as diagonal displacement)?

---

## Implementation plan

### Step 1 — Hook hidden states during forward pass

Modify `KVMemModel.__call__` or wrap it to return intermediate activations.
No architecture change; just collect outputs at each block:

```python
def forward_with_activations(model, tokens, mask):
    x = jax.vmap(model.embed)(tokens)
    layer_acts = [x]
    for block in model.blocks:
        x = block(x, mask)
        layer_acts.append(x)
    x = jax.vmap(model.norm_out)(x)
    logits = x @ model.W_out.T
    return logits, layer_acts   # layer_acts: list of (L, d) arrays
```

### Step 2 — Hook attention weights

Modify `MHAttention.__call__` to optionally return `attn` before the softmax dropout:

```python
def __call__(self, x, mask, return_attn=False):
    ...
    attn_weights = jax.nn.softmax(attn, axis=-1)   # (H, L, L)
    if return_attn:
        return out @ self.W_O.T, attn_weights
    return out @ self.W_O.T
```

Average over heads or keep per-head for richer analysis.

### Step 3 — Data collection script

```python
# kvmem/collect_activations.py
def collect(model, x_S, N, slot_style, seq_name):
    """Run one forward pass, return slot and Y hidden states + attention."""
    # Build full sequence [x_S | <m> | slots | </m> | x_S]
    # Run forward_with_activations
    # Extract slot positions: [seg_len+3 : seg_len+3+N]
    # Extract Y positions:    [seg_len+7+N : ]
    # Return dict with layer_acts, attn_weights, seq_name
```

### Step 4 — Dimensionality reduction

```python
from sklearn.decomposition import PCA

def project_3d(slot_acts, y_acts):
    """Jointly project slot and Y hidden states to 3D."""
    combined = np.vstack([slot_acts, y_acts])   # (N+seg_len, d)
    pca = PCA(n_components=3)
    projected = pca.fit_transform(combined)
    slot_3d = projected[:N]
    y_3d    = projected[N:]
    return slot_3d, y_3d, pca.explained_variance_ratio_
```

Also try UMAP for non-linear structure (may reveal tighter manifolds).

### Step 5 — 3D plotting

Use **Plotly** for interactive rotation (renders in browser/notebook):

```python
import plotly.graph_objects as go

fig = go.Figure()

# Slot manifold
fig.add_trace(go.Scatter3d(
    x=slot_3d[:,0], y=slot_3d[:,1], z=slot_3d[:,2],
    mode='markers+lines',
    marker=dict(color=np.arange(N), colorscale='Viridis', size=4),
    name='KV slots'
))

# Y trajectory
fig.add_trace(go.Scatter3d(
    x=y_3d[:,0], y=y_3d[:,1], z=y_3d[:,2],
    mode='markers+lines',
    marker=dict(color=np.arange(seg_len), colorscale='Plasma', size=6),
    line=dict(color='red', width=2),
    name='Y generation path'
))

# Warmup marker
fig.add_trace(go.Scatter3d(
    x=[y_3d[0,0]], y=[y_3d[0,1]], z=[y_3d[0,2]],
    mode='markers',
    marker=dict(color='white', size=12, symbol='diamond'),
    name='warmup token'
))

fig.write_html('hidden_state_viz.html')
```

### Step 6 — Animation (bonus)

Use Plotly frames to animate the Y trajectory token-by-token, showing how
the model "walks" through the slot manifold during generation.

---

## Files to create

```
kvmem/
  collect_activations.py   — forward pass with activation hooks
  viz_hidden_states.py     — runs collection + PCA + Plotly rendering
notebooks/
  hidden_states.ipynb      — interactive exploration notebook
```

---

## Experiments to run with this tool

1. **Recall (up_counter, seg=128)**: slot manifold should be a smooth helix/curve.
   Y trajectory should trace it closely.

2. **Extrapolation phase-shift**: show that warmup from `x_S[-4:]` projects onto
   the BEGINNING of the slot manifold (not the end), explaining the phase error.

3. **Random source vs structured source**: compare manifold shapes to confirm
   the KV is a semantic encoder for structured sequences.

4. **NoPE vs YaRN**: show that without positional encoding, the slot manifold
   is a disordered cloud; with YaRN, it is organized.

5. **Generalization**: compare slot manifolds for seen (random) vs unseen (deterministic)
   sequences. Do they overlap, suggesting shared representation?
