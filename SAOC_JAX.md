# Structure-Aware Optimizer Compiler — Implementation Handover

**A certificate-driven, e-graph-extracted optimizer that *synthesizes* per-block update
rules over an operator algebra, built on top of JAX.**

> One-line thesis: don't `switch` over hand-written optimizers; *derive* them.
> Read the loss's curvature, propagate it as a second tape alongside the cotangent,
> express the exact second-order step in a small operator algebra, then let
> approximate equality-saturation under a hardware cost model extract the cheapest
> *sound* update. K-FAC / NGD / Muon / Adam become **extracted terms**, not branches.

---

## Table of contents

1. [Motivation and scope](#1-motivation-and-scope)
2. [Theory](#2-theory)
   1. [The certificate: a second tape](#21-the-certificate-a-second-tape)
   2. [The abstract domain $\mathcal{G}$](#22-the-abstract-domain-mathcalg)
   3. [The operator algebra](#23-the-operator-algebra)
   4. [Discovery by equality saturation](#24-discovery-by-equality-saturation)
   5. [The generalization axes](#25-the-generalization-axes)
3. [System architecture on JAX](#3-system-architecture-on-jax)
4. [Transfer functions](#4-transfer-functions)
5. [Worked derivation: 2-layer ReLU MLP](#5-worked-derivation-2-layer-relu-mlp)
6. [Testing plan and toy examples](#6-testing-plan-and-toy-examples)
7. [Architecture coverage](#7-architecture-coverage)
8. [Extension catalog (the axes as an API)](#8-extension-catalog-the-axes-as-an-api)
9. [Implementation roadmap](#9-implementation-roadmap)
10. [Open problems and honest limitations](#10-open-problems-and-honest-limitations)
11. [References](#11-references)

---

## 1. Motivation and scope

Modern optimizers that beat SGD/Adam — K-FAC, NGD, Shampoo, SOAP, Muon — all
precondition the gradient by an (approximation of an) inverse curvature operator.
Today a human authors both the curvature approximation and the inverse-application
strategy. **That authorship is the entire content of optimizer design, and it lives in
the human's head.** This project moves it into a compiler.

The system has three layers that JAX does not provide and that we build:

- a **certificate** — an abstract interpretation over the computation graph whose
  abstract domain is *optimization-relevant* (curvature class, Jacobian structure,
  separability, Lipschitz/strong-convexity, symmetry);
- an **operator algebra** — a small combinator IR over the matrix-free
  pushforward/pullback/curvature programs JAX already produces;
- an **approximate equality-saturation** layer that searches this algebra under a
  hardware cost model, guarded by the certificate, and extracts a runnable update.

Everything load-bearing and hard (AD, higher-order AD, transpose, `jit`, XLA codegen)
is **reused from JAX**. See §3.

Scope of this document: enough theory to implement, the JAX module layout and APIs,
the transfer-function tables, the validation suite (toy examples with expected
results), and per-architecture coverage for traditional DL (MLP, CNN, LSTM/GRU) and
modern AI (Transformer/ViT, RoPE, SSM/Mamba, DeltaNet/gated DeltaNet, TTT).

---

## 2. Theory

### 2.1 The certificate: a second tape

Reverse-mode AD seeds the loss with cotangent $1$ and propagates cotangents
$\bar v$ backward. We add a **second tape** that propagates an abstract element
$g \in \mathcal{G}$ summarizing the geometry of the map
$\theta_b \mapsto L$ for each parameter block $b$. The backward message becomes the
pair $(\bar v, g)$.

Frame it as a **backward abstract interpretation** seeded at the loss, structurally
parallel to reverse-mode AD: at each node we summarize the composite map
$(\text{downstream}\circ\text{this block}) \to L$. For each primitive
$y = \mathrm{op}(x)$ with downstream certificate $g_y$ we define a transfer function
$g_x = T_{\mathrm{op}}(g_y)$ such that concrete chain-rule composition corresponds to a
**sound** lattice operation on $\mathcal{G}$.

### 2.2 The abstract domain $\mathcal{G}$

A reduced product (à la Cousot) of factors, each a lattice; $\top$ means "know
nothing, fall back to AdamW".

$$
\mathcal{G} \;=\; \mathcal{C} \times \mathcal{S} \times \mathcal{P} \times (\mathcal{L},\mu) \times \mathcal{I}
$$

| Factor | Meaning | Lattice (informal) |
|---|---|---|
| $\mathcal{C}$ | curvature / smoothness class | $\textsf{LINEAR}\sqsubseteq\textsf{QUADRATIC}\sqsubseteq\textsf{CONVEX}\sqsubseteq\textsf{L-SMOOTH}\sqsubseteq\top$ |
| $\mathcal{S}$ | structure of the Jacobian-to-loss (carries **axis roles**) | $\textsf{DIAG}\sqsubseteq\{\textsf{KRON},\textsf{LOWRANK},\textsf{DIAG{-}RANK1}\}\sqsubseteq\textsf{DENSE}\sqsubseteq\top$ |
| $\mathcal{P}$ | separability / coupling | $\textsf{SEP}\sqsubseteq\textsf{BLOCK{-}SEP}\sqsubseteq\textsf{COUPLED}\sqsubseteq\top$ |
| $(\mathcal{L},\mu)$ | Lipschitz $L$, strong-convexity $\mu$ | quantitative; $\mu=0$ if not provable |
| $\mathcal{I}$ | symmetry / invariance / flat directions | set of subspaces (e.g. $\langle\mathbf 1\rangle$, radial) |

Soundness of heterogeneous per-block rules rests on $\mathcal{P}$: distinct rules on
distinct blocks are valid only up to certified separability; otherwise blocks are
merged and certificates **joined** (meet $\sqcap$ = least informative = safest).

### 2.3 The operator algebra

For a parameter block let $J = \partial o/\partial\theta$ (params $\to$ output). JAX
gives two composable, matrix-free operator-programs:

- **pullback** $J^\top$ (VJP), so the gradient is $g = J^\top\,\partial L/\partial o$;
- **pushforward** $J$ (JVP),

and they are transposes (the *You Only Linearize Once* relationship: linearize, then
transpose). With the loss-output curvature $H = \partial^2 L/\partial o^2$ the
**Gauss–Newton / Fisher operator is itself a program**:

$$
F \;=\; \text{pullback}\circ H\circ\text{pushforward}\;=\;J^\top H J ,
\qquad
F v \;=\; \text{vjp}\big(H\cdot\text{jvp}(v)\big)
$$

— one JVP + one VJP per matvec; $F$ is never materialized. The **seed update** is

$$
\Delta\theta = -\,\mathrm{solveCG}_k(F, g).
$$

The optimizer zoo is then a set of **terms** over a handful of combinators:

| Method | Term |
|---|---|
| Newton / exact GN | $-\,\mathrm{solveCG}_k(F, g)$ |
| NGD | $-\,F^{-1} g$ with $H=$ output Fisher |
| K-FAC | $F \approx A\otimes B,\;\; -\,(A^{-1}\otimes B^{-1})\,g$ |
| Shampoo / whitening | $-\,(L^{-1/4}\otimes R^{-1/4})\,g$ |
| Muon | $-\,\mathrm{NewtonSchulz}_k(MM^\top)^{-1/2} M$ (per matrix block) |
| Adam | $F\approx\mathrm{diag}(\mathrm{EMA}(g^2)),\;\; -\,\mathrm{EMA}(g)\oslash\sqrt{\mathrm{diag}}$ |
| SGD | $F\approx I,\;\; -\eta\, g$ |

These are not seven cases; they are a small grid:
$$
\{\text{curvature approx: } \textsf{exact}\mid\textsf{Kron}\mid\textsf{diag}\mid\textsf{spectral-2nd-moment}\mid\textsf{id}\}
\times
\{\text{inverse: } \textsf{CG}_k\mid\textsf{Kron-inv}\mid\textsf{pth-root}\mid\textsf{NewtonSchulz}_k\mid\textsf{elementwise}\}
\times
\{\text{power: } {-1}\mid{-\tfrac12}\}.
$$

### 2.4 Discovery by equality saturation

Seed an e-graph with the **exact** term $\Delta\theta=-F^{-1}g$, then add rewrites:

- **exact** algebraic rules: $(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}$, transpose laws,
  $\text{pullback}\circ\text{pushforward}$ fusions;
- **approximate-equality** rules, each tagged with an error bound supplied by the
  certificate: $J^\top H J \approx A\otimes B$, $F\approx\mathrm{diag}\,F$,
  $F^{-1}\approx\mathrm{CG}_k$, $F^{-1/2}\approx\mathrm{NewtonSchulz}_k$, low-rank
  truncation, damping.

Each approximate rewrite carries a **guard** read from the certificate (e.g. the
Kronecker rewrite fires only where $\mathcal{S}=\textsf{KRON}$ is provable).
**Extract** the term minimizing a hardware-aware cost (flops/memory/bits) traded
against accumulated approximation error. The `switch` disappears: it is replaced by
saturate-and-extract, with the certificate as the side-condition. K-FAC/NGD/Muon emerge
as cost-optimal extractions; novel hybrids are reachable.

This discovery is **amortized**: run it once per architecture as an *optimizer
compilation* pass (analogous to Ansor/Felix compiling a kernel); the extracted program
runs every step.

### 2.5 The generalization axes

The static algebra acts on primitives $\{g, Fv, Tv\}$ at one instant. Families beyond
Newton/NGD are reached by extending along orthogonal axes (full catalog in §8). The
first four:

1. **Source** — exact-AD $\mid$ temporal-EMA estimate $\mid$ finite-difference probe
   (covers zeroth-order, empirical Fisher).
2. **Time** — delay $z^{-1}$, EMA, rational filters
   (momentum $m=(1-\beta z^{-1})^{-1}g$; Adam = temporal diagonal whitening).
3. **Order** — iterate AD $k$ times for $k$-th order operators.
4. **Meta** — combinators $\textsf{Update}\to\textsf{Update}$, incl. AD-through-update
   (lookahead, hypergradients, iterate averaging).

Each axis is *vocabulary*; the extractor/cost/soundness machinery never changes. The
zoo is the **product** of the axes.

---

## 3. System architecture on JAX

**Build on JAX. Do not write a new autodiff backbone.** JAX's jaxpr is the operator IR
substrate; `linearize`/`vjp`/`transpose` produce the fragments; JAX's tracing is itself
an abstract interpreter, so the certificate is "one more interpretation of the jaxpr."
The only genuinely new components are the certificate interpreter and the
approximate-rewrite e-graph, both of which attach *beside* JAX.

### 3.1 Reuse vs build

| Need | Source |
|---|---|
| pushforward/pullback, matrix-free $F$, HVP/GNvp | **reuse** `jax.linearize`, `jax.vjp`, forward-over-reverse |
| higher-order AD, grad-through-update | **reuse** nested `jax.grad`/`jax.jvp` |
| state/time combinators, scan, zeroth-order probes | **reuse** functional state, `jax.lax.scan`, PRNG |
| execution / codegen | **reuse** emit JAX fn + `jax.jit` + XLA |
| certificate AI ($\mathcal C,\mathcal S,\mathcal P,\mathcal L,\mu,\mathcal I$) | **build**: custom jaxpr interpreter |
| approximate-rewrite e-graph + cost extraction | **build**: egglog (Python binding), runs beside JAX |

### 3.2 Two-IR pipeline

Do **not** rewrite jaxprs in place. Keep JAX's jaxpr (for AD + execution) and a small
typed **operator IR** (for analysis + rewriting), bridged by lift/lower.

```
1. user model      f(params, x) -> output  +  loss          # plain JAX/Flax/Equinox; NO optimizer written
2. JAX AD          linearize -> linear jaxpr (JVP);  transpose -> VJP;  H from loss node
3. lift + certify  walk (linearized) jaxpr -> operator IR;
                   run certificate interpreter; canonicalize + atom-match (softmax-CE, LayerNorm, conv)
4. seed term       Δθ = -solveCG(F, g),  F = pullback ∘ H ∘ pushforward     (in operator IR)
5. e-graph         egglog: seed + exact & approximate rewrites (cost+error tags)
                   + certificate facts as fire-guards;  saturate
6. extract         hardware cost model -> one operator term
7. lower           emit term as a JAX function (jvp/vjp/solve/kron/newton_schulz/ema primitives); jit it
```

The subtle step is **3**: JAX's AD hands you an *executable* HVP, but the *structural*
fact "$F_{W_2}=(rr^\top)\otimes H$" lives in the **shape of the linearized jaxpr**. So run
`jax.linearize` to obtain a linear jaxpr, then run the certificate analysis over *that*
program to read off the Kronecker/low-rank/diagonal structure. This is "analysis of
JAX's AD output," not new AD.

### 3.3 Module layout

```
soco/                              # "structure-aware optimizer compiler"
  ir/
    operator_ir.py                 # typed operator-algebra nodes (dataclasses)
    lift.py                        # jaxpr (+ linearized jaxpr) -> operator IR
    lower.py                       # extracted term -> jitted JAX function
    canonicalize.py                # eqsat-based normalization so atoms match
  certificate/
    domain.py                      # C, S (axis-roles), P, (L, mu), I; join/meet; ⊤
    interpret.py                   # custom jaxpr interpreter producing certificates
    transfer/                      # per-primitive transfer functions (see §4)
      unary.py  binary.py  reduce.py  shape.py
    atoms.py                       # softmax-CE, LayerNorm/RMSNorm, linear+bias, conv
  algebra/
    combinators.py                 # F, solveCG, kron, diag, newton_schulz, ema, prox, ...
    rewrites.py                    # exact + approximate egglog rules, with guards + cost
    cost.py                        # hardware cost model (flops/mem/bits) + error budget
    extract.py                     # saturate + cost-guided extraction
  runtime/
    optimizer.py                   # optax-compatible GradientTransformation built from a term
    state.py                       # per-block factor buffers, refresh schedule (K-FAC-style)
  dsl/
    register.py                    # user-facing combinator registration (op, cost, guard, rewrites)
  tests/                           # see §6
```

### 3.4 Key data structures (sketch)

```python
# certificate/domain.py
from dataclasses import dataclass
import enum, jax.numpy as jnp

class Curvature(enum.IntEnum):     # lattice order
    LINEAR=0; QUADRATIC=1; CONVEX=2; L_SMOOTH=3; TOP=4

class Struct(enum.Enum):
    DIAG="diag"; KRON="kron"; LOWRANK="lowrank"; DIAG_RANK1="diag_rank1"
    DENSE="dense"; TOP="top"

@dataclass(frozen=True)
class AxisRoles:                   # which axes are matrix-row/col/batch/contracted
    rows: tuple; cols: tuple; batch: tuple; contracted: tuple

@dataclass(frozen=True)
class Cert:
    C: Curvature
    S: Struct
    axes: AxisRoles
    P: str                         # "sep" | "block_sep" | "coupled" | "top"
    L: float                       # propagated Lipschitz bound
    mu: float                      # strong-convexity (0 if unprovable)
    I: tuple                       # flat subspaces, e.g. ("ones",), ("radial",)
    exact: bool = True             # whether S/H is exact or approximate (GN drop, K-FAC factorization)

    def join(self, o):  ...        # ⊔ : pointwise lattice join (used at fan-in)
    def meet(self, o):  ...        # ⊓ : safest shared rule (used on coupling)

TOP = Cert(Curvature.TOP, Struct.TOP, AxisRoles((),(),(),()), "top",
           L=jnp.inf, mu=0.0, I=())
```

```python
# ir/operator_ir.py  — the manipulable operator algebra (NOT jaxpr)
@dataclass(frozen=True)
class Op:                          # an operator-algebra term node
    kind: str                      # "F" | "g" | "kron" | "diag" | "inv" | "isqrt"
                                   # | "newton_schulz" | "ema" | "prox" | "proj" | "scan" | ...
    args: tuple                    # child Ops
    attrs: dict                    # power, k_iters, beta, guard refs, cost annotation
```

```python
# dsl/register.py  — how a user adds momentum / Adam / lookahead / probe / ...
@register_combinator
class EMA:
    state = Buffer(like="grad")
    beta: float = 0.9
    def op(self, g, s):                          # (a) JAX implementation
        s = self.beta * s + g
        return s, s
    cost = Cost(madds=1, buffers=1, bits=32)     # (b) for extraction
    def guard(self, cert) -> bool:               # (c) soundness/quality guard
        return True                              #     (always sound; tags bias/lag)
    rewrites = [                                 # (d) egglog rules to algebra objects
        "(approx (F) (ema (outer g)) :error (staleness beta))",
    ]
```

The runtime target is an `optax.GradientTransformation`: `init` allocates the per-block
factor/state buffers, `update` applies the lowered, jitted term and refreshes dynamic
factors on a schedule (static $\mathcal C/\mathcal S/\mathcal P/\mathcal I$ are computed
once; only numeric factors — $rr^\top$, $\mathrm{diag}(s)-ss^\top$, $L$ — refresh).

---

## 4. Transfer functions

Two layers: a **sound, total primitive layer** (every ONNX-ish op has a transfer
function; analysis defined on any graph, degrading to $\top$) and an **atom library**
(e-matched patterns whose tight certificate is hand-proven, recovering the
non-compositional facts). Target ~30 precise primitives + ~6 atoms; the long tail
returns $\top$.

### 4.1 Unary (elementwise) $y_i=\varphi(x_i)$

Diagonal Jacobian $\mathrm{diag}(\varphi'(x_i))$, so $\mathcal{S}=\textsf{DIAG}$,
$L\mathrel{*}= \sup|\varphi'|$, diagonal second-order term $\mathrm{diag}(\varphi'')$.
Track sign of $\varphi'$ with convexity (monotonicity is needed for DCP composition).

| $\varphi$ | $\sup|\varphi'|$ | curvature effect |
|---|---|---|
| ReLU | $1$ | CONVEX, nonsmooth, **gates** ($D=\mathrm{diag}(\mathbf 1[x>0])$) |
| softplus | $1$ | CONVEX + monotone $\Rightarrow$ preserves CONVEX |
| sigmoid | $1/4$ | smooth, bounded $\varphi''$, **destroys** CONVEX |
| tanh | $1$ | smooth, **destroys** CONVEX |
| GELU/SiLU | $\approx 1.1$ | smooth, **destroys** CONVEX |

### 4.2 Binary $z = \mathrm{op}(x, y)$ — **per-operand**

The certificate lives on **edges**: a binary op carries two incoming certificates and
emits a *different* one per input (mirrors the VJP routing).

| op | behaviour |
|---|---|
| add/sub | linear; preserves $\mathcal C,\mathcal S,\mathcal P$; $L$ adds |
| mul (Hadamard/scalar) | bilinear $\Rightarrow$ indefinite curvature, couples; **but** if one operand is a constant/stop-grad it is linear (scaling) |
| div $x/y$ | linear in $x$; nonconvex in $y$ |
| max/min | piecewise-linear, **gates**; CONVEX (max), no finite Hessian; $L=1$ in $\ell_\infty$ |
| prod | multilinear $\Rightarrow$ $\mathcal C\to\top$, $\mathcal P\to\textsf{COUPLED}$ |

### 4.3 Reductions

- **sum/mean**: linear; $L$ scales by $\sqrt{k}$ (sum, $\ell_2$) or $1/\sqrt{n}$ (mean).
  **Transpose of broadcast.** Sum over the **batch** axis is the K-FAC enabler
  (weight-sharing $\to$ Kronecker); sum over a coupling axis breaks $\mathcal P$.
- **max**: gates, input-dependent; CONVEX-nonsmooth; selector Jacobian.
- **logsumexp**: an **atom** (see 4.5), CONVEX with $H=\mathrm{diag}(s)-ss^\top$.
- **prod**: multilinear, $\to\top$.

### 4.4 Shape ops (reshape, transpose, permute, broadcast)

Geometrically trivial — $L=1$, $\mathcal C$ identity — but **structurally dangerous**:
they remap **axis roles**, and $\mathcal S$ is only meaningful relative to those roles.
The transfer function applies the index permutation/grouping to `AxisRoles`, degrading
to $\top$ when it merges a structured axis (matrix row/col) with an incompatible one
(batch). Broadcast and sum are **transposes**; in $\ell_2$ both scale $L$ by $\sqrt{k}$.

> Practical consequence: an optimizer that keeps selecting Kronecker/spectral rules
> must be **layout-aware**; flatten/im2col silently destroys exploitable structure
> unless the analysis tracks axis roles.

### 4.5 Atoms (non-compositional; recovered by e-matching)

Primitive composition is sound but loses precision. The two most valuable certificates
are exactly the non-compositional ones.

**Softmax + cross-entropy.** $L=\mathrm{LSE}(z)-z_y$, $s=\mathrm{softmax}(z)$:
$$
g_o = s - e_y,\qquad H = \mathrm{diag}(s) - ss^\top \ (\textsf{PSD}),\qquad
\mathcal S = \textsf{DIAG-RANK1}\ (\text{Woodbury, } O(c)),\quad \mathcal I=\langle\mathbf 1\rangle.
$$
Composed from primitives ($\log\circ\mathrm{sum}\circ\exp$) DCP **cannot** prove
convexity (concave $\circ$ convex), so $\mathcal C\to\top$; the atom restores
$\mathcal C=\textsf{CONVEX}$ and the exact $H$.

**LayerNorm** $n=\gamma\odot\hat u + \beta$, $\hat u=(u-\mu)/\sigma$:
$$
\frac{\partial n}{\partial u} = \frac{\mathrm{diag}(\gamma)}{\sigma}\Big(I - \tfrac{1}{p}\mathbf 1\mathbf 1^\top - \tfrac{1}{p}\hat u\hat u^\top\Big),
$$
so $\mathcal S=\textsf{DIAG-RANK2 (scaled)}$, $\mathcal C$ collapses CONVEX$\to$L-SMOOTH
(division by $\sigma$), $L\mathrel{*}=\|\gamma\|_\infty/\sigma$, and the gift:
**shift+scale invariance** $\Rightarrow \mathcal I=\langle\mathbf 1,\hat u\rangle$ flat
in $u$ (project these out of the preceding layer; the mean part of a pre-norm bias is
exactly in the null space).

**Linear layer** $z=Wr$ (and bias): linear in $W$, so per-example
$F_{W}=(rr^\top)\otimes H$ **exactly** ($\mathcal S=\textsf{KRON}$, exact).

**Conv**: linear, weight-shared; $F$ Kronecker-factorizes with a spatial sum (KFC).
Track the im2col layout (§4.4).

---

## 5. Worked derivation: 2-layer ReLU MLP

Network: $a=W_1x+b_1$, $r=\mathrm{relu}(a)=Da$ ($D=\mathrm{diag}(\mathbf 1[a>0])$),
$o=W_2r+b_2$. Loss: MSE $\tfrac12\|o-y\|^2$ ($H=I$) or softmax-CE
($H=\mathrm{diag}(s)-ss^\top$).

**Step 1 — assemble $F$ (NGD is born here, un-named).** The system composes
$F=\text{pullback}\circ H\circ\text{pushforward}$ and reads $H$ off the loss:
- MSE $\Rightarrow H=I\Rightarrow F=J^\top J$ (Gauss–Newton),
- CE $\Rightarrow H=\mathrm{diag}(s)-ss^\top\Rightarrow F=J^\top(\mathrm{diag}(s)-ss^\top)J$ (**Fisher** = the NGD metric).

Writing $F$ *is* writing the metric. No rule was selected.

**Step 2 — ReLU makes it exact.** Within an activation region $o$ is affine in each
weight block separately (bilinear across $W_1,W_2$), so within-block
$\nabla^2_\theta o=0$ and the **diagonal blocks of the true Hessian equal the
Gauss–Newton blocks exactly**. The only gap is the cross-layer block.

**Step 3 — per-block structure (read off $F$).**
$$
F_{W_2}=(rr^\top)\otimes H,\qquad
F_{W_1}=(xx^\top)\otimes\big(D\,W_2^\top H\,W_2\,D\big),
$$
both Kronecker (exact / exact-given-gate). The certificate's $\mathcal S=\textsf{KRON}$
is the *guard* certifying the rewrite below is exact.

**Step 4 — rewrite + extract.** From $\Delta\theta=-\mathrm{solveCG}(F,g)$, fire
$(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}$, block-diagonal drop (error bounded by
$\mathcal P$), Woodbury for the CE head. Cost-extraction picks the closed form:
$$
\Delta W_2=-\,H^{-1}\,G_{W_2}\,(rr^\top)^{-1},\qquad
\Delta W_1=-\,(D W_2^\top H W_2 D)^{-1}\,G_{W_1}\,(xx^\top)^{-1}.
$$
Specialize $H$:
- **MSE** ($H^{-1}=I$): $\Delta W_2=-G_{W_2}(rr^\top)^{-1}$ — Gauss–Newton, which by
  Step 2 is the **exact block-Newton step**.
- **CE** ($H^{-1}=(\mathrm{diag}(s)-ss^\top)^{+}$): $\Delta W_2=-(\mathrm{diag}(s)-ss^\top)^{+}G_{W_2}(rr^\top)^{-1}$ — the exact **natural-gradient** head step.

Term-for-term this is K-FAC / NGD / Newton, reached by minimizing the cost of applying
$F^{-1}g$ — never by naming a method. The rule **adapts to dimensions**: huge vocab
$c\Rightarrow$ Woodbury keeps the head $O(c)$; huge width $h\Rightarrow$ extraction
prefers CG/low-rank for that block.

---

## 6. Testing plan and toy examples

A correct implementation must pass, in order, the following. Each toy case has a
**closed-form expected result** so tests are exact (up to float tolerance), not
"looks reasonable."

### 6.1 Transfer-function unit tests (certificate layer)

For each primitive, assert the transfer function is (a) monotone w.r.t. the lattice,
(b) sound under composition against a brute-force numeric check on random small graphs.

```python
def test_relu_diag_and_gate():
    cert = transfer_unary(relu_p, downstream=Cert(C=CONVEX, S=DIAG, ...))
    assert cert.S == Struct.DIAG
    assert "ones" not in cert.I            # relu introduces no symmetry
    # numeric soundness: certificate L >= empirical Lipschitz on random inputs
    assert cert.L >= empirical_lipschitz(jax.nn.relu)

def test_layernorm_atom_invariances():
    cert = atom_layernorm(...)
    assert "ones" in cert.I and "radial" in cert.I
    assert cert.S == Struct.DIAG_RANK2
    assert cert.C == Curvature.L_SMOOTH    # NOT convex, NOT top

def test_softmax_ce_atom():
    cert = atom_softmax_ce(...)
    assert cert.C == Curvature.CONVEX
    assert cert.S == Struct.DIAG_RANK1
    assert "ones" in cert.I
    # H == diag(s) - s s^T, PSD, null space span(1)
    H = materialize_head_hessian(s)
    assert is_psd(H) and onehot_ones_in_null(H)
```

### 6.2 End-to-end derivation tests (the math must be exact)

| Toy problem | Expected extracted step | Convergence target |
|---|---|---|
| 1-D quadratic $\tfrac12 a\theta^2$ | $\Delta\theta=-g/a$ (Newton) | exact min in **1 step** |
| Linear regression $\tfrac12\|X\theta-y\|^2$ | $-(X^\top X)^{-1}X^\top r$ (normal equations) | 1 step (up to CG tol) |
| Logistic / softmax head (linear) | $-(\mathrm{diag}(s)-ss^\top)^{+}G(XX^\top)^{-1}$ | local quadratic convergence |
| 2-layer ReLU MLP (§5) | block-Kron $\Rightarrow$ matches K-FAC update | matches reference K-FAC trajectory |

```python
def test_quadratic_newton_one_step():
    f = lambda th: 0.5 * 3.0 * th**2
    opt = compile_optimizer(f, example=jnp.array(7.0))
    th, state = opt.init(7.0)
    th, _ = opt.update(th, state)           # one step
    assert jnp.allclose(th, 0.0, atol=1e-6) # Newton solves a quadratic in 1 step

def test_linreg_matches_normal_equations():
    X, y = random_regression(n=64, d=8)
    f = lambda th: 0.5 * jnp.sum((X @ th - y)**2)
    th_star = jnp.linalg.solve(X.T@X, X.T@y)
    th = run(compile_optimizer(f, jnp.zeros(8)), steps=1)
    assert jnp.allclose(th, th_star, atol=1e-4)

def test_mlp_extracts_kfac():
    term = compile_optimizer(mlp_ce_loss, params).extracted_term
    assert is_block_diagonal_kron(term)     # structurally equals K-FAC
```

### 6.3 The minimal kill-test (falsifiable, ship first)

Implement **only**: per-block step size $\eta_b=c/L_b$ + Muon-on-matmul / Adam-elsewhere
(the $\mathcal S$ factor) + symmetry projection on pre-norm linear and softmax heads.
On a small Transformer, compare against tuned AdamW / Muon / SOAP.

- **Hypothesis:** matches the best baseline **without any per-block LR sweep**; edges
  ahead on heterogeneous-block mixtures.
- **Kill condition:** if the $L$-only variant cannot remove the LR sweep without losing
  ground, the richer factors will not rescue it — stop and rethink.

### 6.4 Property / regression tests

- **Homogeneity:** on a matmul-only net the extractor must select Muon everywhere and
  **match** Muon (it should not beat it). Gains may appear only on mixed graphs.
- **Per-step descent:** for every extracted block rule with a soundness guard
  satisfied, assert (per batch) $L(\theta+\Delta\theta)\le L(\theta)$ for
  $\eta_b\le 1/L_b$.
- **Join safety:** when separability is removed, blocks must coarsen to the meet and the
  extracted rule must equal the safe (AdamW-with-$L$) fallback, never a Kron/spectral
  rule that the certificate did not license.
- **Lowering equivalence:** the lowered jitted term must numerically equal a reference
  hand-written implementation of the same algebra term.

---

## 7. Architecture coverage

Relevance $\propto$ exploitable provable structure. Two kinds: **outer-loop** (train
the weights — does the certificate find structure?) and **inner-loop** (the layer's
forward pass *is* an optimizer — does update-synthesis apply to designing the layer?).

### 7.1 Traditional deep learning

| Arch | Structure found | Rules that fire | Status |
|---|---|---|---|
| **MLP** | linear layers $\Rightarrow$ exact Kron; activations $\Rightarrow$ diag; softmax-CE head $\Rightarrow$ exact NG | K-FAC / NGD / exact head | works (§5); modest novelty |
| **ConvNet** | conv linear + weight-shared $\Rightarrow$ structured Kron (KFC); BN $\Rightarrow$ scale-invariance | per-layer Kron + symmetry projection | works; **track im2col layout** |
| **RNN / LSTM / GRU** | gate projections Kron-friendly; gating $\odot$ bilinear $\Rightarrow$ $\mathcal C\to\top$, coupled; BPTT unroll $\Rightarrow$ $L$ detects vanishing/exploding | Kron on projections; $L$-scaled steps in the core | moderate; recurrent core degrades |

### 7.2 Modern AI

| Arch | Structure / mechanism | Rules / needs | Loop |
|---|---|---|---|
| **Transformer / ViT** | ~all params in matmuls (QKVO/FFN $\Rightarrow$ Kron) + embeddings (row-sparse, separable) + norms ($\mathcal I$ symmetry) + residuals (linearize coupling, help $\mathcal P$) + softmax-CE head (exact NG); GELU/softmax are curvature firewalls | spectral/Kron on projections; sparse row updates on embeddings; symmetry projection; exact head NG | outer (strong; Muon/SOAP already win here) |
| **RoPE** | parameter-free orthogonal rotation; isometry ($L=1$, singular values preserved) | transparent; $\mathcal I/\mathcal L$ keep Q/K Kron intact | outer (non-breaking) |
| **SSM / Mamba** | linear recurrence, diagonal/structured $A$ $\Rightarrow$ per-channel separable; selective $A,B,C$ = input-dependent gate | needs a **scan/recurrence combinator**; curvature-through-unroll | outer; **requires algebra extension** |
| **DeltaNet / gated DeltaNet** | state update is **one online gradient step on a local regression loss** $\tfrac12\|S^\top k_t - v_t\|^2$; delta projector $(I-\beta_t k_tk_t^\top)$; channel-wise decay/gates | derive the inner update as **preconditioned online Gauss–Newton on KV regression with gating**; rank-1 / diag-gate combinators | **inner** (high novelty) |
| **TTT** | hidden state *is* a model (linear or 2-layer MLP), updated by SGD on a self-supervised loss $\|f(\tilde x_t;W)-x_t\|^2$ | synthesize/precondition the inner step (currently hand-tuned: SGD + output norm); applies at inner **and** outer levels | **inner** (cleanest fit) |

The deepest relevance is inner-loop: DeltaNet, gated DeltaNet, and TTT put
*learning inside the forward pass*, so update-synthesis becomes an **architecture-design
tool**, not just a training accelerator. Honest caveats: recurrent/scan/fast-weight
cores need new combinators (`scan`, gated-linear-recurrence, low-rank-state) and
curvature-through-unroll is the least-developed analysis; outer-loop accelerator value
compresses at the very largest LLM scale where Muon/SOAP already capture the structure.

---

## 8. Extension catalog (the axes as an API)

Each axis is registered vocabulary (op, cost, guard, rewrites); extractor/cost/soundness
machinery is untouched. Optimizers are **points in the product space**.

| # | Axis | What varies | Named instances | Guard source |
|---|---|---|---|---|
| 1 | Source | how $g,Fv$ are obtained | exact-AD, empirical-Fisher (EMA), probe (SPSA/ES) | — |
| 2 | Time | filtering across steps | momentum, heavy-ball, Nesterov, Adam moments | — |
| 3 | Order | iterate AD $k\times$ | Newton, Halley, cubic-regularized | $\mathcal C$ |
| 4 | Meta | $\textsf{Update}\to\textsf{Update}$ | lookahead, hypergradient, iterate averaging | — |
| 5 | Geometry | norm / LMO / Bregman | SGD ($\ell_2$), Muon (spectral), Lion/signSGD ($\ell_\infty$), mirror/EG | $\mathcal S$ |
| 6 | Constraint | projection / prox / manifold | ISTA/FISTA, Frank–Wolfe, Riemannian (Stiefel), ADMM | $\mathcal C,\mathcal I$ |
| 7 | Curvature identity | which $H$ | Hessian, GN, Fisher, empirical-F, generalized-GN | $\mathcal C,\mathcal S$ |
| 8 | Variance reduction | control variates / snapshots | SVRG, SAGA, SARAH | unbiasedness |
| 9 | Adaptivity | scalars from measurements | line search, trust region, Polyak, LM damping | descent cond. |
| 10 | Granularity | grain + schedule | per-param/layer/global; Jacobi vs Gauss–Seidel | $\mathcal P$ |
| 11 | Objective | what is differentiated | SAM, entropy-SGD, proximal-point/implicit, teacher | — |
| 12 | Trajectory | multi-point relationships | L-BFGS (secant), Anderson, CMA-ES, SVGD | — |
| 13 | Representation | bits + compression | 8-bit states, error-feedback, top-k, GaLore | error-feedback |
| 14 | Aggregation | combine estimates | clip, LARS/LAMB, median/Krum, DP | robustness/privacy |

Orthogonality is approximate (quasi-Newton straddles 3/12; AdaGrad straddles 1/9;
clipping straddles 6/14). A method on a seam just means two combinators co-fire.

---

## 9. Implementation roadmap

- **v0 — kill-test (weeks).** No e-graph. Certificate computes only $\mathcal S$, $L$,
  $\mathcal I$. Hard-coded selection: Muon on matmul, Adam elsewhere, $\eta=c/L$,
  symmetry projection on pre-norm + softmax. Validate §6.3 against tuned AdamW/Muon/SOAP.
- **v1 — atoms + exact e-graph.** Add the atom library (softmax-CE, LayerNorm, linear,
  conv), the operator IR, exact rewrites only (Kron-inverse, transpose, fusion), cost
  extraction. Reproduce §5 derivations and §6.2 exactly. K-FAC/NGD emerge.
- **v2 — approximate e-graph + axes 1–4.** Lossy rewrites with certificate-supplied
  error bounds; Source/Time/Order/Meta combinators. Adam, momentum, hypergradients,
  zeroth-order patches become reachable. Add the validation loop (meta-eval or online
  curvature residual).
- **v3 — recurrent/scan + inner-loop.** `scan`/gated-linear-recurrence/low-rank-state
  combinators (SSM, RNN). Inner-loop synthesis for DeltaNet/gated DeltaNet/TTT:
  derive/precondition the fast-weight update on its local regression loss.
- **v4 — full axis catalog + hardware cost.** Geometry/Constraint/VR/Adaptivity/etc.;
  hardware-aware cost model (flops/mem/**bits**), closing the loop with §7's
  representation axis.

---

## 10. Open problems and honest limitations

- **Approximate e-graphs.** Standard equality saturation is exact-equality. We need
  *lossy* rewrites carrying error bounds (certificate-supplied), with extraction trading
  cost against accumulated error. This is the hardest, least-solved piece; the
  floating-point/Herbie line is the closest precedent.
- **Validation of novel terms.** A discovered hybrid must be shown to actually converge
  — via offline meta-eval (AutoML-Zero style) or an online curvature-fit residual. JAX's
  rewrites are semantics-preserving and need no such loop; ours are speculative and do.
- **$\top$-domination.** The interesting properties are precise on linear layers, norms,
  embeddings, elementwise ops, and uninformative on the genuinely coupled nonlinear
  core. Realistic claim: specialize where provable, fall back elsewhere; the empirical
  question is whether the provable fraction is large enough (for transformers it largely
  is — most params are in matmuls and embeddings).
- **Curvature-through-scan.** Curvature through long unrolled recurrences (RNN/SSM/fast
  weights) is the least-developed analysis; high potential, requires research.
- **Stochasticity.** All curvature here is the empirical-batch one; guarantees are
  per-batch / in-expectation, as with every second-order method.
- **Scale.** Structure-aware optimizers' edge over AdamW shrinks at frontier-LLM scale
  (~$1.4\times$ at 0.1B $\to$ ~$1.1\times$ at $>$1B); biggest outer-loop wins are likely
  small-to-medium scale and in the novel inner-loop architectures.
- **Relative to prior art.** Optimizer *program search* exists (AutoML-Zero; Lion via
  symbolic search) but over **elementwise** gradient DSLs — curvature is not first-class,
  so they structurally cannot discover K-FAC/NGD/Muon. The new content is making the
  propagated pushforward/pullback/curvature first-class IR so second-order rules become
  *reachable* in the search.

---

## 11. References

Foundations of AD and its program-transformation view:
- Elliott. *The Simple Essence of Automatic Differentiation.* ICFP 2018.
- Radul, Paszke, Frostig, Johnson, Maclaurin. *You Only Linearize Once: Tangents
  Transpose to Gradients.* POPL 2023.
- Wang, Zheng, Decker, Wu, Essertel, Rompf. *Demystifying Differentiable Programming:
  Shift/Reset the Penultimate Backpropagator.* ICFP 2019.
- Moses, Churavy. *Instead of Rewriting Foreign Code for ML, Automatically Synthesize
  Fast Gradients* (Enzyme). NeurIPS 2020.
- Vákár, Smeding. *CHAD: Combinatory Homomorphic Automatic Differentiation.* 2022.
- Abadi, Plotkin. *A Simple Differentiable Programming Language.* POPL 2020.

Categorical / semantic foundations:
- Cruttwell, Gavranović, Ghani, Wilson, Zanasi. *Categorical Foundations of
  Gradient-Based Learning.* ESOP 2022.
- Cockett et al. *Reverse Derivative Categories.* CSL 2019.
- Fong, Spivak, Tuyéras. *Backprop as Functor.* LICS 2019.

Equality saturation / e-graphs:
- Tate, Stepp, Tatlock, Lerner. *Equality Saturation.* POPL 2009.
- Willsey, Nandi, Wang, Flatt, Tatlock, Panchekha. *egg: Fast and Extensible Equality
  Saturation.* POPL 2021.
- Zhang, Wang, Flatt, Cao, Zucker, Rosenthal, Tatlock, Willsey. *Better Together:
  Unifying Datalog and Equality Saturation* (egglog). PLDI 2023.
- Suciu, Wang, Zhang. *Semantic Foundations of Equality Saturation.* ICDT 2025.
- Barrett, Tiurin, Ghica. *Equivalence Hypergraphs: E-Graphs for Monoidal Theories.* 2024.
- Yang, Phothilimthana, Wang, Willsey, Roy, Pienaar. *Equality Saturation for Tensor
  Graph Superoptimization* (TENSAT). MLSys 2021.
- Wang, Hutchison, Leang, Howe, Suciu. *SPORES.* 2020.
- Coward, Constantinides, Drane. *Automating Constraint-Aware Datapath Optimization
  using E-Graphs.* DAC 2023.

Abstract interpretation / convexity:
- Cousot, Cousot. *Abstract Interpretation.* POPL 1977.
- Grant, Boyd. *Disciplined Convex Programming* / CVXPY.

Optimizers (the targets the system should derive):
- Martens, Grosse. *Optimizing Neural Networks with Kronecker-Factored Approximate
  Curvature* (K-FAC). ICML 2015.
- Gupta, Koren, Singer. *Shampoo.* ICML 2018.
- Jordan et al. *Muon.* 2024.
- Vyas, Morwani, Zhao, et al. *SOAP: Improving and Stabilizing Shampoo using Adam.* 2025.
- Pethick et al. *Scion* (LMO-based optimization). 2025.

Optimizer program search (prior art and its gap):
- Real et al. *AutoML-Zero.* ICML 2020.
- Chen et al. *Symbolic Discovery of Optimization Algorithms* (Lion). 2023.

Modern architectures (inner-loop relevance):
- Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*
  (TTT). 2024.
- Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* 2024/2025.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.