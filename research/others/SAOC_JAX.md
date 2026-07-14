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
   - [Build decision](#11-build-decision)
2. [Theory](#2-theory)
   1. [The certificate: a second tape](#21-the-certificate-a-second-tape)
   2. [The abstract domain $\mathcal{G}$](#22-the-abstract-domain-mathcalg)
   3. [The operator algebra](#23-the-operator-algebra)
   4. [Discovery by equality saturation](#24-discovery-by-equality-saturation)
   5. [Generalization: the axis taxonomy](#25-generalization-the-axis-taxonomy)
3. [System architecture on JAX](#3-system-architecture-on-jax)
   - [Quickstart, onboarding, and the 80/20 knobs](#37-quickstart-onboarding-and-the-8020-knobs)
4. [Transfer functions](#4-transfer-functions)
5. [Worked derivation: 2-layer ReLU MLP](#5-worked-derivation-2-layer-relu-mlp)
6. [Testing plan and toy examples](#6-testing-plan-and-toy-examples)
7. [Architecture and problem coverage](#7-architecture-and-problem-coverage)
8. [Taxonomy: problem vs method](#8-taxonomy-problem-vs-method)
9. [Implementation roadmap](#9-implementation-roadmap)
10. [Open problems and honest limitations](#10-open-problems-and-honest-limitations)
11. [References](#11-references)
12. [Research vision: the relaxation ladder](#12-research-vision-the-relaxation-ladder)
13. [Duals and inverses](#13-duals-and-inverses)

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

### 1.1 Build decision

After weighing the conservative compiler, the fully generalized belief/trajectory system
(§12), a monolithic learned optimizer, and the hybrids, the commitment is:

> **Implement the conservative compiler core (§2–§11) first, architected explicitly as
> the conservative corner of the generalized system (§12), then relax in priority order
> — belief → posterior → trajectory/objective — and attach a *bounded learned residual*
> only at the very end.**

Rationale: the conservative core is the only part that has hard guarantees, is cheap and
amortized, transfers zero-shot to new architectures, has a falsifiable kill-test (§6.3),
and is implementable on JAX today with reused AD. It is also a strict special case of the
generalized system (§12.1), so building it is the foundation the relaxations open up, not
throwaway work. The principled endpoint — *derive what you can prove, learn only the
bounded residual* (§12.4) — **requires** the derived core to exist first, since the
residual is defined relative to it.

We explicitly **do not** build: a monolithic learned optimizer (expensive meta-training,
no transfer, no guarantees), data/architecture co-design (out of the optimizer frame),
or anything requiring the test distribution (information-theoretically impossible). The
phased plan is §9; the target it relaxes toward is §12.

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
nothing, fall back to AdamW". The certificate has **two tiers**: a *problem class*
(global facts about what kind of optimization this even is) and *local structure*
(per-block geometry). The taxonomy in §8 organizes the whole system around exactly
this split — the certificate detects the **problem**, the operator algebra searches the
**method**.

**Tier 1 — problem class** (detected once; governs which methods are admissible):

| Fact | Meaning | Values |
|---|---|---|
| $\textsf{Op}$ | operator class — is there a curvature at all? | $\textsf{potential}\mid\textsf{fixed-point}\mid\textsf{saddle}\mid\textsf{VI/monotone}\mid\textsf{root}$ |
| $\textsf{Dom}$ | variable / domain geometry | $\mathbb R^n\mid\textsf{manifold}\mid\textsf{simplex}\mid\textsf{lattice}\mid\textsf{function}\mid\textsf{measure}$ |
| $\textsf{Evo}$ | problem evolution | $\textsf{static}\mid\textsf{streaming}\mid\textsf{incremental}\mid\textsf{non-stationary}$ |

A non-conservative field ($\textsf{Op}\ne\textsf{potential}$) means $F$ is **not** a PSD
curvature — Newton is the wrong move and the relevant object is the (possibly
non-symmetric) operator Jacobian, calling for extragradient/fixed-point combinators.
$\textsf{Dom}\ne\mathbb R^n$ redefines what "gradient/curvature" means (Riemannian,
natural/Fisher–Rao, Wasserstein).

**Tier 2 — local structure** (per parameter block):

$$
\mathcal{G}_{\text{local}} \;=\; \mathcal{C} \times \mathcal{S} \times \mathcal{P} \times (\mathcal{L},\mu) \times \mathcal{I}
$$

| Factor | Meaning | Lattice (informal) |
|---|---|---|
| $\mathcal{C}$ | curvature / smoothness class | $\textsf{LINEAR}\sqsubseteq\textsf{QUADRATIC}\sqsubseteq\textsf{CONVEX}\sqsubseteq\textsf{L-SMOOTH}\sqsubseteq\top$ |
| $\mathcal{S}$ | structure of the Jacobian/curvature (carries **axis roles**) | $\textsf{DIAG}\sqsubseteq\{\textsf{KRON},\textsf{LOWRANK},\textsf{DIAG{-}RANK1},$ $\textsf{BANDED},\textsf{SCHUR},\textsf{CIRCULANT},\textsf{HIERARCHICAL},\textsf{MULTISCALE}\}\sqsubseteq\textsf{DENSE}\sqsubseteq\top$ |
| $\mathcal{P}$ | separability / coupling | $\textsf{SEP}\sqsubseteq\textsf{BLOCK{-}SEP}\sqsubseteq\textsf{COUPLED}\sqsubseteq\top$ |
| $(\mathcal{L},\mu)$ | Lipschitz $L$, strong-convexity $\mu$ | quantitative; $\mu=0$ if not provable |
| $\mathcal{I}$ | symmetry / invariance / **conserved quantities** | subspaces ($\langle\mathbf 1\rangle$, radial) and invariants (energy, symplectic, divergence-free) |

The $\mathcal{S}$ lattice is **enriched** beyond the deep-learning core
(diag/kron/lowrank) with the structures that scientific computing, control, vision, and
signals expose: $\textsf{BANDED}$ (control → Riccati), $\textsf{SCHUR}$/arrow (bundle
adjustment, factor graphs), $\textsf{CIRCULANT}$/Toeplitz → FFT-diagonalizable (audio,
stationary signals), $\textsf{HIERARCHICAL}$/$\mathcal H$-matrix (N-body, kernels),
$\textsf{MULTISCALE}$/elliptic (PDE). Each licenses a structured solve (see the
preconditioner family in §8, stage **Direct**). $\mathcal{I}$ now also carries conserved
quantities (Noether ties these to the symmetries it already tracked), licensing
structure-preserving / symplectic updates.

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

### 2.5 Generalization: the axis taxonomy

The static algebra acts on primitives $\{g, Fv, Tv\}$ at one instant. Every other
optimizer family is reached by extending **vocabulary**, never the selector. The
extension space is not a flat list of axes but a clean two-level taxonomy that mirrors
this system's own architecture:

- **PROBLEM** (detected by the certificate, §2.2): *operator class · domain · evolution ·
  local structure* — read-only facts that say what kind of problem this is and which
  methods are admissible.
- **METHOD** (chosen combinators, searched by the e-graph): a six-stage pipeline of one
  update — *Reframe → Sense → Model → Direct → Pace → Compose*.

Each METHOD combinator carries a **guard** that is a predicate over PROBLEM facts; that
guard *is* the link between the certificate (abstract interpretation) and the algebra
(equality saturation). The full taxonomy, the named-method decomposition, and the
per-combinator guards are in §8. An optimizer is a point in the METHOD product space,
admissible for a region of the PROBLEM space.

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
    inputs.py                      # CompileInput / Constraints / CompileConfig; ModuleAdapter (Flax/Equinox/Haiku)
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

### 3.5 Compiler input contract

The compiler consumes a **traceable JAX program + examples**, not a hand-built graph.
Everything else is metadata that *improves precision* but is optional — the compiler
degrades gracefully to jaxpr inference when it is absent.

```python
# dsl/inputs.py
@dataclass
class CompileInput:
    model:          Callable        # f(params, batch) -> output   (pure, jit-traceable)
    loss:           Callable        # loss(output, batch) -> scalar
    params_example: PyTree          # for tracing AND block structure
    batch_example:  PyTree          # for tracing (shapes/dtypes)
    loss_atom:      str | None      # "mse" | "softmax_ce" | ... | None  -> infer / HVP fallback
    module_adapter: ModuleAdapter | None   # Flax/Equinox/Haiku -> blocks + atom labels + domains
    constraints:    Constraints
    config:         CompileConfig

@dataclass
class Constraints:
    domains:      dict[BlockId, Domain]   # Euclidean | Stiefel | Simplex | Box(lo,hi) | ...  (sets P2)
    equalities:   list[Callable]          # c(params) == 0   (dynamics, KKT)
    inequalities: list[Callable]          # c(params) <= 0
    invariants:   list[Callable]          # quantities to preserve (energy/symplectic -> I factor)

@dataclass
class CompileConfig:
    flop_budget:    float = 1.5           # multiple of AdamW cost the step may use
    mem_budget:     float = 2.0           # multiple of AdamW optimizer-state memory
    precision:      str   = "fp32"        # "fp32"|"bf16"|"fp8"; non-fp32 implies error-feedback
    error_tol:      float = 1e-2          # max approximation error a lossy rewrite may add
    second_order:   bool  = True          # if False, Model is pinned to diag/identity
    enabled_stages: set   = frozenset()   # restrict METHOD combinators (default: all)
    target_hw:      str   = "a100"        # cost-model target
    refresh_every:  int   = 10            # dynamic-factor refresh schedule
    overrides:      dict   = field(default_factory=dict)  # {BlockPattern: RuleId} escape hatch
    fallback:       str   = "adamw"       # rule for TOP / over-budget / pinned-off regions
```

**Granularity / levels — the DAG is read at several resolutions, and blocks snap to the
finest separable grain:**

1. *array level* — pytree leaves (a kernel, a bias). Always available.
2. *layer level* — a leaf plus its immediate companions (kernel+bias+norm-scale).
3. *module-subtree level* — an attention block, an MLP block. From `module_adapter`.
4. *model level* — one global rule (rare; only when nothing is separable).

The block partition is the **finest grouping that $\mathcal P$ certifies as independently
updatable**, *snapped to module boundaries when a `module_adapter` is present* (modules
are strong separability priors). Where coupling is detected, blocks are merged and their
certificates `meet`-joined (§2.2).

**Loss curvature input.** Tag the loss when possible (`loss_atom="softmax_ce"`) so the
head $H=\mathrm{diag}(s)-ss^\top$ is known *symbolically* and the exact-NG head fires
unconditionally. Untagged losses fall back to a matrix-free HVP
($\textsf{jvp}\circ\textsf{grad}$) — still usable via CG, but the head $\mathcal S$
degrades to dense (no Woodbury/Kronecker exactness).

### 3.5.1 Selection policy — when SGD- vs Adam- vs NGD-like, and where "nothing"

The user does **not** specify per-region rules — that is the compiler's job. The user
sets *budget and tolerance* (`CompileConfig`); the per-region rule is the **extraction
output**, decided mechanically per block from `(cert, config)`:

```python
def select_rule(block, cert, config):
    if block matches config.overrides:            # escape hatch
        return config.overrides[block]
    if cert is TOP or not config.second_order:    # no provable structure -> plain rule
        return config.fallback                    #   ("adamw" or "sgd")
    if block.num_params < SIZE_THRESHOLD:         # overhead not worth it on tiny blocks
        return config.fallback
    rule, err, cost = egraph_extract(cert, config) # cheapest SOUND term under the budget
    if cost > config.flop_budget * adamw_cost(block) or err > config.error_tol:
        return degrade(rule, config)              # step down the lattice toward fallback
    return rule
```

The resulting map onto familiar rules (all *derived*, not named):

| Region | Certificate | Extracted rule |
|---|---|---|
| softmax-CE / regression head | CONVEX, exact $H$, DIAG-RANK1/KRON | **NGD / Newton** (exact, Woodbury) |
| dense / matmul layer | KRON (or DENSE-MATRIX) | **K-FAC** or **Muon/spectral** (cost decides) |
| conv layer | structured KRON (KFC) | **K-FAC-conv** |
| embedding | row-sparse, separable rows | **sparse row-wise** (NG or Adam) |
| activation / elementwise | DIAG | **Adam-like / diagonal** |
| norm scale/bias, small vectors | CONVEX, low-rank | small **Newton** or scaled GD |
| coupled nonlinear core | $\top$ | **SGD / AdamW** (the "nothing special" case) |
| tiny block / over-budget / user-pinned-off | any | **fallback** (SGD/AdamW) |

So "which part gets plain SGD and which does not" is exactly: $\top$-regions,
sub-threshold blocks, over-budget blocks, and user-pinned ones get the fallback;
everything with provable, affordable structure gets its derived structured rule.
Changing `flop_budget`/`error_tol` slides the whole network between "mostly SGD/Adam"
(tight budget) and "mostly NGD/K-FAC where provable" (loose budget) — one knob, global
effect, no per-region authoring.

### 3.6 Output: building the optimizer

**Primary output is a programmatically-constructed `optax.GradientTransformation`, not
raw jaxpr and not string-generated Python.** It jits, composes with the optax ecosystem,
and carries per-block state as a PyTree matching `params`. Raw jaxpr is rejected as the
*artifact* (internal, opaque, unstable); hand-written-template Python is rejected as
*brittle codegen*. The transform is assembled from the extracted operator-IR term by the
`lower.py` interpreter (one JAX implementation per algebra node) and made heterogeneous
per region via `optax.multi_transform`:

```python
# runtime/optimizer.py
def build_optimizer(compiled) -> optax.GradientTransformation:
    rule_table   = {label: lower_to_optax(term)   # operator-IR term -> GradientTransformation
                    for label, term in compiled.terms.items()}
    param_labels = compiled.param_labels           # PyTree mirroring params; leaf -> rule label
    return optax.multi_transform(rule_table, param_labels)

@dataclass
class CompiledOptimizer:
    tx:           optax.GradientTransformation   # the runnable optimizer (jits)
    param_labels: PyTree                         # leaf/block -> rule label
    terms:        dict[str, Op]                  # label -> extracted operator-IR term
    source:       str                            # secondary: readable Python rendering (audit)
    report:       dict                           # per block: rule, cert, cost, error, why

    def emit_source(self, path: str) -> None: ...  # write a standalone, editable JAX/optax module
```

The output is therefore **structure-aware by construction**: heterogeneous per-region
rules are applied through a `param_labels` PyTree (the JAX-idiomatic
`multi_transform`/`masked` pattern), with labels coming from the block partition (§3.5).
A *secondary*, optional artifact is a **readable Python rendering** of each term — the
operator IR is a small typed AST, so pretty-printing it to an inspectable `optax.chain`
is straightforward and is the human-facing/audit form. The artifact is deterministic in
`(model, config, example)`, so cache it as a compile product.

#### 3.6.1 Two output modes: live object (default) vs emitted source

The same extracted IR can be rendered to **standalone, human-readable JAX/optax source**
and used *instead of* the live object. The operator-IR term is a small typed AST, so
AST → source is a direct codegen pass (the same traversal as `lower_to_optax`, printing
instead of building). Two modes, one compile:

```python
compiled = soco.compile(...)

tx = compiled.tx                       # MODE 1 (default): live object — robust, guarded, re-derivable

compiled.emit_source("my_opt.py")      # MODE 2: write an editable module, then use that instead
# from my_opt import make_optimizer
# tx = make_optimizer()
```

The generated module is self-contained and *is* the audit — every rule carries its block,
its certificate, and the math as comments:

```python
# my_opt.py  — AUTOGENERATED by soco for MLP-regression @ flop_budget=1.5, fp32
# FROZEN SNAPSHOT: shapes + config + architecture baked in. Recompile if any change.
import jax, jax.numpy as jnp, optax

def _exact_gn_output(eps=1e-6):
    # block w3 : H=I (MSE) -> ΔW = -G (rrᵀ)⁻¹     [cert: CONVEX, KRON exact]
    def init(p): ...
    def update(grads, state, params=None): ...
    return optax.GradientTransformation(init, update)

def _kfac_layer(eps=1e-6, refresh=10):
    # blocks w1,w2 : GN-Kron  (rrᵀ)⁻¹ ⊗ (DWᵀHWD)⁻¹   [cert: L-SMOOTH, KRON approx]
    ...
    return optax.GradientTransformation(init, update)

def _small_newton(): ...    # biases  [cert: CONVEX, LOWRANK]

def make_optimizer():
    labels = {"w1": "kfac", "w2": "kfac", "w3": "gn_out", "b1": "newton", ...}
    return optax.multi_transform(
        {"gn_out": _exact_gn_output(), "kfac": _kfac_layer(), "newton": _small_newton()},
        labels)
```

**It is brittle — use it knowing exactly why:**

- *Pinned snapshot.* Specialized to the traced shapes, the chosen budget/config, and the
  architecture at compile time. Change vocab/width/batch structure, or want a different
  budget → it is stale; recompile.
- *Guards dropped.* The certificate soundness conditions that *licensed* each
  approximation are not re-checked at runtime — the file is the conclusion, not the proof.
  You can hand-edit freely, but a manual change (e.g. forcing Kron on a block that was not
  certified separable) silently voids soundness.
- *Data-dependent structure frozen.* The emitter keeps the dynamic-factor refresh code (so
  K-FAC factors still update each `refresh` steps), but the *structural* decisions — which
  rule per block, gate-dependent $\mathcal S$ — are frozen to compile-time assumptions. If
  the activation regime shifts, the live object re-derives; the source does not.
- *One-shot.* No re-extraction; the live object can be recompiled with a new config, the
  source cannot.

**When to prefer source:** auditing or teaching what was derived; debugging a suspicious
update; forking into a hand-maintained custom optimizer; shipping to an environment that
should not carry the compiler (and its e-graph) as a runtime dependency. The live object
is for everything else.

**Equivalence guarantee.** The emitted source must numerically equal `compiled.tx` on the
traced shapes — this is the lowering-equivalence property test (§6.4), and CI should diff
the two. The moment you hand-edit the file, that guarantee is void by construction and the
file is yours to maintain.

**Framework awareness — Flax/Equinox make both ends easier.** Provide a `ModuleAdapter`:

| Framework | Block tree / labels | Domains | Output keying |
|---|---|---|---|
| **Flax** | module paths via `flax.traverse_util`; class names (`nn.Dense`, `nn.LayerNorm`, `nn.MultiHeadDotProductAttention`) give atom labels directly | from module config | `multi_transform` keyed by module path |
| **Equinox** | the module *is* the params PyTree; block boundaries = module boundaries; `eqx.partition`/`eqx.filter` select leaves | dataclass field tags | filtered `multi_transform` over the module tree |
| **Haiku / raw** | pytree key paths (`jax.tree_util.keystr`) as weak priors | explicit `Constraints.domains` | path-prefix labels |

With Flax/Equinox the certificate's **atom-matching is trivial** (read the module class
instead of inferring softmax/LayerNorm from the jaxpr) and the **block partition is
given** (module subtree = block), so steps 3 (lift+certify) and the output keying both
shortcut. Absent an adapter, the compiler infers block structure from pytree paths +
jaxpr pattern-matching — same result, more work, lower precision on atoms.

### 3.7 Quickstart, onboarding, and the 80/20 knobs

**Mental model: you write a model and a loss; you do not write an optimizer.** You hand
the compiler a normal JAX/Flax/Equinox `model` + `loss` + example pytrees, and it returns
an `optax.GradientTransformation` you drop into a standard training loop. It *derives* a
per-region rule (NGD on the head, K-FAC/Muon on matmuls, Adam on the rest) and tells you
what it did in a `report`.

The 30-second version:

```python
import soco, optax

compiled = soco.compile(model, loss, params, batch, loss_atom="softmax_ce")
tx = compiled.tx                       # a normal optax optimizer; jits and composes
state = tx.init(params)
# standard loop:
grads = jax.grad(lambda p: loss(model(p, batch), batch))(params)
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)

print(compiled.report)                 # what it derived per block, and why
```

**The 20% of knobs that cover 80% of usage** (everything else has good defaults):

| Knob | What it does | Default | Touch when |
|---|---|---|---|
| `loss_atom` | tags the loss so head curvature is exact (the head-NG win) | `None` → infer/HVP | **always** — set `"mse"` / `"softmax_ce"`; biggest single win, free |
| `module_adapter` | hands over your block tree + atom labels | `None` → infer | **always if Flax/Equinox** — precision + zero inference |
| `flop_budget` | master dial: tight → SGD/Adam-like, loose → NGD/K-FAC where provable | `1.5×` | first thing to sweep; raise to admit more structure |
| `second_order` | global on/off | `True` | set `False` for a safe well-scaled-AdamW baseline first |
| `fallback` | rule for $\top$/over-budget regions | `"adamw"` | `"sgd"` for a leaner baseline |
| `precision` | optimizer-state dtype + error-feedback | `"fp32"` | `"bf16"`/`"fp8"` when memory-bound (large LMs) |
| `refresh_every` | curvature-factor refresh cadence | `10` | raise to cut overhead, lower for fast-moving curvature |
| `overrides` | pin a region to a rule | `{}` | rarely — debugging or hard domain knowledge |

**Reading the report** is the core onboarding skill: each block shows `(rule, certificate,
cost, error, why)`, e.g. `head Dense → exact-NGD (CONVEX, DIAG-RANK1)`,
`block3.conv → KFC`, `mlp.glue → AdamW (S=⊤)`. That last line is how you see *which parts
got "nothing special"* and why.

#### Example A — MLP regression (the §5 case, runnable)

```python
def model(p, b):
    h = jnp.tanh(b["x"] @ p["w1"] + p["b1"])
    h = jnp.tanh(h     @ p["w2"] + p["b2"])
    return h @ p["w3"] + p["b3"]
def loss(pred, b): return 0.5 * jnp.mean((pred - b["y"]) ** 2)

compiled = soco.compile(model, loss, params, batch, loss_atom="mse")   # that's it
```
Derived report:
```
w3 (output)  -> exact Gauss-Newton   # H=I  -> whiten grad by activation Gram (rrᵀ)⁻¹
w1, w2       -> K-FAC (approx)
b1..b3       -> small Newton
```
What to tune: usually nothing. If steps feel expensive, `refresh_every=25`; for more
aggressive curvature use, `flop_budget=3.0`.

#### Example B — ResNet on CIFAR (Flax)

```python
net = ResNet18(num_classes=10)                      # flax.linen module
params = net.init(key, x)["params"]
fwd  = lambda p, b: net.apply({"params": p}, b["x"])
loss = lambda logits, b: optax.softmax_cross_entropy_with_integer_labels(logits, b["y"]).mean()

compiled = soco.compile(
    fwd, loss, params, batch,
    loss_atom="softmax_ce",
    module_adapter=soco.flax(net),                  # ResNet blocks + conv/bn/dense labels for free
    config=soco.Config(flop_budget=2.0, precision="bf16"),
)
```
Derived report:
```
head Dense        -> exact NGD (Woodbury on diag(s)-ssᵀ)
conv blocks       -> K-FAC-conv (KFC); im2col layout tracked
BatchNorm scales  -> symmetry projection + scaled GD
residual glue     -> AdamW (S=⊤)
```
What to tune: `flop_budget` is the lever — convs hold most FLOPs, so `2.0` lets them get
KFC while `1.2` pushes them to Adam; `precision="bf16"` for memory.

#### Example C — Transformer LM (Equinox or Flax)

```python
compiled = soco.compile(
    fwd, loss, params, batch,
    loss_atom="softmax_ce",
    module_adapter=soco.equinox(model),
    config=soco.Config(flop_budget=1.3, precision="bf16", refresh_every=20),
)
```
Derived report:
```
LM head            -> exact NGD (Woodbury; cheap even at large vocab)
Q/K/V/O, FFN       -> Muon / spectral   # cost picks spectral over Kron at this width
token embeddings   -> sparse row-wise
LayerNorm/RMSNorm  -> symmetry projection
RoPE               -> transparent (isometry; no params)
```
What to tune: at scale keep `flop_budget` tight (≈1.2–1.3) so matmuls get cheap spectral
and only the head gets exact NG (the scale caveat, §7/§10); raise `refresh_every` to
amortize factor cost; `precision="fp8"` states if memory-bound.

#### Recommended onboarding loop

1. **Start safe.** `Config(second_order=False, fallback="adamw")` → a well-scaled AdamW
   (per-block $\eta=1/L$ + symmetry projection). Confirms plumbing; gives a baseline.
2. **Turn on the cheap wins.** `second_order=True`, set `loss_atom`, pass
   `module_adapter`. The exact-NG head and symmetry projections come first — cheapest and
   safest gains.
3. **Sweep `flop_budget` upward**, watching the `report` and a val curve; stop where the
   cost/benefit flattens.
4. **Only then** reach for `overrides` (debugging) and `precision` (memory).

Two expectations from §6.4: on a *homogeneous* net (e.g. matmul-only) the compiler should
**match** the best single rule, not beat it — gains appear on *heterogeneous* graphs
(everything above). And compilation is amortized: it runs once per architecture, so cache
`compiled` and reuse across runs.

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

### 6.5 Relaxation baselines and the non-stationary validation gate

The core tests (§6.1–6.4) validate v0–v5. The relaxations (v6–v9) need a stricter
discipline, imported from a sibling MHE/MPC project (§12.6) whose experiments showed that
sophisticated estimation-plus-planning **fails to beat trust-region Newton on stationary
smooth problems** and wins only in the non-stationary / saddle regime.

- **Isolate the marginal value of each rung against the cheaper version of itself**, not
  against AdamW. v6 must beat *AdamW + a cheap diagonal curvature estimate*; v7 must beat
  *single surrogate + LM damping*; v8a must beat *plain extragradient / optimistic gradient
  at matched compute*. A rung that only beats AdamW has not earned its layer.
- **Stationary null check.** On stationary smooth problems, a trajectory relaxation must
  **not lose** to trust-region Newton (the sibling project's confirmed null). If a rung
  adds cost without benefit on the default cell (`P1 = potential`, `P3 = static`), gate it
  off there — it is for the non-default cells only.
- **The decisive non-stationary gate (single go/no-go for the trajectory layer).** On a
  bilinear / WGAN-style game (`P1 = saddle`) and an online-drift task (`P3 =
  non-stationary`), the compiler must **automatically derive curvature-aware optimism**
  (v5 operator-class → extragradient, plus v6 belief → curvature) and **beat plain
  optimism at matched compute**. This tests the *compiler thesis itself* — that it can
  derive, from the problem class, the combination a human found by hand — and is more
  decisive for the relaxation layer than any stationary benchmark. If it merely ties plain
  optimism, the trajectory layer is a unification, not a new capability.
- **Dual-control diagnostic.** With confidence-gated belief (v6), log estimator covariance
  and an excitation/observability measure; verify the optimizer is not starving its own
  estimator (the trajectory shaping which directions are observable).

---

## 7. Architecture and problem coverage

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

### 7.3 Problem classes beyond weight-training

The deep-learning core assumes a static, differentiable, scalar loss over Euclidean
parameters. Other domains break that (see PROBLEM tier in §2.2 / §8). Each row gives the
PROBLEM coordinates, what existing METHOD stages already cover, and what must be added.

| Domain | PROBLEM coordinates | Covered by | Needs |
|---|---|---|---|
| **RL** | $\textsf{Op}=\textsf{fixed-point}$ (TD), $\textsf{Evo}=\textsf{non-stationary}$ | Sense (score vs pathwise), variance reduction (baselines/GAE), Direct (PPO/TRPO KL trust) | operator-class combinators (fixed-point/damped), drift-aware Compose |
| **Robot / optimal control** | $\textsf{Op}=\textsf{potential}$ + dynamics constraints, $\mathcal S=\textsf{BANDED}$, $\textsf{Evo}=\textsf{incremental}$ (MPC) | Direct (constraint/prox), Pace (LM damping) | Riccati preconditioner, warm-start/incremental Compose |
| **Computer vision (deep)** | standard | §7.1 ConvNet | — |
| **Geometric vision / bundle adjustment** | $\textsf{Op}=\textsf{root}$ (NLS), $\mathcal S=\textsf{SCHUR}$ | Model (GN), Pace (LM) | Schur-complement preconditioner |
| **SLAM** | $\textsf{Dom}=SE(3)$ manifold, $\mathcal S=\textsf{SCHUR}$, $\textsf{Evo}=\textsf{incremental}$ | Model (GN) | manifold retraction (Dom), sparse-Cholesky precond, incremental factor update |
| **Spatial / speech audio** | $\mathcal S=\textsf{CIRCULANT}$; sequence structure | Model (GN), scan | FFT/spectral preconditioner (basis-aware $\mathcal S$) |
| **Scientific computing / physics / ODE / PDE** | $\textsf{Dom}=\textsf{function}$, $\mathcal S=\textsf{MULTISCALE}$, conserved $\mathcal I$ | Sense (adjoint = pullback), Reframe (implicit for stiffness), Direct (PDE-constrained) | multigrid/spectral/domain-decomposition preconditioners; symplectic/structure-preserving Compose |
| **Biology** | mix: physics-energy; ODE fitting; $\textsf{Dom}=\textsf{lattice}$ (trees/alignment) | physics rows above, inverse/GN | combinatorial-domain combinators (DP, tree search) |
| **Discrete / distribution optimization** | $\textsf{Dom}=\textsf{lattice}$ or $\textsf{measure}$; $\textsf{Op}=\textsf{saddle}$ (adversarial) | Reframe/Sense (relaxation, score), Direct (mirror/simplex; Wasserstein), Sense (particles/SVGD) | domain-geometry combinators (lattice, measure); saddle/extragradient |

**Honest value.** Many of these fields already have mature, near-optimal hand-built
structure exploiters — multigrid, iLQR/DDP, sparse bundle adjustment, symplectic
integrators, ADMM, extragradient. There the system mostly *re-derives* known solvers (a
good correctness test, like re-deriving K-FAC). The marginal value is highest where
structure exists but is not routinely exploited — the learning-heavy interiors:
RL optimizers, neural PDE solvers / operators, differentiable SLAM and control, learned
samplers — i.e. precisely where these classical domains are now being fused with deep
learning and nobody has hand-tuned the structure-aware optimizer yet.

---

## 8. Taxonomy: problem vs method

The flat list of axes collapses under one observation: some dimensions describe the
**problem** (you detect them, you do not choose them) and the rest describe the
**method** (you choose them, guarded by the problem). This is exactly the system's own
architecture — the certificate detects PROBLEM, the e-graph searches METHOD, and each
combinator's guard is a predicate over PROBLEM facts. Every previously-listed axis folds
into one of these.

### 8.1 PROBLEM — four detected facts (the certificate)

| # | Dimension | Values | Detected fact / guard it supplies |
|---|---|---|---|
| **P1** | Operator class | potential $\mid$ fixed-point $\mid$ saddle $\mid$ VI/monotone $\mid$ root | conservative? contractive? monotone? — decides whether a PSD curvature exists |
| **P2** | Domain geometry | $\mathbb R^n\mid$ manifold $\mid$ simplex $\mid$ lattice $\mid$ function $\mid$ measure | which gradient/curvature/metric is even meaningful |
| **P3** | Evolution | static $\mid$ streaming $\mid$ incremental $\mid$ non-stationary | drift bound; whether to reuse work |
| **P4** | Local structure | $\mathcal C,\mathcal S$ (enriched), $\mathcal P,(\mathcal L,\mu),\mathcal I$ | per-block curvature/structure/separability/conditioning/symmetry |

### 8.2 METHOD — a six-stage pipeline (the combinators)

The data flow of one update. Each stage is registered vocabulary `(op, cost, guard,
rewrites)`; the extractor/cost/soundness machinery never changes. Stages are *mostly*
independent → the method space is a product the e-graph searches.

| Stage | Question | Folded axes | Named instances | Guard from |
|---|---|---|---|---|
| **M1 Reframe** | what objective/operator do I actually attack? | Objective | SAM/sharpness, smoothing/entropy-SGD, proximal-point/implicit, teacher/distillation | P1, $\mathcal C$ |
| **M2 Sense** | how do I obtain & clean the primitives $g,Fv,Tv$? | Source, Order, Variance-reduction, Trajectory, Representation, Aggregation | AD / EMA-estimate / probe (SPSA, ES) / secant (L-BFGS) / population (CMA-ES); $k$-th order; SVRG/SAGA; 8-bit + error-feedback; median/Krum/clip | unbiasedness, error-feedback, P2 |
| **M3 Model** | which curvature, which approximation? | Curvature-identity + the curvature-approx grid | Hessian / GN / Fisher / empirical-F; exact / Kron / diag / spectral-2nd-moment / low-rank / sparse | $\mathcal C,\mathcal S$, P1 |
| **M4 Direct** | turn signal+model into a direction under what geometry & constraints? | Geometry, Constraint + preconditioner family | $\ell_2$/spectral/sign/mirror; prox/projection/Frank–Wolfe/retraction; Kron-inv / Newton–Schulz / pth-root / **multigrid / spectral(FFT) / Schur / Riccati** | $\mathcal S,\mathcal I$, P2 |
| **M5 Pace** | magnitude and temporal dynamics? | Time, Adaptivity | momentum/heavy-ball/Nesterov/Adam-EMA; line search / trust region / Polyak / LM damping; extragradient/optimistic | $\mathcal L,\mu$, P1 |
| **M6 Compose** | where, when, wrapped, reused? | Granularity, Meta, Online-reuse | per-param/layer/global; Jacobi vs Gauss–Seidel; lookahead/hypergradient/averaging; warm-start/incremental factor update | $\mathcal P$, P3 |

Mnemonic: **Reframe → Sense → Model → Direct → Pace → Compose**, admissible for a region
of (P1, P2, P3, P4).

### 8.3 Named methods as coordinates

The taxonomy organizes the zoo: each method is a point.

| Method | PROBLEM | Reframe | Sense | Model | Direct | Pace | Compose |
|---|---|---|---|---|---|---|---|
| **SGD** | pot, $\mathbb R^n$, static | — | AD $g$ | $F\!\approx\!I$ | $\ell_2$ | fixed $\eta$ | global |
| **Adam** | pot, $\mathbb R^n$, static | — | AD $g$ | $\mathrm{diag}$(temporal $g^2$) | elementwise isqrt | EMA momentum | per-param |
| **Muon** | pot, $\mathbb R^n$, static | — | AD $g$ | spectral 2nd-moment | Newton–Schulz (spectral) | momentum | per-matrix |
| **K-FAC** | pot, $\mathbb R^n$, static | — | AD $g$ | GN, Kron | Kron-inverse | — | per-layer |
| **L-BFGS** | pot, $\mathbb R^n$, static | — | secant history | quasi-Newton | 2-loop recursion | line search | global |
| **SAM** | pot, $\mathbb R^n$, static | sharpness | AD $g$ at perturbed pt | $F\!\approx\!I$ | $\ell_2$ | momentum | global |
| **PPO** | pot, $\mathbb R^n$, **non-stat** | clipped surrogate | AD policy grad | — | KL trust region | adaptive | online |
| **TD learning** | **fixed-point**, $\mathbb R^n$, non-stat | — | bootstrap target | none (no curvature) | contraction | damped | per-value |
| **GAN / extragradient** | **saddle**, $\mathbb R^n$ | — | AD both players | game Jacobian | extragradient/optimistic | — | alternating |
| **iLQR / DDP** | pot, $\mathbb R^n$, $\mathcal S=$ banded | — | AD $g$ | GN | **Riccati** | LM damping | block (time) |
| **SVGD** | pot, **measure** | — | population particles | — | kernel / Wasserstein | — | per-particle |
| **Multigrid–Newton (PDE)** | pot, **function**, multiscale | — | adjoint $g$ | GN | **multigrid** | — | global |

### 8.4 Invariant and seams

Adding a family = adding combinators under a METHOD stage (or a detected fact under
PROBLEM); the selector is never edited. Orthogonality is approximate: quasi-Newton
straddles Sense/Model/Pace; AdaGrad straddles Sense/Pace; clipping straddles
Direct/Sense. A method on a seam just means two combinators co-fire — the stages are a
*generating* vocabulary, not a partition.

---

## 9. Implementation roadmap

Per the build decision (§1.1): **v0–v5 are the conservative core** — the corner of §12,
fully guaranteed and implementable today. **v6–v9 are the relaxations** toward the
generalized system, added in increasing-risk order, with the bounded learned residual
last. Every phase is independently shippable.

**Core — the conservative corner (§12.1):**

- **v0 — kill-test (weeks).** No e-graph. Certificate computes only $\mathcal S$, $L$,
  $\mathcal I$. Hard-coded selection: Muon on matmul, Adam elsewhere, $\eta=c/L$,
  symmetry projection on pre-norm + softmax. Validate §6.3 against tuned AdamW/Muon/SOAP.
- **v1 — atoms + exact e-graph.** Add the atom library (softmax-CE, LayerNorm, linear,
  conv), the operator IR, exact rewrites only (Kron-inverse, transpose, fusion), cost
  extraction. Reproduce §5 derivations and §6.2 exactly. K-FAC/NGD emerge.
- **v2 — approximate e-graph + METHOD Sense/Model/Pace/Compose.** Lossy rewrites with
  certificate-supplied error bounds; the Sense (source/order/VR), Pace (time/adaptivity),
  and Compose (meta) combinators. Adam, momentum, hypergradients, zeroth-order patches
  become reachable. Add the validation loop (meta-eval or online curvature residual).
- **v3 — recurrent/scan + inner-loop.** `scan`/gated-linear-recurrence/low-rank-state
  combinators (SSM, RNN). Inner-loop synthesis for DeltaNet/gated DeltaNet/TTT:
  derive/precondition the fast-weight update on its local regression loss.
- **v4 — full METHOD vocabulary + hardware cost.** Direct (geometry/constraint +
  multigrid/spectral/Schur preconditioners), enriched $\mathcal S$ lattice, Reframe
  (surrogates); hardware-aware cost model (flops/mem/**bits**), closing the loop with the
  Representation combinators in Sense.
- **v5 — new PROBLEM dimensions.** Operator class (extragradient/fixed-point for
  RL/saddles), domain geometry (Riemannian/mirror/Wasserstein for SLAM/control/discrete/
  distribution), evolution (warm-start/incremental, change-trust-region). This is where
  coverage extends past weight-training (§7.3).

**Relaxations — toward the generalized system (§12), in increasing-risk order:**

- **v6 — belief (lowest-risk relaxation).** Replace the binary guard with online,
  confidence-gated empirical estimators (Hutchinson curvature, gradient-noise scale,
  effective rank); soundness becomes the `confidence = 1` special case. **Implement as an
  MHE-style recursive estimator with a covariance, not a bare EMA** (see §12.6): the
  covariance feeds both the confidence guards and the dual-control probe. **Dual-control
  gap:** a greedy structure-aware optimizer can starve its own estimator — the trajectory
  shapes which directions are observable — so v6 must consider excitation/probing. Unlocks
  the empirical-curvature family (Sophia / AdaHessian / Adafactor / GGT). §12.1 axis
  *evidence*.
- **v7 — posterior over surrogates + robust action.** Maintain an ensemble/posterior over
  local models; act pessimistically or posterior-averaged. Unlocks Bayesian-optimization,
  CMA-ES covariance, derivative-free, and DRO. §12.1 axis *surrogate*.
- **v8a — trajectory tracking for non-stationary / saddle problems (validated regime).**
  A predictive tracker (not EMA) over a horizon, gated to `P1 = saddle` / `P3 =
  non-stationary` — the cells where a sibling MHE/MPC project showed real wins (≈7× over
  exact-Hessian Newton on a moving quadratic; a divergent Dirac-GAN made convergent; §12.6).
  **Do the schedule-level version first** (predictive control of per-block lr/momentum
  over a cheap surrogate): near-zero overhead, directly comparable to cosine/warmup.
- **v8b — trajectory control toward a generalization/flatness objective (high-risk,
  unvalidated).** A *low-capacity*, dimensionless-feature outer controller optimizing a
  held-out/flatness proxy. Makes SAM / SWA / warmup / noise-injection / early-stopping
  *reachable* — but credit assignment is RL-hard (reachable ≠ reliably discovered), and the
  sibling result is **silent** on this and raises skepticism for any trajectory relaxation
  on *stationary* training. §12.1 axes *horizon/controller/objective*.
- **v9 — bounded learned residual (endpoint, §12.4).** Attach a *small* learned head that
  models only the unprovable complement of the derived update; the derived bulk keeps its
  guarantees and transfer, and the un-guaranteed behavior stays confined to a bounded
  residual. Only after v6 is mature, since the residual is defined relative to what is
  derived.

**Not building** (per §1.1): a monolithic learned optimizer (no transfer, no guarantees,
astronomical meta-training); data/architecture co-design (out of the optimizer frame);
anything requiring the test distribution (information-theoretically impossible).

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
- **Non-conservative fields (P1).** When the field is not the gradient of a potential
  (RL/TD, saddles, VIs), there is no PSD curvature; the certificate must detect
  monotonicity/contraction and the algebra needs extragradient/fixed-point combinators
  with their own convergence guards. The "$F=$ curvature" assumption of §2.3 is
  potential-only.
- **Non-Euclidean domains (P2).** Manifolds, simplices, lattices, and measure spaces need
  a Riemannian/mirror/Wasserstein layer (retraction, exp-map, transport, mirror map, LP
  oracle, particle discretization). The operator algebra and its rewrites are currently
  written for $\mathbb R^n$; lifting them to these domains soundly is open.
- **Problem evolution (P3).** Drift bounds (how fast the objective/data moves per update)
  are needed to license warm-start/incremental reuse and change-trust-regions; estimating
  them online is unsolved in general.

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

Problem classes beyond weight-training (the new PROBLEM dimensions):
- Schulman et al. *Trust Region Policy Optimization* (TRPO) 2015; *Proximal Policy
  Optimization* (PPO) 2017 — change-trust-region under non-stationarity.
- Korpelevich. *The extragradient method* 1976; Mertikopoulos et al. *Optimistic
  gradient / mirror descent for saddles.* — operator-class (saddle/VI).
- Absil, Mahony, Sepulchre. *Optimization Algorithms on Matrix Manifolds.* 2008 —
  domain = manifold (retraction, transport).
- Liu, Wang. *Stein Variational Gradient Descent.* NeurIPS 2016 — domain = measure.
- Kaess, Ranganathan, Dellaert. *iSAM / iSAM2* — incremental factor-graph optimization.
- Tassa, Erez, Todorov. *iLQR / DDP* — banded/Riccati structure in control.
- Briggs, Henson, McCormick. *A Multigrid Tutorial.* 2000 — multiscale preconditioning.
- Hairer, Lubich, Wanner. *Geometric Numerical Integration* — symplectic /
  structure-preserving updates.

Modern architectures (inner-loop relevance):
- Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*
  (TTT). 2024.
- Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* 2024/2025.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.

---

## 12. Research vision: the relaxation ladder

This section is the **target the core relaxes toward**, not part of the v0–v5 build. Per
§1.1 we implement §2–§11 first; §12 describes the generalized system whose conservative
corner that core *is*. It is included so the §9 relaxation phases (v6–v9) have a
specification.

### 12.1 The ladder: the core is one corner of a larger design space

Each capability the core lacks is a *constraint it deliberately keeps*. Relaxing each
opens an axis; the core sits at the most-constrained (maximally guaranteed, minimally
adaptive) corner.

| Axis | Core corner (tightest) | Relaxes to | Phase |
|---|---|---|---|
| evidence | proof (`confidence = 1`) | belief (any confidence, online) | v6 |
| surrogate | point estimate of $F$ | posterior / ensemble, robust action | v7 |
| horizon | 1 step | trajectory (MPC over a window) | v8 |
| controller | frozen at compile time | online, low-capacity, invariant-feature | v8 |
| objective | per-step fidelity / cost | held-out / flatness / generalization | v8 |

Reading down each column is *sound, amortizable, transferable, less powerful*; reading up
trades guarantees for reach. The relaxations are continuous, so the core is recovered
exactly by pinning every axis to its tightest value (it is a strict special case, not a
different system).

### 12.2 Why relax: the core's three inherent blind spots

The core optimizes *per-step progress on the training objective using provable local
structure*. Its inherent blind spots are the gaps between that and "a good model":

1. **Soundness gap** — it forfeits empirically-real-but-unprovable structure (relaxed by
   *belief*, v6).
2. **Generalization gap** — faster training $\ne$ better model; it has no objective
   rewarding a worse local step that generalizes better (relaxed by *trajectory +
   generalization objective*, v8).
3. **Wrong-object gap** — curvature is irrelevant for black-box/discrete and fragile under
   noise (relaxed by *posterior over surrogates* + non-curvature models, v7).

### 12.3 What each relaxation newly recovers

The methods unlocked are exactly the complement of the blind spots — a coherent set, not
a grab-bag:

- **v6 (belief):** Sophia, AdaHessian, Adafactor, GGT, online empirical Fisher.
- **v7 (posterior):** Bayesian optimization, CMA-ES covariance, derivative-free
  trust-region, DRO / group-DRO.
- **v8 (objective):** SAM/ASAM/GSAM, SWA / weight-EMA, warmup / cosine / one-cycle,
  SGLD / entropy-SGD noise injection, early stopping — *reachable*, with the caveat that
  reliable discovery is RL-hard credit assignment.

Already expressible in the core (the §8 catalog) and merely *re-motivated*, not newly
recovered: momentum, Nesterov, Lion, mirror, FISTA, Frank–Wolfe, Riemannian, SVRG, line
search, lookahead, LARS/LAMB, GaLore.

### 12.4 Endpoint: derive-plus-bounded-residual (v9)

The chosen endpoint is **not** a monolithic learned optimizer. It is a hybrid: the core
*derives* the guaranteed, transferable, provable part of the update, and a **small learned
residual** models only the unprovable complement the belief cannot certify. This closes
the largest learned-optimizer advantage (exploiting unprovable structure) while preserving
the core's efficiency and transfer (the learned part is small — it models a residual, so
low sample complexity and better off-distribution behavior) and confining un-guaranteed
behavior to a bounded correction. One line: *derive what you can prove, learn the residual,
never let the learned part exceed the bounded complement.*

### 12.5 Ceilings no relaxation removes

- **Test-distribution proxy (information-theoretic).** Every system optimizes a *proxy*
  for generalization; true generalization needs the unknown test distribution. v8 buys a
  *better* proxy and *calibrated uncertainty about it* — the real achievable win is
  converting silent misalignment into surfaced, flagged misalignment — but not a guarantee
  of a better model.
- **Coverage vs guarantees.** Maximal coverage (a high-capacity learned function) and
  maximal guarantee (certifiable steps) are mutually constraining; v9 keeps guarantees by
  *bounding* the learned part, accepting less than full coverage.
- **Scope.** Data ordering, augmentation, curriculum, and architecture co-design are out
  of the optimizer frame — a different, larger system.
- **Primitive invention.** The e-graph recombines a registered vocabulary; genuinely new,
  interpretable optimization *principles* must still be added by a human.

### 12.6 Empirical evidence from a sibling project (MHE/MPC)

A separate, hand-built effort implemented a **low-rank-plus-scalar curvature surrogate**
estimated backward over a horizon (**MHE** — moving horizon estimation, a Kalman/fixed-lag
smoother *with covariance*) and used forward (**MPC** — model-predictive control planning
an update sequence). In our terms it is a hand-built instance of **v6 (belief, via
estimation theory) + v8a (trajectory, via control)** on one $\mathcal S$-structure choice
— a partial, control-theoretic build of this section's generalized system. Its
experimental verdict is directly informative:

- **No-go on stationary smooth training.** Trust-region Newton tied or beat the predictive
  method ("no information the predictor lacks"; a stiff-valley test had Newton at ~1e-19
  while the predictor crawled). This **empirically confirms §12.2's claim** that the
  trajectory relaxation collapses to the greedy second-order step on the default cell, and
  it **independently re-derives the §1.1 decision** ("kill the better-Adam-for-normal-
  training framing").
- **Go on non-stationary / saddle problems.** ≈7× over exact-Hessian Newton on a moving
  quadratic; a divergent Dirac-GAN rotation turned convergent. The advantage is
  information-theoretic — a reactive method (even with the exact Hessian) is permanently
  one step behind a moving target. This **confirms the PROBLEM tier**: trajectory
  relaxations pay exactly in `P1 = saddle` / `P3 = non-stationary`, the cells v5 introduces.
- **Gating conditions (import as guards):** the drift must be smooth/low-order, exceed the
  estimation noise (it is a *large-batch* method), use a proper tracker (not EMA), and be
  the *dominant* difficulty; otherwise the advantage vanishes or reverses.

Lessons folded into the plan: (1) **v8 split into v8a (tracking, validated) and v8b
(flatness/generalization, unvalidated and silent in this evidence)** — they are different
phenomena despite both being "trajectory-level"; (2) **v6 should be an MHE-style estimator
with covariance**, not a bare EMA; (3) **dual control is a real gap** — a greedy
structure-aware optimizer starves its own estimator, so v6/v8 must consider excitation
(§6.5 diagnostic); (4) the **baseline discipline** of §6.5 (beat the cheaper version of
each rung; beat plain optimism on games) comes from this project; (5) its **honest novelty
caveat** — the non-stationary method "largely rediscovers optimistic gradient / extragradient
and Kalman/MHE trackers; the value-add is *jointly* modeling curvature and drift plus the
unifying framing" — matches §12.3 exactly and is, in fact, the *intended* value of this
compiler (derive and combine known methods per problem class, automatically), not a threat
to it.

The sharpest takeaway is a **warning that strengthens §1.1**: a sophisticated
estimation-plus-planning optimizer could not beat Newton on the default regime, so be
*more* skeptical that v6–v9 win on ordinary training (including v8b), build the conservative
core so it stands without them, and gate the relaxations to the non-stationary/saddle
niche — which is also the most decisive place to validate the compiler thesis (§6.5 gate).

---

## 13. Duals and inverses

§1–§12 are one corner of a larger design space: *fix the DAG, read its structure, in
reverse mode, estimating backward*. Flipping each of four independent knobs gives a
coherent sibling research direction. This section sketches them as vision, not build.

### 13.1 The expressivity ↔ tractability ladder

Both the inverse problem and the forward-mode direction operate on this ladder. Each rung
gains expressivity by changing the *parameterization* and sheds a specific optimization
property — and the properties detach one at a time, they are not one cliff.

| Class | Retained property | Lost | Mechanism |
|---|---|---|---|
| Linear regression | **closed-form** (one solve) | — | quadratic objective **and** linear-in-params |
| Ridge / GLS | closed-form | — | still quadratic |
| Logistic / GLM | **global convex**, Newton ~quadratic rate | closed-form | linear-in-params, *nonlinear convex* loss |
| Quadratic (any sign) | **one-step Newton** | convexity may be gone | degree-2 objective (order, not curvature sign) |
| Kernel / SVM | global convex | closed-form, cheap Hessian | convex composite in a *fixed* feature space |
| Linear net $W_2W_1$ | **benign nonconvex** (all local min global) | convexity | bilinear-in-params — *no* expressivity gain |
| 2-layer ReLU | convex *reformulation* exists | convexity in natural params | nonlinearity (but see below) |
| NTK / infinite width | **effective convexity**, linear-rate GD | **feature learning** | over-parameterization → lazy regime |
| Finite-width DNN | local structure only | all of the above | finite-width nonlinearity → feature learning |

Three load-bearing observations:

1. **The properties detach.** Closed-form needs quadratic + linear; one-step-Newton needs
   only degree-2 (a nonconvex quadratic is one-step — to a saddle); global-convex needs a
   convex composite; benign-nonconvex (linear nets, matrix factorization) keeps
   all-local-are-global *without* convexity. You shed them individually.
2. **The tension is mediated by parameterization, and the thing fundamentally at odds with
   tractability is *feature learning*.** Linear-in-params → tractable; nonlinear-in-params
   → expressive but hard. Infinite width buys back effective convexity but kills feature
   learning (a tractability ↔ feature-learning trade, not a free lunch). The linear-net rung
   is the cautionary case: convexity lost, expressivity unchanged.
3. **Some lost properties are parameterization artifacts, recoverable by re-representation.**
   2-layer ReLU training has an exact *convex* reformulation (Pilanci–Ergen) — convexity
   was hidden by the weight parameterization, not truly gone. This is the hinge that makes
   the inverse problem real.

### 13.2 The inverse: write the certificate instead of reading it

The forward project (§1–§12) is: *fix the DAG → **read** its certificate → synthesize the
optimizer that copes.* The DAG's rung on the ladder is given.

The **inverse** is an *architecture compiler*: *fix a tractability target (a certificate
property to hold) → search DAGs / re-parameterizations for the most expressive one that
**realizes** it.* Same certificate, opposite causality — the optimizer compiler reads it
as an observation; the architecture compiler writes it as a specification.

And just as the forward compiler *recovers* SGD/Adam/K-FAC/NGD, the inverse compiler
*recovers* existing architecture tricks as its frontier:

| Target property | Recovered architecture method |
|---|---|
| closed-form readout | random features / extreme-learning-machines / reservoir / **linear probing** (frozen body + exact head) |
| global convex | Input-Convex Neural Networks; **convex reformulations** (ReLU duality) |
| benign nonconvex | linear bottlenecks, over-parameterized / deep-linear designs, matrix factorization |
| effective convex | very wide (NTK/lazy) layers — accepting no feature learning |

**Co-design** is the join — search DAG *and* optimizer together — with the certificate as a
**shared contract**: the architecture compiler writes a property, the optimizer compiler
reads and exploits it, and the two negotiate the (expressivity × tractability ×
convergence-rate) frontier. The original project is co-design with the architecture side
pinned. This corner directly attacks the deepest tension on the ladder (feature learning
vs tractability) instead of taking the DAG as fixed and coping — the highest-value sibling.

### 13.3 The forward-mode dual

Reverse mode is cheap for *few outputs, many inputs* (a scalar loss over millions of
params) and rides a stored tape — right for bulk training. Forward mode is cheap for *many
outputs, few inputs* and is *tapeless and causal*. Flipping to forward mode is not
symmetric; it relocates the project to a different niche:

- **Tapeless / streaming.** Forward gradient (one JVP along a random direction, unbiased,
  no backward pass, O(1) depth memory) is the substrate for online/streaming/real-time
  training — the same **non-stationary niche** §12.6 found wins in. Forward mode is the
  streaming-friendly differentiation.
- **The certificate flips polarity.** A forward certificate is a *forward* abstract
  interpretation — sensitivity / Lipschitz / **reachability** ("which directions can I
  steer"), dual to the backward certificate's influence / **observability** ("which
  directions can I see"). This is the same observability ↔ controllability duality of the
  MHE/MPC memo, reappearing as reverse ↔ forward AD.
- **Cheap schedule / hyperparameter control.** Forward-mode hypergradients
  (`d(loss)/d(few hyperparameters)`) are cheap precisely *because* the input count is
  small — making forward mode the natural substrate for the scheduler variant (§12.6's
  cheapest, most-deployable first win).

### 13.4 The lattice of duals

| Knob | This project | Sibling |
|---|---|---|
| fixed side | optimizer-searched | DAG-searched (§13.2) / both (co-design) |
| certificate causality | **read** (observe a fixed DAG) | **write** (specify the DAG) |
| AD direction | reverse (training, observability) | forward (streaming/control, reachability; §13.3) |
| time direction | backward estimate (MHE) | forward plan (MPC) — §12.6 |

This project is the corner *{optimizer-searched, certificate-read, reverse-mode,
backward-estimate}*. The richest single new direction is the **co-design corner with a
shared-certificate contract** (§13.2): an architecture compiler that writes tractability
properties and an optimizer compiler that reads and exploits them, jointly navigating the
expressivity–tractability frontier rather than taking the architecture as fixed.