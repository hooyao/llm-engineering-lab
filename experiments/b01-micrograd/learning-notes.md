# B1 — micrograd: backprop on scalars — learning notes

Reference notes (fact-based, no Q&A). Learner is doing Track B from the start as
review / gap-fill. Everything here is scalar — micrograd has zero tensors, zero
dims (Karpathy strips tensors so the autograd skeleton is bare). This is the one
day with no shapes to track.

What B1 adds beyond Track A: Track A USED `loss.backward()` as a black box. B1 builds
that black box by hand — a ~100-line `Value` class with `__add__`, `__mul__`,
`backward()`. The gap it fills is the mechanics BETWEEN calling `.backward()` and
`.grad` being filled: the 4 steps below.

---

## Notation facts (read once, applies to everything below)

1. **`d` is the derivative OPERATOR, not a number and not division.** `de/da` is one
   indivisible symbol meaning "derivative of e w.r.t. a" = a MULTIPLIER: push a a
   little, how many times as much does e move. Never split it into `de / da`.
   (In this example a node also happens to be named `d` — pure letter clash, unrelated.)

2. **`de/da` is NOT `e/a`.** `e/a` is a static ratio; `de/da` is a rate of change.
   They coincide only for trivial forms. Proof they differ:

        e = a*b,  a=2, b=-3, e=-6:   e/a = -3.0,   de/da = b  = -3.0   (equal — coincidence)
        e = a*a,  a=2,       e=4:    e/a =  2.0,   de/da = 2a =  4.0   (DIFFERENT)

3. **One line often carries two equals signs: the RULE then the PLUGGED-IN value.**

        dL/dd = f = 2.0
              |   |
              |   +-- f's forward value (the number)
              +------ the derivative rule for multiply (still symbolic)

   Read as: "dL/dd equals f by the multiply rule; f's forward value is 2.0."

4. **Two local-derivative rules cover the whole graph** (no need to memorize each edge):

        multiply  y = u*v  ->  dy/du = v,  dy/dv = u   (= the OTHER input; reads a forward value)
        add       y = u+v  ->  dy/du = 1,  dy/dv = 1   (constant 1; reads no forward value)

   Why multiply gives "the other input": in `e = a*b`, hold b fixed, then `e = a*(-3.0)`
   is a line of slope -3.0 = b; so de/da = b. Linear-regression anchor: `y = w*x`,
   push x -> y moves w times -> dy/dx = w.

5. **A third rule once activations enter: `tanh`** (added at the end of B1). `tanh`
   squashes any real number into (-1, 1); it is the non-linearity that makes stacked
   neurons more than one big linear function. It is a SINGLE-input op, so it has one
   local-derivative rule, and that rule reads the node's own OUTPUT:

        tanh    o = tanh(x)  ->  do/dx = 1 - o^2   (uses the output o, not the input x)

   Sanity: x large -> o ~ 1 -> do/dx ~ 0 (flat tails); x = 0 -> o = 0 -> do/dx = 1
   (steepest in the middle). Same "derivative expressed via the output" shape as A2's
   sigmoid `sig'(x) = sig(x)*(1 - sig(x))`. In the graph a tanh node has one arrow in
   and one arrow out, edge label `1 - o^2`.

---

## The worked example (used in Steps 1–3)

    a = 2.0
    b = -3.0
    e = a * b        # e = -6.0
    c = 10.0
    d = e + c        # d = 4.0
    f = 2.0
    L = d * f        # L = 8.0      <- the "loss"

---

## Reverse-mode autodiff = 4 steps, in order

### Step 1 — forward pass computes values AND records each op as a node

Running the code left-to-right produces the numbers and, at the same time, builds a
**computation graph**: every operation becomes a node that remembers its inputs and its
op. PyTorch builds this same graph invisibly on every op involving a `requires_grad=True`
tensor.

Values produced: `e = -6.0`, `d = 4.0`, `L = 8.0`.

```mermaid
flowchart LR
    a["a = 2.0"]:::inp
    b["b = -3.0"]:::inp
    c["c = 10.0"]:::inp
    f["f = 2.0"]:::inp
    e["e = a*b = -6.0"]:::op
    d["d = e+c = 4.0"]:::op
    L["L = d*f = 8.0"]:::op
    a --> e
    b --> e
    e --> d
    c --> d
    d --> L
    f --> L
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef op  fill:#fff3e0,stroke:#fb8c00,color:#111
```

Blue = inputs (nobody computes them). Orange = computed nodes (label shows how).

### Step 2 — each node knows its LOCAL derivative (output w.r.t. its direct inputs only)

"Local" = only the one op, ignoring everything up/downstream. Apply the two rules from
notation fact 4. Multiply edges carry a forward value; add edges are the constant 1.

    L = d * f  (multiply):   dL/dd = f = 2.0,    dL/df = d = 4.0
    d = e + c  (add):        dd/de = 1,          dd/dc = 1
    e = a * b  (multiply):   de/da = b = -3.0,   de/db = a = 2.0

Same graph, SAME left-to-right layout as Step 1 (a on the left, L on the right) so you can
match it node-for-node. Each op node also carries its two local rules inside it; edge label
= that edge's local derivative. Multiply nodes read forward values; the add node (d) does not.

    values:  a=2.0   b=-3.0   c=10.0   f=2.0     e=a*b=-6.0   d=e+c=4.0   L=d*f=8.0

```mermaid
flowchart LR
    a["a = 2.0"]:::inp
    b["b = -3.0"]:::inp
    c["c = 10.0"]:::inp
    f["f = 2.0"]:::inp
    e["e = a*b = -6.0<br/>de/da = b = -3.0<br/>de/db = a = 2.0"]:::op
    d["d = e+c = 4.0<br/>dd/de = 1<br/>dd/dc = 1"]:::op
    L["L = d*f = 8.0<br/>dL/dd = f = 2.0<br/>dL/df = d = 4.0"]:::op
    a -->|"de/da = b = -3.0"| e
    b -->|"de/db = a = 2.0"| e
    e -->|"dd/de = 1"| d
    c -->|"dd/dc = 1"| d
    d -->|"dL/dd = f = 2.0"| L
    f -->|"dL/df = d = 4.0"| L
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef op  fill:#fff3e0,stroke:#fb8c00,color:#111
```

### Step 3 — chain rule: multiply local derivatives ALONG a path to get global dL/dvar

The global gradient of a variable = (global gradient of its child) × (local derivative
of child w.r.t. the variable). Start by seeding the output with `dL/dL = 1`, then walk
backward, multiplying in each local derivative.

    dL/dL = 1                          (seed)
    dL/dd = dL/dL * (dL/dd local) = 1 * f   = 1 * 2.0  = 2.0
    dL/df = dL/dL * (dL/df local) = 1 * d   = 1 * 4.0  = 4.0
    dL/de = dL/dd * (dd/de local) = 2.0 * 1            = 2.0
    dL/dc = dL/dd * (dd/dc local) = 2.0 * 1            = 2.0
    dL/da = dL/de * (de/da local) = 2.0 * b = 2.0 * -3.0 = -6.0
    dL/db = dL/de * (de/db local) = 2.0 * a = 2.0 * 2.0  = 4.0

Backward graph — SAME left-to-right layout as Steps 1 and 2 (a on the left, L on the
right). Only the node CONTENTS change: each node now shows the chain-rule product that
produces its global gradient. The `d` node (add) is included. Edge label = the local
derivative multiplied in as gradient flows from L back toward the inputs.

    values:  a=2.0   b=-3.0   c=10.0   f=2.0     e=-6.0   d=4.0   L=8.0

```mermaid
flowchart LR
    a["a<br/>dL/da = dL/de × de/da<br/>= 2.0 × b = 2.0 × -3.0<br/>= -6.0"]:::g
    b["b<br/>dL/db = dL/de × de/db<br/>= 2.0 × a = 2.0 × 2.0<br/>= 4.0"]:::g
    c["c<br/>dL/dc = dL/dd × dd/dc<br/>= 2.0 × 1<br/>= 2.0"]:::g
    f["f<br/>dL/df = dL/dL × dL/df<br/>= 1 × d = 1 × 4.0<br/>= 4.0"]:::g
    e["e<br/>dL/de = dL/dd × dd/de<br/>= 2.0 × 1<br/>= 2.0"]:::g
    d["d<br/>dL/dd = dL/dL × dL/dd<br/>= 1 × f = 1 × 2.0<br/>= 2.0"]:::g
    L["L<br/>dL/dL = 1<br/>(seed)"]:::seed
    a -->|"de/da = b = -3.0"| e
    b -->|"de/db = a = 2.0"| e
    e -->|"dd/de = 1"| d
    c -->|"dd/dc = 1"| d
    d -->|"dL/dd = f = 2.0"| L
    f -->|"dL/df = d = 4.0"| L
    classDef seed fill:#e6f4ea,stroke:#34a853,color:#111
    classDef g fill:#fce8e6,stroke:#ea4335,color:#111
```

Read each node as: (global gradient of my child) × (local derivative of child w.r.t. me).
The add node d passes gradient through unchanged (× 1), so dL/de = dL/dc = dL/dd = 2.0.
Arrows still point a -> e -> d -> L (forward direction); the gradient VALUE travels the
other way, L back to a, which is why each node's formula reads its child's gradient.

### Step 4 — walk in REVERSE topological order, ACCUMULATE grad where a node fans out

Two mechanical rules that make Step 3 correct in general:

**Reverse topological order.** A node's global gradient can only be finalized after ALL
nodes that consume it are done (each consumer contributes part of the gradient). So you
process the graph output-first: `L` -> `{d, f}` -> `{e, c}` -> `{a, b}`. Processing a
node before its consumers would use an incomplete gradient.

**Accumulate at fan-out.** If one node feeds MORE THAN ONE downstream node, its gradient
is the SUM of the contributions along each outgoing path (`+=`, not `=`). The main
example has no fan-out (every node is used once), so accumulation is trivial there.
A second minimal example where `x` fans out to two nodes:

    x = 3.0
    p = x * 2        # p = 6.0
    q = x + 5        # q = 8.0
    L = p + q        # L = 14.0

    dL/dp = 1,  dL/dq = 1
    path through p:  dL/dp * dp/dx = 1 * 2 = 2
    path through q:  dL/dq * dq/dx = 1 * 1 = 1
    dL/dx = 2 + 1 = 3      <- the two paths are SUMMED

Same left-to-right layout (x on the left, L on the right). Node contents show the
per-path products; x sums the two paths.

    values:  x=3.0     p=x*2=6.0     q=x+5=8.0     L=p+q=14.0

```mermaid
flowchart LR
    x["x<br/>dL/dx = (via p) + (via q)<br/>= dL/dp×dp/dx + dL/dq×dq/dx<br/>= 1×2 + 1×1<br/>= 3.0"]:::acc
    p["p = x*2 = 6.0<br/>dL/dp = 1"]:::g
    q["q = x+5 = 8.0<br/>dL/dq = 1"]:::g
    L["L = p+q = 14.0<br/>dL/dL = 1 (seed)"]:::seed
    x -->|"dp/dx = 2"| p
    x -->|"dq/dx = 1"| q
    p -->|"dL/dp = 1"| L
    q -->|"dL/dq = 1"| L
    classDef seed fill:#e6f4ea,stroke:#34a853,color:#111
    classDef g fill:#fce8e6,stroke:#ea4335,color:#111
    classDef acc fill:#fef7e0,stroke:#f9ab00,color:#111
```

Two arrows leave x (it fans out to p and q); their gradient contributions (2 and 1) are
SUMMED to 3.0. In code this is why each `Value` does `self.grad += ...` inside `backward()`,
and why `.grad` must be zeroed between steps (`optimizer.zero_grad()` in Track A) —
otherwise the `+=` keeps piling on across iterations.

---

## How the 4 steps become the `Value` class (the B1 deliverable)

- Step 1: every `__add__` / `__mul__` / `tanh` returns a new `Value` that stores its
  parent `Value`s and a local `_backward` closure — that IS recording the node.
- Step 2: the `_backward` closure encodes the local rule (multiply reads the siblings'
  `.data`; add uses 1; tanh reads its own output `1 - out^2`).
- Step 3+4: `Value.backward()` topologically sorts the graph, seeds `self.grad = 1`, and
  calls each node's `_backward` in reverse order, each doing `parent.grad += local * self.grad`.

The implementation is in `micrograd.py`; run it to see all four demos (worked example,
fan-out accumulation, torch cross-check, and a trained neuron).

---

## Extending Steps 1–4 with activation (tanh)

Everything above uses only `*` and `+`. Adding `tanh` changes NOTHING about the 4-step
machine — it just adds one more op with one more local rule. This section shows the same
4 steps on a tiny neuron so activation is not a special case, just a third node type.

**Worked example — one neuron** (this is `micrograd.py` demo 4, first forward pass):

    x1 = 1.0    w1 = -0.5       # inputs x fixed; weights w and bias b are parameters
    x2 = -2.0   w2 = 0.8
    b  = 0.1
    m1 = w1*x1 = -0.5
    m2 = w2*x2 = -1.6
    z  = m1 + m2 + b = -2.0     # the linear part: w.x + b
    o  = tanh(z) = -0.964       # the activation (non-linearity)

### Step 1 (with activation) — forward builds the graph, now with a tanh node

The multiply/add nodes are as before; `tanh` adds ONE single-input node at the end.

```mermaid
flowchart LR
    x1["x1 = 1.0"]:::inp
    w1["w1 = -0.5"]:::inp
    x2["x2 = -2.0"]:::inp
    w2["w2 = 0.8"]:::inp
    b["b = 0.1"]:::inp
    m1["m1 = w1*x1 = -0.5"]:::op
    m2["m2 = w2*x2 = -1.6"]:::op
    s["s = m1+m2 = -2.1"]:::op
    z["z = s+b = -2.0"]:::op
    o["o = tanh(z) = -0.964"]:::act
    w1 --> m1
    x1 --> m1
    w2 --> m2
    x2 --> m2
    m1 --> s
    m2 --> s
    s --> z
    b --> z
    z --> o
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef op  fill:#fff3e0,stroke:#fb8c00,color:#111
    classDef act fill:#f3e8fd,stroke:#a142f4,color:#111
```

Purple = the activation node. It has exactly one input arrow (z) and one output (o).

### Step 2 (with activation) — the tanh node's local derivative

The new node's local rule reads its OWN output, not an input:

    o = tanh(z)  (activation):   do/dz = 1 - o^2 = 1 - (-0.964)^2 = 0.0708

Same graph, edge labels = local derivatives; the tanh edge carries `1 - o^2`:

```mermaid
flowchart LR
    x1["x1 = 1.0"]:::inp
    w1["w1 = -0.5"]:::inp
    x2["x2 = -2.0"]:::inp
    w2["w2 = 0.8"]:::inp
    b["b = 0.1"]:::inp
    m1["m1 = w1*x1 = -0.5<br/>dm1/dw1 = x1 = 1.0<br/>dm1/dx1 = w1 = -0.5"]:::op
    m2["m2 = w2*x2 = -1.6<br/>dm2/dw2 = x2 = -2.0<br/>dm2/dx2 = w2 = 0.8"]:::op
    s["s = m1+m2 = -2.1<br/>ds/dm1 = 1<br/>ds/dm2 = 1"]:::op
    z["z = s+b = -2.0<br/>dz/ds = 1<br/>dz/db = 1"]:::op
    o["o = tanh(z) = -0.964<br/>do/dz = 1 - o^2 = 0.0708"]:::act
    w1 -->|"= x1 = 1.0"| m1
    x1 -->|"= w1 = -0.5"| m1
    w2 -->|"= x2 = -2.0"| m2
    x2 -->|"= w2 = 0.8"| m2
    m1 -->|"= 1"| s
    m2 -->|"= 1"| s
    s -->|"= 1"| z
    b -->|"= 1"| z
    z -->|"= 1 - o^2 = 0.0708"| o
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef op  fill:#fff3e0,stroke:#fb8c00,color:#111
    classDef act fill:#f3e8fd,stroke:#a142f4,color:#111
```

### Step 3 (with activation) — chain rule through the tanh

Seed do/do = 1, walk back. The tanh contributes its `1 - o^2 = 0.0708` factor first; then
the add nodes pass gradient straight through (x 1), and the multiply nodes give "the other
input". This is what makes the gradient small here: tanh is saturated (z = -2.0, near the
flat tail), so `do/dz` is only 0.0708 and every upstream gradient is scaled by it.

    values:  x1=1.0  w1=-0.5   x2=-2.0  w2=0.8   b=0.1   z=-2.0   o=-0.964

```mermaid
flowchart LR
    w1["w1<br/>do/dw1 = do/dm1 x dm1/dw1<br/>= 0.0708 x 1.0<br/>= 0.0708"]:::g
    x1["x1<br/>do/dx1 = 0.0708 x w1<br/>= 0.0708 x -0.5<br/>= -0.0354"]:::g
    w2["w2<br/>do/dw2 = do/dm2 x dm2/dw2<br/>= 0.0708 x -2.0<br/>= -0.1416"]:::g
    x2["x2<br/>do/dx2 = 0.0708 x w2<br/>= 0.0708 x 0.8<br/>= 0.0566"]:::g
    b["b<br/>do/db = do/dz x dz/db<br/>= 0.0708 x 1<br/>= 0.0708"]:::g
    m1["m1<br/>do/dm1 = do/ds x 1<br/>= 0.0708"]:::g
    m2["m2<br/>do/dm2 = do/ds x 1<br/>= 0.0708"]:::g
    s["s<br/>do/ds = do/dz x 1<br/>= 0.0708"]:::g
    z["z<br/>do/dz = do/do x (1-o^2)<br/>= 1 x 0.0708<br/>= 0.0708"]:::act
    o["o<br/>do/do = 1 (seed)"]:::seed
    w1 -->|"= x1 = 1.0"| m1
    x1 -->|"= w1 = -0.5"| m1
    w2 -->|"= x2 = -2.0"| m2
    x2 -->|"= w2 = 0.8"| m2
    m1 -->|"= 1"| s
    m2 -->|"= 1"| s
    s -->|"= 1"| z
    b -->|"= 1"| z
    z -->|"= 1 - o^2 = 0.0708"| o
    classDef seed fill:#e6f4ea,stroke:#34a853,color:#111
    classDef g fill:#fce8e6,stroke:#ea4335,color:#111
    classDef act fill:#f3e8fd,stroke:#a142f4,color:#111
```

### Step 4 (with activation) — reverse order + accumulate, then the training step

Reverse topological order here is o -> z -> s -> {m1, m2} -> {inputs}. No fan-out in this
single neuron, so no accumulation is needed inside one backward pass — BUT across training
steps the `+=` still requires zeroing grads each step, exactly as in Track A. The training
loop (demo 4) closes B1:

    for step in range(20):
        forward:  z = w1*x1 + w2*x2 + b ; o = tanh(z)
        loss   =  (o - target)^2
        zero grads on w1, w2, b            # Step 4: or += accumulates across steps
        loss.backward()                    # our engine fills every .grad
        w -= lr * w.grad                   # gradient descent nudges params down-gradient

Result (demo 4): out climbs -0.964 -> +0.914 toward target +1.0; loss 3.86 -> 0.007.
A real neuron, trained end to end by our own backward() — three rules (multiply, add,
tanh) are enough. That is the whole point of B1.

---

## Connecting neurons — two hidden layers with two neurons each

The previous section used one neuron:

    z = w1*x1 + w2*x2 + b       # pre-activation: one scalar
    h = activation(z)           # this neuron's output: one scalar

To connect neurons, the next layer uses the previous layer's `h` values as its inputs.
The direction is:

    input scalars -> hidden layer 1 outputs -> hidden layer 2 outputs -> output -> loss

One neuron still produces one scalar. Two neurons in a layer therefore produce two
scalars, and each neuron in the next layer can read both of them. In `Value` terms,
every scalar below is one `Value`; a neuron is a small subgraph made from several
`Value` objects and operations. A `Value` is not itself a neuron.

This example uses ReLU only to keep every number integral. ReLU is one more local op:

    h = ReLU(z) = max(0, z)
    dh/dz = 1 when z > 0; 0 when z < 0

Every `z` below is positive, so every ReLU local derivative is `1`. As in the earlier
worked examples, the diagrams keep the same left-to-right layout in every step. Only
the contents change from forward values, to local derivatives, to global gradients.

To keep the full network readable, one box below represents one neuron's complete
scalar subgraph (`multiply -> add -> ReLU`). The `Value` engine still records every
individual multiply, add, and ReLU node inside that box.

### Step 1 — forward pass builds one connected graph

The network has two hidden layers, each with two neurons, followed by one output
neuron. An edge label is the weight used to connect one neuron's output to the next
neuron:

```mermaid
flowchart LR
    x1["x1 = 1"]:::inp
    x2["x2 = 2"]:::inp
    n11["hidden 1, neuron 1<br/>z1_1 = 3<br/>h1_1 = ReLU(z1_1) = 3"]:::h1
    n12["hidden 1, neuron 2<br/>z1_2 = 4<br/>h1_2 = ReLU(z1_2) = 4"]:::h1
    n21["hidden 2, neuron 1<br/>z2_1 = 7<br/>h2_1 = ReLU(z2_1) = 7"]:::h2
    n22["hidden 2, neuron 2<br/>z2_2 = 10<br/>h2_2 = ReLU(z2_2) = 10"]:::h2
    y["output<br/>y = 17"]:::out
    L["loss<br/>L = 0.5"]:::loss
    x1 -->|"u11 = 1"| n11
    x2 -->|"u12 = 1"| n11
    x1 -->|"u21 = 2"| n12
    x2 -->|"u22 = 1"| n12
    n11 -->|"v11 = 1"| n21
    n12 -->|"v12 = 1"| n21
    n11 -->|"v21 = 2"| n22
    n12 -->|"v22 = 1"| n22
    n21 -->|"q1 = 1"| y
    n22 -->|"q2 = 1"| y
    y -->|"target = 16"| L
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef h1 fill:#fff3e0,stroke:#fb8c00,color:#111
    classDef h2 fill:#f3e8fd,stroke:#a142f4,color:#111
    classDef out fill:#e6f4ea,stroke:#34a853,color:#111
    classDef loss fill:#fce8e6,stroke:#ea4335,color:#111
```

The four arrows between the two hidden layers are the important part. `h1_1` feeds
both neurons in hidden layer 2, and `h1_2` also feeds both. Backprop must therefore
add two path contributions into each of `h1_1.grad` and `h1_2.grad`.

#### Forward values, expanded

Use `u` for hidden-layer-1 weights, `v` for hidden-layer-2 weights, and `q` for
output weights. The first index selects a neuron; the second selects that neuron's
input. All biases are zero in the forward pass, but they remain independent trainable
parameters and receive gradients.

    x1 = 1
    x2 = 2

    # Hidden layer 1, neuron 1: u11=1, u12=1, bias1_1=0
    z1_1 = u11*x1 + u12*x2 + bias1_1
         = 1*1 + 1*2 + 0 = 3
    h1_1 = ReLU(z1_1) = 3

    # Hidden layer 1, neuron 2: u21=2, u22=1, bias1_2=0
    z1_2 = u21*x1 + u22*x2 + bias1_2
         = 2*1 + 1*2 + 0 = 4
    h1_2 = ReLU(z1_2) = 4

    # Hidden layer 2, neuron 1: v11=1, v12=1, bias2_1=0
    z2_1 = v11*h1_1 + v12*h1_2 + bias2_1
         = 1*3 + 1*4 + 0 = 7
    h2_1 = ReLU(z2_1) = 7

    # Hidden layer 2, neuron 2: v21=2, v22=1, bias2_2=0
    z2_2 = v21*h1_1 + v22*h1_2 + bias2_2
         = 2*3 + 1*4 + 0 = 10
    h2_2 = ReLU(z2_2) = 10

    # Output neuron: q1=1, q2=1, bias_out=0
    y = q1*h2_1 + q2*h2_2 + bias_out
      = 1*7 + 1*10 + 0 = 17

    target = 16
    L = 0.5 * (y - target)^2
      = 0.5 * 1^2 = 0.5

The layer boundary does not create a special operation. `h1_1` and `h1_2` are simply
the parent `Value` objects used while constructing `z2_1` and `z2_2`. All of these
operations become one computation graph rooted at `L`.

### Step 2 — each connection has a local derivative

For a neuron `z = w1*a1 + w2*a2 + b`, the local derivative with respect to one input
activation is the weight on that connection:

    dz/da1 = w1
    dz/da2 = w2

ReLU adds a factor `dh/dz`. Every `z` in this example is positive, so that factor is
`1`. The edge labels below are therefore the local multipliers that backward will use.
The loss node has its own local rule, `dL/dy = y - target = 1`.

```mermaid
flowchart LR
    x1["x1 = 1"]:::inp
    x2["x2 = 2"]:::inp
    n11["z1_1 = 3; h1_1 = 3<br/>dh1_1/dz1_1 = 1"]:::op
    n12["z1_2 = 4; h1_2 = 4<br/>dh1_2/dz1_2 = 1"]:::op
    n21["z2_1 = 7; h2_1 = 7<br/>dh2_1/dz2_1 = 1"]:::act
    n22["z2_2 = 10; h2_2 = 10<br/>dh2_2/dz2_2 = 1"]:::act
    y["y = 17"]:::out
    L["L = 0.5<br/>dL/dy = y-target = 1"]:::loss
    x1 -->|"dz1_1/dx1 = u11 = 1"| n11
    x2 -->|"dz1_1/dx2 = u12 = 1"| n11
    x1 -->|"dz1_2/dx1 = u21 = 2"| n12
    x2 -->|"dz1_2/dx2 = u22 = 1"| n12
    n11 -->|"dz2_1/dh1_1 = v11 = 1"| n21
    n12 -->|"dz2_1/dh1_2 = v12 = 1"| n21
    n11 -->|"dz2_2/dh1_1 = v21 = 2"| n22
    n12 -->|"dz2_2/dh1_2 = v22 = 1"| n22
    n21 -->|"dy/dh2_1 = q1 = 1"| y
    n22 -->|"dy/dh2_2 = q2 = 1"| y
    y -->|"dL/dy = 1"| L
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef op fill:#fff3e0,stroke:#fb8c00,color:#111
    classDef act fill:#f3e8fd,stroke:#a142f4,color:#111
    classDef out fill:#e6f4ea,stroke:#34a853,color:#111
    classDef loss fill:#fce8e6,stroke:#ea4335,color:#111
```

### Step 3 — chain rule computes global gradients

#### Start at the loss

For the squared-error loss:

    dL/dy = y - target = 17 - 16 = 1

The output neuron sends this gradient to both hidden-layer-2 outputs:

    dL/dh2_1 = dL/dy * dy/dh2_1 = 1 * q1 = 1
    dL/dh2_2 = dL/dy * dy/dh2_2 = 1 * q2 = 1

Both pre-activations are positive, so ReLU passes the gradient through unchanged:

    dL/dz2_1 = dL/dh2_1 * dh2_1/dz2_1 = 1 * 1 = 1
    dL/dz2_2 = dL/dh2_2 * dh2_2/dz2_2 = 1 * 1 = 1

#### The key step — sum gradient contributions at fan-out

`h1_1` affects the loss through both hidden-layer-2 neurons:

    path 1: h1_1 -> z2_1 -> ... -> L
    path 2: h1_1 -> z2_2 -> ... -> L

Its global gradient is the sum of those paths:

    dL/dh1_1
      = dL/dz2_1 * dz2_1/dh1_1
      + dL/dz2_2 * dz2_2/dh1_1
      = 1 * v11 + 1 * v21
      = 1 * 1   + 1 * 2
      = 3

`h1_2` also has two paths:

    dL/dh1_2
      = dL/dz2_1 * dz2_1/dh1_2
      + dL/dz2_2 * dz2_2/dh1_2
      = 1 * v12 + 1 * v22
      = 1 * 1   + 1 * 1
      = 2

The graph below has exactly the same topology as Steps 1 and 2. Arrows remain in the
forward direction; gradient values travel from right to left. Each yellow node contains
a sum because its activation has two outgoing forward paths.

```mermaid
flowchart LR
    x1["x1 = 1<br/>dL/dx1 = 3*u11 + 2*u21<br/>= 3*1 + 2*2 = 7"]:::acc
    x2["x2 = 2<br/>dL/dx2 = 3*u12 + 2*u22<br/>= 3*1 + 2*1 = 5"]:::acc
    h11["z1_1 = 3; h1_1 = 3<br/>dL/dh1_1 = 1*v11 + 1*v21 = 3<br/>dL/dz1_1 = 3*ReLU'(3) = 3"]:::acc
    h12["z1_2 = 4; h1_2 = 4<br/>dL/dh1_2 = 1*v12 + 1*v22 = 2<br/>dL/dz1_2 = 2*ReLU'(4) = 2"]:::acc
    z21["z2_1 = 7; h2_1 = 7<br/>dL/dh2_1 = 1*q1 = 1<br/>dL/dz2_1 = 1*ReLU'(7) = 1"]:::g
    z22["z2_2 = 10; h2_2 = 10<br/>dL/dh2_2 = 1*q2 = 1<br/>dL/dz2_2 = 1*ReLU'(10) = 1"]:::g
    y["y = 17<br/>dL/dy = 1"]:::g
    L["L = 0.5<br/>dL/dL = 1 (seed)"]:::seed
    x1 -->|"u11 = 1"| h11
    x2 -->|"u12 = 1"| h11
    x1 -->|"u21 = 2"| h12
    x2 -->|"u22 = 1"| h12
    h11 -->|"dz2_1/dh1_1 = v11 = 1"| z21
    h12 -->|"dz2_1/dh1_2 = v12 = 1"| z21
    h11 -->|"dz2_2/dh1_1 = v21 = 2"| z22
    h12 -->|"dz2_2/dh1_2 = v22 = 1"| z22
    z21 -->|"q1 = 1"| y
    z22 -->|"q2 = 1"| y
    y -->|"dL/dy = 1"| L
    classDef seed fill:#e6f4ea,stroke:#34a853,color:#111
    classDef g fill:#fce8e6,stroke:#ea4335,color:#111
    classDef acc fill:#fef7e0,stroke:#f9ab00,color:#111
```

Because the first-layer pre-activations are also positive, their ReLU derivatives are
`1`:

    dL/dz1_1 = dL/dh1_1 * 1 = 3
    dL/dz1_2 = dL/dh1_2 * 1 = 2

### Step 4 — reverse order, accumulate, and fill parameter gradients

The reverse-topological walk is:

    L -> y -> {hidden-2 neuron 1, hidden-2 neuron 2}
      -> {hidden-1 neuron 1, hidden-1 neuron 2} -> {x1, x2}

Both hidden-layer-2 neurons must run before either `h1_1.grad` or `h1_2.grad` is
complete. Their `_backward` closures contribute into the same parent with `+=`; this
is exactly how the two path sums in Step 3 are produced.

#### Local multiply rule fills each neuron's parameters

Once a neuron's `dL/dz` is known, each incoming weight gradient is:

    weight.grad = dL/dz * that weight's input value

For hidden layer 2:

    dL/dv11 = dL/dz2_1 * h1_1 = 1 * 3 = 3
    dL/dv12 = dL/dz2_1 * h1_2 = 1 * 4 = 4
    dL/dv21 = dL/dz2_2 * h1_1 = 1 * 3 = 3
    dL/dv22 = dL/dz2_2 * h1_2 = 1 * 4 = 4
    dL/dbias2_1 = 1
    dL/dbias2_2 = 1

For hidden layer 1:

    dL/du11 = dL/dz1_1 * x1 = 3 * 1 = 3
    dL/du12 = dL/dz1_1 * x2 = 3 * 2 = 6
    dL/du21 = dL/dz1_2 * x1 = 2 * 1 = 2
    dL/du22 = dL/dz1_2 * x2 = 2 * 2 = 4
    dL/dbias1_1 = 3
    dL/dbias1_2 = 2

For the output neuron:

    dL/dq1 = dL/dy * h2_1 = 1 * 7  = 7
    dL/dq2 = dL/dy * h2_2 = 1 * 10 = 10
    dL/dbias_out = 1

The final diagram again keeps the same network topology. Each neuron box now shows the
gradients of the trainable parameters inside that neuron:

```mermaid
flowchart LR
    x1["x1 = 1"]:::inp
    x2["x2 = 2"]:::inp
    n11["hidden 1, neuron 1<br/>u11.grad = 3<br/>u12.grad = 6<br/>bias1_1.grad = 3"]:::p1
    n12["hidden 1, neuron 2<br/>u21.grad = 2<br/>u22.grad = 4<br/>bias1_2.grad = 2"]:::p1
    n21["hidden 2, neuron 1<br/>v11.grad = 3<br/>v12.grad = 4<br/>bias2_1.grad = 1"]:::p2
    n22["hidden 2, neuron 2<br/>v21.grad = 3<br/>v22.grad = 4<br/>bias2_2.grad = 1"]:::p2
    y["output neuron<br/>q1.grad = 7<br/>q2.grad = 10<br/>bias_out.grad = 1"]:::out
    L["L = 0.5<br/>backward seed = 1"]:::loss
    x1 --> n11
    x2 --> n11
    x1 --> n12
    x2 --> n12
    n11 --> n21
    n12 --> n21
    n11 --> n22
    n12 --> n22
    n21 --> y
    n22 --> y
    y --> L
    classDef inp fill:#e8f0fe,stroke:#4285f4,color:#111
    classDef p1 fill:#fff3e0,stroke:#fb8c00,color:#111
    classDef p2 fill:#f3e8fd,stroke:#a142f4,color:#111
    classDef out fill:#e6f4ea,stroke:#34a853,color:#111
    classDef loss fill:#fce8e6,stroke:#ea4335,color:#111
```

Every parameter receives its own gradient. `Value.backward()` does not need to know
about neurons or layers: it only reverse-walks the connected scalar graph and executes
the same local rules. In particular, its `+=` operations produce the two sums:

    h1_1.grad = 1*v11 + 1*v21 = 3
    h1_2.grad = 1*v12 + 1*v22 = 2

This is how the single-neuron graph extends to a multilayer neural network: more
connected `Value` objects create more paths, while the four autodiff steps stay exactly
the same.
