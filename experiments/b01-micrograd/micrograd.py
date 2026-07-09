"""
micrograd.py — a scalar reverse-mode autodiff engine, by hand.

This is the B1 deliverable. Every piece maps to one of the 4 steps from
learning-notes.md. The `Value` class is ~40 lines of real logic; the rest is the
worked example + verification so you can SEE your hand-derived gradients come out.

    Step 1  forward pass builds a graph : __add__ / __mul__ return a new Value that
                                          remembers its parents (_prev) and its op.
    Step 2  each node's LOCAL derivative : encoded inside each op's _backward closure
                                          (multiply reads the sibling's forward value;
                                          add uses 1).
    Step 3  chain rule                   : _backward multiplies the local derivative by
                                          out.grad (the child's global gradient).
    Step 4  reverse topological order    : backward() topo-sorts the graph, seeds
            + accumulate at fan-out        dL/dL = 1, walks it in REVERSE, and every
                                          parent does grad += ... (accumulation).
"""


class Value:
    """A single scalar that remembers how it was computed, so it can backprop."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data          # the forward value (Step 1)
        self.grad = 0.0           # dL/d(self); filled by backward() (starts at 0)
        # --- graph bookkeeping (Step 1: "record each op as a node") ---
        self._prev = set(_children)   # the parent Values this node was computed from
        self._op = _op                # label, e.g. '*' or '+', for readability
        self._backward = lambda: None # how to push grad to parents; set by each op

    # ---- Step 1 + Step 2: an op builds a node AND defines its local derivative ----

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")   # Step 1: new node

        def _backward():
            # multiply rule:  d(out)/d(self) = other,  d(out)/d(other) = self
            # (the OTHER input's forward value — exactly de/da = b in the notes)
            # times out.grad = child's global gradient (Step 3 chain rule)
            # += not = : accumulate if this node fans out (Step 4)
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")   # Step 1: new node

        def _backward():
            # add rule:  d(out)/d(self) = 1,  d(out)/d(other) = 1  (no forward value read)
            # gradient passes straight through, times out.grad (Step 3), accumulated (Step 4)
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # activation: a THIRD op, so a third local-derivative rule.
        # single input -> single output; squashes any real number into (-1, 1).
        import math
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)   # = tanh(x)
        out = Value(t, (self,), "tanh")                     # Step 1: new node, one parent

        def _backward():
            # tanh rule:  d(out)/d(self) = 1 - out^2   (expressed via the OUTPUT value,
            # not the input — same shape as sigmoid's o*(1-o) from A2).
            # times out.grad (Step 3), accumulated (Step 4).
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    # ---- Step 3 + Step 4: run the whole backward pass ----

    def backward(self):
        # Step 4: build a reverse-topological order. A node is appended only AFTER all
        # its parents are visited, so `reversed(topo)` processes every node only once
        # ALL of its consumers are already done (their grad contributions are in).
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)     # parents first, self last

        build_topo(self)

        self.grad = 1.0            # Step 3 seed: dL/dL = 1
        for v in reversed(topo):   # Step 4: output-first walk
            v._backward()          # each node pushes grad into its parents

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # convenience so the demo reads naturally (Value + number, number * Value, etc.)
    __radd__ = __add__
    __rmul__ = __mul__


# ---------------------------------------------------------------------------
# Demo 1 — the worked example from learning-notes.md
#   a=2, b=-3, e=a*b, c=10, d=e+c, f=2, L=d*f
# Hand-derived gradients (Step 3 in the notes):
#   dL/da=-6, dL/db=4, dL/dc=2, dL/df=4  (and dL/de=2, dL/dd=2 internally)
# ---------------------------------------------------------------------------
def demo1():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(2.0)

    e = a * b        # -6.0
    d = e + c        #  4.0
    L = d * f        #  8.0

    L.backward()

    print("Demo 1 — worked example")
    print(f"  forward:  e={e.data}  d={d.data}  L={L.data}")
    print(f"  a.grad = {a.grad:+.1f}   (hand-derived -6.0)")
    print(f"  b.grad = {b.grad:+.1f}   (hand-derived +4.0)")
    print(f"  c.grad = {c.grad:+.1f}   (hand-derived +2.0)")
    print(f"  f.grad = {f.grad:+.1f}   (hand-derived +4.0)")
    print(f"  e.grad = {e.grad:+.1f}   (hand-derived +2.0)")
    print(f"  d.grad = {d.grad:+.1f}   (hand-derived +2.0)")

    assert a.grad == -6.0 and b.grad == 4.0 and c.grad == 2.0 and f.grad == 4.0
    print("  OK: engine matches hand-derived gradients.\n")


# ---------------------------------------------------------------------------
# Demo 2 — fan-out / accumulation (Step 4)
#   x=3, p=x*2, q=x+5, L=p+q  ->  dL/dx = (via p) + (via q) = 2 + 1 = 3
# ---------------------------------------------------------------------------
def demo2():
    x = Value(3.0)
    p = x * 2        # x used here ...
    q = x + 5        # ... and here -> x fans out to two nodes
    L = p + q
    L.backward()

    print("Demo 2 — fan-out accumulation")
    print(f"  x.grad = {x.grad:+.1f}   (2 via p + 1 via q = 3.0)")
    assert x.grad == 3.0
    print("  OK: the two paths were SUMMED (+= accumulation).\n")


# ---------------------------------------------------------------------------
# Demo 3 — verify against torch.autograd (optional; skipped if torch missing)
# ---------------------------------------------------------------------------
def demo3():
    try:
        import torch
    except ImportError:
        print("Demo 3 — torch not installed here, skipping the cross-check.")
        print("  (Run this file where torch is available to confirm the match.)")
        return

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    c = torch.tensor(10.0, requires_grad=True)
    f = torch.tensor(2.0, requires_grad=True)
    L = (a * b + c) * f
    L.backward()

    print("Demo 3 — cross-check vs torch.autograd")
    print(f"  torch: a={a.grad.item():+.1f} b={b.grad.item():+.1f} "
          f"c={c.grad.item():+.1f} f={f.grad.item():+.1f}")
    assert (a.grad.item(), b.grad.item(), c.grad.item(), f.grad.item()) == (-6.0, 4.0, 2.0, 4.0)
    print("  OK: our engine == torch.autograd == hand-derived. All three agree.\n")


# ---------------------------------------------------------------------------
# Demo 4 — one neuron, trained (activation in action)
#   A neuron = linear part (w.x + b) then an activation (tanh).
#       z   = w1*x1 + w2*x2 + b        (the linear-regression part)
#       out = tanh(z)                  (the non-linearity)
#   We push `out` toward a target with gradient descent, using OUR engine's
#   backward() (multiply + add + tanh rules) to get every gradient, and watch loss fall.
# ---------------------------------------------------------------------------
def demo4():
    x1, x2 = Value(1.0), Value(-2.0)       # one training example, two features (fixed)
    w1, w2 = Value(-0.5), Value(0.8)       # parameters (training updates these)
    b = Value(0.1)
    target = 1.0                           # want the neuron to output +1.0 here

    print("Demo 4 — one neuron, trained with our own backward()")
    lr = 0.1
    out = None
    for step in range(20):
        # forward: z = w1*x1 + w2*x2 + b, then out = tanh(z)
        z = w1 * x1 + w2 * x2 + b
        out = z.tanh()
        diff = out + (-target)             # out - target
        loss = diff * diff                 # squared error

        for p in (w1, w2, b):              # Step 4: zero grads or += piles up across steps
            p.grad = 0.0
        loss.backward()

        for p in (w1, w2, b):              # gradient descent: nudge each param down its grad
            p.data += -lr * p.grad

        if step % 4 == 0 or step == 19:
            print(f"  step {step:2d}:  out={out.data:+.4f}  loss={loss.data:.6f}")

    print(f"  final out={out.data:+.4f}  (target {target:+.1f}) — loss fell, neuron learned.\n")


if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
