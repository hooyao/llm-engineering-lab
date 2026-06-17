# Backpropagation 从零开始 —— 离线学习笔记（中文版）

> **关于这份中文版：** 仓库里的 note 默认是英文的（CLAUDE.md 指令 3/5）。这一份是
> 经用户明确要求的**例外**——中文版，方便配 YouTube 视频离线看。英文正本是同目录的
> `backprop-primer.md`，两份内容一致。注意：所有 ML/systems **术语保持英文不翻译**
> （bias、weight、gradient、forward pass、activation……），这是用户的硬要求；中文只用
> 在串联的句子上。

**写给谁看：** 你写过 linear regression，有 systems/backend 背景，neural network 的
内核全忘了。这份 note 自洽完整。配你自己挑的 YouTube 视频一起看（Karpathy 那个
"The spelled-out intro to backpropagation" 是经典，反正也是后面 Track B1 的内容）。
它的作用就是给视频配一份正好对着你水平写的文字版。

**它在哪一环：** 这是 `experiments/a01-mem-budget/` 底下的"为什么"。A1 的 calculator
需要你知道：training 时每个 parameter 除了自己，还要存一个 *gradient*（2 bytes）加上
一堆 *activation*，而 activation 的大小取决于网络结构。这两样都直接来自 backprop。
同目录的 `teaching-notes.md` 讲了 parameter 是什么、forward pass 怎么跑——先看那份，
这份从那里接着往下。

---

## 0. 一句话总结

train 一个 model 就是反复跑这个循环：

```
1. forward pass：  把 input 喂进网络，得到 output，算出 loss
2. backward pass： 算出每个 weight 对 loss 贡献了多少
3. update：        把每个 weight 朝"让 loss 变小"的方向挪一小步
```

**Backpropagation 就是第 2 步**——高效算出"每个 weight 对 loss 的贡献"，一次把所有
weight 都算出来。这个"贡献"就是 *gradient* `∂L/∂w`。下面全部都是在拆解这一件事。

---

## 1. 我们到底要算什么

training 的 update 规则是 **gradient descent**：

```
w  ←  w  -  lr · ∂L/∂w
```

每个 weight `w`，朝着让 loss `L` 变小的方向挪一小步（步长是 `lr`，learning rate）。
`∂L/∂w` 的符号告诉你往哪个方向挪，大小告诉你 loss 对它有多敏感。

所以 training 唯一缺、而我们手上还没有的东西，就是对每个 weight 算出：

```
∂L/∂w  =  "把这个 weight 往上推一丁点，loss 会变多少、朝哪个方向变？"
```

`∂L/∂w` 读作 "L 对 w 的 partial derivative"。calculus 忘了也没事：derivative 就是个
**斜率**——input 推一丁点，output 变了多少，相除。这里用到的不比这更玄。

backprop 就是把所有这些 `∂L/∂w` 算出来的流程。

---

## 2. 障碍：loss 离 weight 隔了好几层

看一个小小的 2-layer 网络（和 `teaching-notes.md` 里同一个 toy model）：

```
x  ──[W1, b1]──►  z1  ──ReLU──►  h  ──[W2, b2]──►  y  ──loss──►  L
```

weight `W1` 不直接碰 `L`。它影响 `z1`，`z1` 影响 `h`，`h` 影响 `y`，`y` 影响 `L`。
隔了四跳。你没法一步写出 `∂L/∂W1`——中间夹着一串函数。

"穿过一串函数求导"的工具是 **chain rule**：

```
∂L/∂W1  =  ∂L/∂y · ∂y/∂h · ∂h/∂z1 · ∂z1/∂W1
            └─────────────┬──────────────┘
              一串每一跳的斜率，连乘起来
```

从右往左读：`W1` 动一丁点 → `z1` 变（斜率 `∂z1/∂W1`）→ `h` 变（斜率 `∂h/∂z1`）→
`y` 变 → `L` 变。把每一跳的斜率乘起来，就得到从头到尾的总斜率 `∂L/∂W1`。

**关键的效率点**（这就是为什么叫 back-propagation）：那个乘积的左半段
`∂L/∂y · ∂y/∂h · ...`，是网络里这个点**之后**所有 weight **共用**的。所以与其给每个
weight 都把整条链重算一遍，不如**从 loss 这端开始算一次，把这个"running product"
一层一层往回带**。每一层直接复用后一层递给它的结果。这个被往回带的量，就是
"propagation"。

---

## 3. 为什么 forward pass 必须把中间值存下来

再看 chain rule 里那些因子：`∂y/∂h`、`∂h/∂z1`、`∂z1/∂W1`。真去算这些斜率时，会发现
它们用到的正是 **forward pass 里算出来的 activation**——`h`、`z1`、`x`（§4 你会看到
具体在哪一步）。这就是为什么 training 的 forward pass 要留住中间结果，而 inference 的
forward pass 不用：backward pass 需要拿它们当原料。这也是 A1 budget 里
"activation memory" 那一项的来源。

---

## 4. 带真实数字完整走一遍

### Forward

```
x  = [1.0, 2.0]                         # layer 1 的 input activation
W1 = [[0.5, -0.3],
      [0.2,  0.8]]
b1 = [0.1, -0.1]
z1 = W1 @ x + b1 = [0.0, 1.7]           # pre-activation
h  = ReLU(z1)    = [0.0, 1.7]           # layer 2 的 input activation
W2 = [0.7, -0.4]
b2 = [0.05]
y  = W2 @ h + b2 = -0.63                 # output

target t = 0
L  = 0.5·(y - t)² = 0.5·(-0.63)² = 0.198 # squared-error loss
```

（`@` 是 matrix/vector 乘。`ReLU(z) = max(0, z)`，逐元素作用。）

### Backward —— 从 loss 出发，往回走到 input

**Step 0 —— loss 对 output `y` 的 gradient。** 整条 backward 链的种子。对
`L = 0.5·(y - t)²`，导数就是 `(y - t)`：

```
g_y = ∂L/∂y = (y - t) = -0.63
```

**Step A —— 对 `W2` 的 gradient**（我们要的东西之一）。因为 `y = W2 @ h + b2`，
`y` 对 `W2` 的斜率是 `h`，于是由 chain rule：

```
∂L/∂W2 = g_y · hᵀ = -0.63 × [0.0, 1.7] = [0.0, -1.071]   ← 用到 h
∂L/∂b2 = g_y                          = -0.63
```

**Step B —— 把 gradient 往回推到 `h`**（好继续往 layer 1 走）。`y` 对 `h` 的斜率是
`W2`：

```
g_h = W2ᵀ · g_y = [0.7, -0.4] × (-0.63) = [-0.441, 0.252]
```

**Step C —— 穿过 ReLU，推到 `z1`。** ReLU 的斜率：input 为正的地方是 1，为负的地方
是 0（它会掐断 dead unit 上的 gradient）：

```
z1 = [0.0, 1.7]  →  mask = [0, 1]
g_z1 = g_h ⊙ mask = [-0.441×0, 0.252×1] = [0.0, 0.252]
```

（`⊙` 是逐元素乘。）

**Step D —— 对 `W1` 的 gradient**（我们要的另一个）。因为 `z1 = W1 @ x + b1`，对
`W1` 的斜率是 `x`：

```
∂L/∂W1 = g_z1 · xᵀ = [0.0, 0.252]ᵀ ⊗ [1.0, 2.0]
       = [[0.0,   0.0  ],
          [0.252, 0.504]]                            ← 用到 x
∂L/∂b1 = g_z1 = [0.0, 0.252]
```

**Step E —— update 每个 weight**（gradient descent，取 `lr = 0.1`）：

```
W2 ← W2 - 0.1·∂L/∂W2 = [0.7, -0.4] - 0.1·[0.0, -1.071] = [0.7, -0.293]
W1 ← W1 - 0.1·∂L/∂W1 = [[0.5, -0.3],[0.2,0.8]] - 0.1·[[0,0],[0.252,0.504]]
b2 ← b2 - 0.1·∂L/∂b2 ; b1 ← b1 - 0.1·∂L/∂b1
```

这就是**一个 training step**。这条样本上的 loss 现在比 0.198 略小了。在大量样本上、
重复很多次——这个循环本身就是 training。

---

## 5. 最容易把人绕晕的区分：两种 gradient

backward pass 里流动着**两种完全不同**的东西，混淆它们是"我搞不懂 backprop"的头号原因：

| 记号 | 是什么 | 干嘛用 | step 之后留不留 |
|---|---|---|---|
| `g_y`、`g_h`、`g_z1` | loss 对某个 **activation** 的 gradient | *中间信使*——唯一作用是传给前一层 | 用完即扔 |
| `∂L/∂W2`、`∂L/∂W1`、`∂L/∂b` | loss 对某个 **weight/bias** 的 gradient | *最终目标*——拿去 update parameter | 这就是 memory budget 里 "gradient = 2 bytes/param" |

每一层只干两件事：
- **(a)** 用流进来的 activation-gradient + 自己存的那份 input activation，算出**自己
  weight 的 gradient**（Step A、D）；
- **(b)** 把一个 activation-gradient 再往前传给前一层（Step B、C）。

把 `g` 想成往回流的信使，`∂L/∂W` 想成每层沿途卸下的货物。

---

## 6. 为什么每一层都要存**自己那份** activation

注意：Step A 用 `h` 算 `∂L/∂W2`，Step D 用 `x` 算 `∂L/∂W1`。不同层的 weight-gradient
锁定**不同的** activation，而且**不能互换**——`h` 帮不了 `W1`，`x` 帮不了 `W2`。
第 `k` 层的一般式：

```
∂L/∂W_k  =  (传到第 k 层的 activation-gradient)  ·  (第 k 层的 input activation)ᵀ
```

所以一个 28-layer 网络要留住 28 份 activation（`x = a₀, a₁, ..., a₂₇`）。backward pass
从 layer 28 倒着走到 layer 1，走到第 `k` 层时正好消耗掉 activation `a_{k-1}`。这正是
为什么 training 的 forward pass 不能像 inference 那样每层算完就扔：**每一份都欠着一个
未来的 backward step。**

这就是"为什么要存每一层的输出，而不是只存最后一个"的答案——不是最后那个 `h` 被需要，
而是**每层各要一份自己的 input activation**，因为 backward pass 会逐层回来，每层向你
要它自己那份。

---

## 7. 怎么接回 A1 的 memory budget

现在 `budget.py` 里那两项"只有 training 才有"的 memory，都有理由了：

- **gradient，2 bytes/param** —— §5 里那个 `∂L/∂W` 货物。每个 weight 一个 gradient
  值，dtype 跟 weight 一样（BF16）。你得留它留到 Step E 做完 update。
- **activations** —— §6 里那些每层存下来的 input。总大小随 `seq_len × batch × hidden ×
  layers` 变大，因为 token 越多 / batch 越大 = 越多中间值要留到 backward pass 来取。
- **activation checkpointing**（Track A5）—— 整份 note 让这个优化变得好懂：forward 时
  *不存*大部分 activation；等 backward pass 要用 `a_{k-1}` 时，**临时重跑那一段 forward
  把它算回来**（recompute）。拿 compute 换 memory。A1 activation 公式里那个 "×6 vs ×1"
  系数，就是"为 backward 保活一个 transformer block 的全部中间值" 和 "需要时再算回来"
  之间的差别。

整个循环就是：forward 存下 activation → backward 从 loss 出发，用 chain rule 逐层倒推，
产出 weight gradient → gradient descent 把它们应用上去。activation 之所以占 memory，
就是因为它们必须从 forward 一路存活到那个来消耗它的 backward step。

---

## 8. 看完视频后读这段 30 行 numpy

视频 + 这份 note 都通了之后，下面就是整个算法、不带任何 framework。读它是确认你真懂了的
最快方式（Track C6 和 Track B1 你会亲手写类似的东西）：

```python
import numpy as np

# one 2-layer MLP, one training step, no autograd
x  = np.array([1.0, 2.0]); t = np.array([0.0])
W1 = np.array([[0.5, -0.3], [0.2, 0.8]]); b1 = np.array([0.1, -0.1])
W2 = np.array([[0.7, -0.4]]);             b2 = np.array([0.05])

# ---- forward (save z1, h, x for the backward pass) ----
z1 = W1 @ x + b1
h  = np.maximum(0, z1)            # ReLU
y  = W2 @ h + b2
L  = 0.5 * np.sum((y - t) ** 2)

# ---- backward ----
g_y  = (y - t)                   # Step 0
gW2  = np.outer(g_y, h)          # Step A   (uses h)
gb2  = g_y
g_h  = W2.T @ g_y                # Step B
g_z1 = g_h * (z1 > 0)            # Step C   (ReLU mask)
gW1  = np.outer(g_z1, x)         # Step D   (uses x)
gb1  = g_z1

# ---- update ----
lr = 0.1
W2 -= lr * gW2; b2 -= lr * gb2
W1 -= lr * gW1; b1 -= lr * gb1
```

把每一行对回上面的 step。当 `gW1 = np.outer(g_z1, x)` 对你变得显然——"`W1` 的 gradient
当然要用 `x`"——你就把 backprop 找回来了。

---

## 推荐视频（挑一个，这份 note 是配套文字）

- **Karpathy, "The spelled-out intro to neural networks and backpropagation"**
  （YouTube, ~2h25m）—— 从头搭 micrograd，一个标量一个标量地推。这就是 Track B1；
  现在做了，B1 等于复习。和这份 note 最对得上。
- **3Blue1Brown, "Backpropagation calculus"**（Deep Learning 第 4 章, ~10m）—— 偏
  视觉/直觉的版本。更短；适合在 Karpathy 之前先过一遍。

两个都行。这里的数字和步骤是照着这两位的讲法对齐写的。
