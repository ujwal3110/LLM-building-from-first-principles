# 🧠 Building an LLM From First Principles

## Day 1 – Foundations, Scope, and Mathematical Intuition

---

## 📌 Project Overview

This project is a **ground-up implementation of a Large Language Model (LLM)** built **from first principles**, without relying on high-level deep learning frameworks initially.

The goal is **deep understanding**, not just training a model.

We focus on:

* How language is represented mathematically
* How neural networks process sequences
* How transformers actually work internally
* How inference and training map to real hardware

This project is educational, research-grade, and systems-oriented.

---

## 🎯 Long-Term Goals

By the end of this project, we will:

* Implement a **Transformer-based LLM** from scratch
* Build our **own tensor engine (minimal NumPy-like)**
* Implement **forward + backward pass manually**
* Train a small language model end-to-end
* Understand **memory, compute, and optimization tradeoffs**
* Prepare the model for **edge / systems-level deployment**

> This aligns with a **Systems ML** mindset rather than just ML usage.

---

## 🗺️ Project Roadmap (High Level)

| Phase   | What We Build                  |
| ------- | ------------------------------ |
| Phase 1 | Math, vectors, tokens, tensors |
| Phase 2 | Neural network basics (MLP)    |
| Phase 3 | Backpropagation from scratch   |
| Phase 4 | Attention mechanism            |
| Phase 5 | Transformer block              |
| Phase 6 | Training loop                  |
| Phase 7 | Inference & optimization       |

Day 1 starts **Phase 1**.

---

## 📅 Day 1 Objectives

Today is about **mental models and foundations**.

### ✅ What We Learn Today

* What language modeling really means
* How text becomes numbers
* What vectors and tensors represent
* Why matrix multiplication is central
* How an LLM differs from classical NLP

No training yet. No transformers yet.

Just **clarity**.

---

## 🧩 What Is a Language Model?

A **Language Model** estimates the probability:

> [ P(next_token | previous_tokens) ]

Example:

```
Input:  "The sky is"
Output: "blue"
```

The model does NOT understand meaning.
It learns **statistical patterns** in token sequences.

---

## 🔢 Step 1: Text → Tokens

LLMs do not read characters or words.
They read **tokens**.

Example:

```
"hello world" → [15496, 995]
```

Tokens are:

* Integers
* Indices into a vocabulary

> Vocabulary = fixed set of known tokens

---

## 📐 Step 2: Tokens → Vectors (Embeddings)

Each token ID maps to a vector:

```
Token ID: 15496
Embedding: [0.12, -0.87, 1.03, ...]
```

If:

* Vocabulary size = 50,000
* Embedding dimension = 512

Then embedding matrix shape:

```
[50000 × 512]
```

This matrix is **learned**.

---

## 🧮 Why Vectors?

Vectors allow:

* Similarity (dot product)
* Direction (semantics)
* Linear algebra operations

Example intuition:

```
king - man + woman ≈ queen
```

This emerges naturally during training.

---

## 📦 From Vectors to Tensors

| Concept         | Shape Example           |
| --------------- | ----------------------- |
| Token embedding | (512,)                  |
| Sentence        | (sequence_length × 512) |
| Batch           | (batch × seq × 512)     |

This 3D structure is a **tensor**.

LLMs operate almost entirely on tensors.

---

## 🔁 The Core Operation: Matrix Multiplication

Almost everything reduces to:

```
Y = X × W + b
```

Where:

* `X` = input tensor
* `W` = learned weights
* `b` = bias

Attention, MLPs, projections — all are matrix multiplies.

---

## 🧠 Mental Model to Keep

> An LLM is a **stack of matrix multiplications**
> with **non-linearities** and **clever routing (attention)**.

There is no magic.
Only math + scale.

---

## 📁 Repository Structure (Initial)

```
llm-from-scratch/
├── README.md
├── notes/
│   └── day01-foundations.md
├── math/
│   └── vectors.py   (coming soon)
└── experiments/
```

We will grow this incrementally.

---

## 🧪 What We Are NOT Doing Today

❌ No PyTorch
❌ No Transformers
❌ No Training
❌ No GPUs

We build understanding before abstraction.

---

## 📚 Recommended Reading (Optional)

* "Attention Is All You Need" (skim only)
* Linear Algebra (dot product, matrices)
* Probability basics

---

## 🧠 Day 1 Takeaway

If you deeply understand:

* Tokens
* Embeddings
* Vectors
* Matrix multiplication

You already understand **50% of an LLM**.

---

## ⏭️ Next: Day 2 Preview

**Day 2 – Vectorized Computation & Tensor Engine**

* Implement vectors manually
* Broadcasting rules
* Batched computation
* Foundations for backprop

---

🔥 *This project is about mastery, not shortcuts.*

# 🧠 Building an LLM From First Principles

## Day 2 – Tensor Engine & Vectorized Computation

---

## 📌 Day 2 Overview

Day 2 is where we **stop thinking in scalars** and start thinking like an LLM.

Modern language models are not built on loops over numbers.
They are built on **vectorized tensor operations**.

Today, we lay the foundation for our **own minimal tensor engine** — the computational heart of everything that follows.

---

## 🎯 Day 2 Objectives

By the end of Day 2, you will understand:

* What tensors really are (beyond NumPy definitions)
* Why vectorization is mandatory, not optional
* How broadcasting works internally
* How batching enables scale
* How a tensor engine maps to real hardware (CPU / SIMD / GPU)

And you will **start implementing** these ideas from scratch.

---

## 🧩 Why We Need a Tensor Engine

An LLM performs operations like:

* Matrix multiplication
* Element-wise addition / multiplication
* Reductions (sum, mean)
* Reshaping and broadcasting

If we rely on black-box libraries too early, we lose:

* Performance intuition
* Memory awareness
* Debugging clarity

So we build a **minimal tensor core** ourselves.

Not fast.
Not fancy.
But correct and understandable.

---

## 📐 What Is a Tensor (Really)?

A **tensor** is:

> A contiguous block of memory + a shape + a way to index it

Example:

| Concept | Value                |
| ------- | -------------------- |
| Data    | `[1, 2, 3, 4, 5, 6]` |
| Shape   | `(2, 3)`             |
| Meaning | 2 rows × 3 columns   |

There is **no inherent dimensional magic**.
Dimensions are an interpretation.

---

## 🔢 Scalars → Vectors → Matrices → Tensors

| Level  | Shape       |
| ------ | ----------- |
| Scalar | `()`        |
| Vector | `(d,)`      |
| Matrix | `(n, d)`    |
| Tensor | `(b, n, d)` |

LLMs almost always operate on **3D tensors**:

```
(batch, sequence_length, embedding_dim)
```

---

## 🚀 Why Vectorization Matters

### ❌ Scalar Thinking (Slow)

```
for i in range(n):
    y[i] = x[i] * w[i]
```

### ✅ Vectorized Thinking (Fast)

```
y = x * w
```

Vectorization:

* Reduces Python overhead
* Enables SIMD instructions
* Maps directly to GPUs

> If it’s not vectorized, it doesn’t scale.

---

## 🧠 Mental Model: Data-Parallel Execution

Vectorized code means:

> Apply the **same operation** to **many data points at once**

Hardware executes this via:

* CPU SIMD (AVX / NEON)
* GPU warps
* TPU systolic arrays

Your code shape determines hardware efficiency.

---

## 📦 Broadcasting Explained (From First Principles)

Broadcasting allows operations on tensors of different shapes.

Example:

```
A: (batch, seq, dim)
b: (dim,)

A + b → (batch, seq, dim)
```

Rules (simplified):

1. Align dimensions from the right
2. Dimensions must be equal OR one of them is 1
3. Size-1 dimensions are virtually expanded

No data is copied.
Only indexing changes.

---

## 🧪 Broadcasting Intuition

```
[1, 2, 3]      → (3,)
[[1,2,3],      → (2,3)
 [4,5,6]]

Add → each row gets the vector added
```

This is foundational for:

* Bias addition
* Layer normalization
* Attention scores

---

## 🧮 Core Tensor Operations We Need

Minimum viable tensor engine supports:

* Element-wise add / mul
* Matrix multiplication
* Reshape
* Transpose
* Reduction (sum, mean)

Everything else builds on these.

---

## 🛠️ Day 2 Implementation Plan

Today we implement **Tensor v0**.

### Features

* Backed by Python lists or NumPy arrays
* Shape tracking
* Basic operations

### Non-Goals (For Now)

❌ Autograd
❌ GPU support
❌ Performance optimization

Correctness first.

---

## 📁 Repository Structure (Updated)

```
llm-from-scratch/
├── README.md
├── docs/
│   ├── day01-foundations.md
│   └── day02-tensors.md
├── tensor/
│   ├── tensor.py
│   └── ops.py
├── experiments/
└── tests/
```

---

## 🧠 Systems Insight

Every tensor operation eventually becomes:

* Pointer arithmetic
* Strided memory access
* Fused loops

Understanding this lets you:

* Optimize kernels
* Reduce memory movement
* Reason about cache efficiency

This is **Systems ML** thinking.

---

## 🧪 Validation Strategy

We validate tensors by:

* Shape assertions
* Small hand-computed examples
* Comparing with NumPy outputs

Never trust silent correctness.

---

## 🧠 Day 2 Takeaway

If Day 1 taught *what* LLMs compute,

Day 2 teaches *how* they compute it efficiently.

> Tensors are not abstractions.
> They are disciplined memory layouts.

---

## ⏭️ Next: Day 3 Preview

**Day 3 – Automatic Differentiation (Backprop from Scratch)**

* Computational graphs
* Gradient flow
* Chain rule in code
* Manual backward passes

---

🔥 *If you can build a tensor engine, you can build an LLM.*

