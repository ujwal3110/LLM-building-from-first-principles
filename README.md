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
