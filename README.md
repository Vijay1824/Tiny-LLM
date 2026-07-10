#  TinyMind

> **Building a Tiny Language Model from scratch to understand how modern LLMs work under the hood.**

TinyMind is an educational project focused on learning the concepts, mathematics, and engineering behind Large Language Models (LLMs). Instead of relying on high-level libraries, the goal is to implement the core components step by step and understand how everything fits together.

This repository documents the journey from a simple language model to a more capable transformer-based model.

---

#  Project Roadmap

## V1 — Educational GPT *(Training Phase)*

The first version focuses entirely on understanding how a GPT-style transformer is built and trained.

At this stage, the model can **train successfully**, but it does **not yet support inference or an interactive chatbot interface**. The objective is to implement every major component manually before moving on to optimization and deployment.

### Features Implemented

#### Tokenization
- Character-level tokenizer
- Vocabulary creation
- Text encoding & decoding

#### Dataset Pipeline
- Sliding context window
- Next-token prediction dataset

#### Embeddings
- Learnable token embeddings
- Learnable positional embeddings

#### Attention
- Scaled Dot-Product Self-Attention
- Query, Key & Value projections
- Attention score computation
- Softmax attention
- Causal masking
- Multi-Head Attention
- Output projection

#### Feed Forward Network (MLP)

```text
Linear
   ↓
GELU
   ↓
Linear
```

#### Transformer Block

```text
LayerNorm
      ↓
Multi-Head Attention
      ↓
Residual Connection
      ↓
LayerNorm
      ↓
Feed Forward Network
      ↓
Residual Connection
```

#### GPT Architecture
- Token embeddings
- Positional embeddings
- Multiple Transformer blocks
- Final LayerNorm
- Language Modeling Head

#### Training Pipeline
- Cross Entropy Loss
- AdamW Optimizer
- Gradient Clipping
- Cosine Learning Rate Scheduler
- CUDA Support
- CPU Support
- Windows Compatibility

---

# Current Model Configuration

| Parameter | Value |
|-----------|------:|
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Context Length | 64 |
| Parameters | ~817K |

---

# Current Limitations

This project is still in its educational stage and intentionally keeps the implementation simple.

Current limitations include:

- Character-level tokenizer only
- No inference engine
- No text generation
- No chatbot interface
- Basic checkpoint saving only
- No checkpoint loading/resume training
- No validation loop
- No mixed precision (AMP)
- No weight tying
- No Rotary Positional Embeddings (RoPE)
- No KV Cache
- No Flash Attention
- No distributed training
- No quantization
- No fine-tuning support

---

# V2 — Towards a Production-Style Tiny LLM

The second version will focus on improving both performance and usability while keeping the code educational and easy to understand.

### Planned Features

- Larger Transformer architecture
- Better tokenizer (BPE / SentencePiece)
- Faster inference engine
- Interactive chatbot interface
- Text generation pipeline
- Checkpoint loading & management
- Validation pipeline
- Mixed Precision (AMP)
- Weight tying
- Rotary Positional Embeddings (RoPE)
- KV Cache
- Flash Attention
- Quantization
- Fine-tuning support
- Retrieval-Augmented Generation (RAG)
- Agent capabilities
- Memory system
- Better training utilities
- Performance optimizations

---

# Learning Goals

This project is designed to explore and understand:

- How tokenizers work
- How embeddings represent language
- How self-attention learns context
- How transformer blocks are built
- How GPT models predict the next token
- How language models are trained
- How inference works
- How modern LLM optimizations improve performance

---

# Project Philosophy

TinyMind is built with one primary goal:

> **Learn by building.**

Rather than relying on existing LLM frameworks, every major component is implemented from scratch wherever practical. The focus is on understanding *why* each component exists and *how* modern language models work internally.

As the project evolves, each version will progressively introduce more advanced techniques, eventually transforming TinyMind from a simple educational model into a modern, production-inspired language model.

---

## Tech Stack

- Python
- PyTorch
- CUDA (GPU Training)
- NumPy

---

## Project Status

| Version | Status |
|---------|--------|
| V1 | 🚧 In Progress |
| V2 | 📋 Planned |

---

>  If you find this project interesting or helpful, consider giving it a star!
>  also please contact to know about contributing to it. 
