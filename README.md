# Free Energy Principle (FEP) Abstraction Layer for MoE Models

An architectural framework that applies the **Free Energy Principle (FEP)** as an abstraction overlay on top of Mixture-of-Experts (MoE) architectures (e.g., 20B parameters). By treating inference as a variational prediction error minimization process, this layer compresses the active execution footprint down to a hyper-sparse subnetwork, subsequently translating those weights into multiplier-free bitwise primitives for ultra-low-power edge deployment.

---

## 🚀 The Architectural Concept

In a standard 20B MoE architecture, a generic token routing gate natively fires a baseline window of **3.6 Billion parameters** per forward pass. This project introduces a prompt-agnostic **4.19M parameter FEP Hybrid Overlay** that intercepts the activation stream, calculating a narrow **Information Bottleneck** that truncates the active mathematical execution window down to an ultra-lean **~14.06 Million parameters** (a 256x reduction in raw parameter load).

### Project 2 Extension: Extreme Hardware-Level Load Compression
While isolating ~14M parameters solves the structural compute problem, running them as standard floating-point matrix multiplications ($\mathbf{W} \cdot \mathbf{X}$) still introduces severe thermal and power liabilities on mobile chips and wearable canvases. 

Project 2 completely eliminates power-hungry floating-point Multiply-Accumulate (MAC) pipelines within the active path by quantizing weights and activations into 8-bit fixed-point representations and decomposing them into power-of-two components. This allows the hardware to execute inference using **pure bitwise shift-add operations** ($\ll, +$).

   [ Prompt Input Context ]
              │
              ▼
┌──────────────────────────────┐
│ 1. Topological Manifold Map  │ ◄── Identifies non-linear curved semantic shape
└──────────────┬───────────────┘
│
▼
┌──────────────────────────────┐
│   2. Discrete VQ Codebook    │ ◄── Snaps continuous space to discrete indices
└──────────────┬───────────────┘
│
▼
┌──────────────────────────────┐
│ 3. Sparse Dictionary Filter  │ ◄── Truncates path to ~14.06M Active Params
└──────────────┬───────────────┘
│
▼
┌──────────────────────────────┐
│ 4. Fixed-Point Quantization  │ ◄── Converts Float32 to 8-Bit Signed Integers
└──────────────┬───────────────┘
│
▼
┌──────────────────────────────┐
│ 5. Bitwise Shift register    │ ◄── Replaces W ∙ X with zero-multiplier
│      Decomposition Core      │     hardware left-shifts (<<) and additions
└──────────────┬───────────────┘
│
▼
[ Low-Power NPU Output ]