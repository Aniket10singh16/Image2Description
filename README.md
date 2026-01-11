# 🧠 Mimic Human-Level Understanding of Images  
### Graph-based Image Captioning with Dual Attention

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green)
![NLP](https://img.shields.io/badge/Domain-Natural%20Language%20Processing-orange)
![GCN](https://img.shields.io/badge/Model-Graph%20Neural%20Network-purple)
![Attention](https://img.shields.io/badge/Model-Dual%20Multi--Head%20Attention-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-informational)

---

## 📌 Project Overview
This project implements a **research‑grade image captioning system** that generates **human‑like descriptions** by learning not only *what* objects appear in an image, but also *how they relate to each other*.

Unlike conventional CNN‑RNN captioning pipelines, this system explicitly models **object relationships** using **Graph Convolutional Networks (GCNs)** and uses **Dual Multi‑Head Attention** to align visual relations with natural language.

This work is based on the MSc project **“Mimic Human‑Level Understanding of Image”** (St. Xavier’s College, Kolkata, 2024).

---

## 🧠 Core Idea

Traditional captioning models treat an image as a flat vector:

```
Image → CNN → RNN → Caption
```

This project treats an image as a **structured graph** of interacting objects:

```
Image → Objects → Relation Graph → GCN → Dual Attention → LSTM → Caption
```

This enables reasoning like:

> “A man in a red shirt riding a motorcycle”  
instead of  
> “man motorcycle”

---

# 🏗️ System Architecture

```
Input Image
     │
     ▼
VGG19 CNN (global + object features)
     │
     ▼
Faster‑RCNN Object Detection
     │
     ▼
Object‑Relationship Graph (IoU‑based)
     │
     ▼
Graph Convolution Network (2 layers)
     │
     ▼
Edge Readout (2048‑D)
     │
     ├─ Global CNN features (2048‑D)
     ▼
Concatenation → 4096‑D Encoder Embedding
     │
     ▼
Dual Attention (Self + Cross)
     │
     ▼
LSTM Decoder
     │
     ▼
Caption
```

---

# 🔍 Image Encoding

## 1️⃣ Global and Object Features
Two pretrained networks are used:

| Model | Purpose |
|------|--------|
| **VGG19** | Extracts spatial feature maps |
| **Faster‑RCNN (ResNet‑50‑FPN)** | Detects objects and bounding boxes |

VGG19 outputs:
```
(batch, 49, 2048)
```

These 49 spatial regions represent different parts of the image.

---

## 2️⃣ Object‑Relationship Graph

Each detected object becomes a **graph node**.  
Edges are created when two objects overlap beyond an **IoU threshold**.

If no objects are detected, the entire image becomes a **single node with a self‑loop**.

Each node stores **VGG19 features extracted from the cropped object region**.

---

## 3️⃣ Graph Convolution Network (GCN)

Two graph convolution layers propagate relational information between objects:

```
H1 = ReLU(GCN1(G, H))
H2 = ReLU(GCN2(G, H1))
```

For each edge (u → v):

```
EdgeFeature = concat(H2[u], H2[v])
```

All edge features are passed through a linear layer and **mean‑pooled** to produce a **2048‑D graph embedding**.

This is concatenated with **2048‑D CNN features → 4096‑D encoder output**.

---

# 🧠 Caption Decoder

The decoder uses **Dual Multi‑Head Attention**:

### 🔹 Multi‑Head Self‑Attention
Models word‑to‑word dependencies inside the generated caption.

### 🔹 Multi‑Head Cross‑Attention
Aligns each generated word with **image‑graph features**.

### 🔹 LSTM Generator
Combines both attention contexts to predict the next word.

```
(Self‑Attention + Cross‑Attention) → LSTM → Softmax → Next Word
```

Attention maps can be visualized to show **which image regions influence each word**.

---

# 🧪 Training Pipeline

## Dataset
**Flickr8k**
- 8,000 images
- 5 captions per image

(Designed to scale to Flickr30k / MS‑COCO.)

---

## Preprocessing

**Images**
```
Resize → 224×224 → Normalize
```

**Captions**
- Tokenized using spaCy
- Lowercased
- Vocabulary built with frequency threshold
- <SOS> and <EOS> tokens added

---

## Graph Construction
For each image:
1. Faster‑RCNN detects objects
2. Crop object regions
3. Extract VGG19 features
4. Build IoU‑based graph

---

## Training

At timestep t:

```
Previous words → Self Attention
Image relations → Cross Attention
→ LSTM → Predict next word
```

Loss:  
```
Cross‑Entropy
```

Optimizer:  
```
Adam
```

---

# 📊 Results

| Metric | Score |
|-------|-------|
| BLEU‑1 | **≈ 0.55** |
| BLEU‑2 | **≈ 0.33** |

The GCN + Dual Attention model significantly outperforms the Bi‑LSTM baseline and produces captions that better capture **object interactions, colors, and actions**.

---

# 🛠️ Tech Stack
- PyTorch
- VGG19
- Faster‑RCNN
- DGL (Graph Learning)
- Multi‑Head Attention
- LSTM
- spaCy

---

# 🔮 Future Work
- Larger datasets (MS‑COCO, Flickr30k)
- Scene‑graph supervision
- Transformer‑based decoders
- Custom‑trained object detectors

---

# 📄 Full Project Documentation

The complete technical report (methodology, algorithms, experiments, and results) is available here:

👉 **Project Report (PDF):**  
[Download / View Full Documentation](https://drive.google.com/file/d/1U95B8e3opCeefdzLl1cd1G8cSxwVSRpG/view?usp=drive_link)

This document contains:
- Mathematical formulation of the GCN + Dual Attention model  
- Graph construction algorithm  
- Training and inference procedures  
- BLEU score evaluation  
- Attention visualizations and qualitative results  
- Limitations and future scope  
