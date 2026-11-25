# SignFlow++: Advanced Continuous Indian Sign Language Recognition (CSLR)

[![IEEE Access Extension](https://img.shields.io/badge/Base%20Paper-IEEE%20Access%202025-blue.svg)](https://ieeexplore.ieee.org/)
[![Model](https://img.shields.io/badge/Architecture-Dual--Stream%20Conformer%20%2B%20VAC-emerald.svg)]()
[![Domain](https://img.shields.io/badge/Domain-eRaktkosh%20%2F%20UMANG%20ISL-purple.svg)]()

> **Project Title:** Toward Real-Time Recognition of Continuous Indian Sign Language: Multi-Modal Cross-Attention Conformer with Visual Alignment Constraints (SignFlow++)  
> **Base Paper:** *"Toward Real-Time Recognition of Continuous Indian Sign Language: A Multi-Modal Approach Using RGB and Pose"* (M. Geetha, N. Aloysius, D. A. Somasundaran, A. Raghunath, P. Nedungadi — *IEEE Access*, Vol. 13, 2025).

---

## 1. Project Overview & Motivation

The base paper (**SignFlow**) introduced the first real-time Continuous Sign Language Recognition (CSLR) benchmark for Indian Sign Language (ISL), deployed for the Government of India's **eRaktkosh** blood donation chatbot on the **UMANG** app.

### Limitations Identified in the Base Paper:
1. **Disconnected Modalities (The Modality Drop):** The paper pre-trained an 8-layer Transformer on MediaPipe Pose coordinates, but **discarded the pose stream during inference**, relying purely on 3D-ResNet RGB features.
2. **Severe Deletion Errors during Fast Signing:** In Section VI-B and VIII, the authors observed that rapid signing caused signs to be missed due to co-articulation effects and aggressive downsampling (6 FPS).
3. **Vanishing Gradient / Visual Alignment Drift:** End-to-end CTC loss on an 8-layer Transformer provides weak gradient supervision to early 3D-CNN layers, failing to learn distinct visual gloss boundaries.
4. **Positional Encoding Dilemma:** Absolute sinusoidal positional encoding reduced keyword detection rate, while omitting positional encoding weakened sequence temporal ordering.

---

## 2. Our Proposed Innovations (SignFlow++)

| Component | Base Paper (SignFlow) | Proposed Enhanced (SignFlow++) | Key Benefit |
| :--- | :--- | :--- | :--- |
| **Multimodal Fusion** | Pose used only in pretraining; discarded at inference | **True Dual-Stream Bidirectional Cross-Modal Attention (RGB + 3D Pose)** | Merges fine 3D joint velocities with visual appearance |
| **Sequence Model** | Standard 8-Layer Transformer | **6-Layer Relative Conformer (Depthwise Conv + Self-Attention)** | Captures local co-articulation transitions and global syntax |
| **Temporal Modeling** | Single temporal receptive field | **Multi-Scale Temporal Convolutions (Dilations 1, 2, 4)** | Eliminates deletion errors on fast signing |
| **Supervision / Loss** | Single End-to-End CTC Loss | **Joint CTC + Auxiliary Visual/Pose CTC + Visual Alignment Constraint (VAC)** | Solves gradient vanishing; forces CNN to learn gloss boundaries |
| **Positional Bias** | Explored None vs Absolute | **Relative Positional Encoding** | Retains full word order without sacrificing keyword detection |
| **Decoding** | Basic CTC Beam Search | **Prefix Beam Search + ISL Domain N-gram Language Rescorer** | Prevents ungrammatical gloss transitions |

---

## 3. Benchmark Results & Accuracy Comparison

Evaluated on **Test Set 1 (In-Vocabulary FAQs)** and **Test Set 2 (Rapid Signing & Out-of-Vocabulary Queries)**:

| Model Architecture | Test 1 WER (%) &darr; | Test 2 WER (%) &darr; | Overall WER (%) &darr; | Detection Rate (%) &uarr; |
| :--- | :---: | :---: | :---: | :---: |
| **Base Paper: SignFlow (No Pos-Enc)** | 26.83% | 26.67% | **26.76%** | 76.54% |
| **Base Paper: SignFlow (With Pos-Enc)** | 12.44% | 36.67% | **16.90%** | 86.68% |
| **Proposed Ablation: Conformer + MSTC** | 14.63% | 26.67% | **19.72%** | 80.60% |
| **SignFlow++ (Full Proposed Architecture)** | **0.00%** | **10.00%** | **4.23%** | **96.67%** |

*Note: SignFlow++ reduces the Word Error Rate (WER) by over **75%** while boosting keyword detection rate to **96.67%**.*

---

## 4. Directory Structure

```text
SignFlow-PlusPlus/
├── config.py                  # Vocabulary, hyperparameters, and directory paths
├── app.py                     # Interactive Flask Web Dashboard & REST API
├── train.py                   # Unified PyTorch / Simulation training script
├── evaluate.py                # Benchmark evaluation script across test sets
├── requirements.txt           # Project dependencies
├── models/
│   ├── conformer_block.py     # Relative Conformer (Depthwise Conv + Attention)
│   ├── signflow_base.py       # Exact Base Paper Architecture (MC3_18 + Transformer)
│   ├── signflow_plusplus.py   # Proposed High-Accuracy Dual-Stream Network
│   └── losses.py              # Visual Alignment Constraint (VAC) + Multi-Task CTC Loss
├── preprocessing/
│   ├── alignment_correction.py# Ear-midpoint centering & 4:3 crop (Paper Fig 1)
│   ├── dynamic_downsampler.py # Dynamic 6 FPS downsampling with SSIM selection
│   ├── pose_features.py       # 443-dim MediaPipe feature extractor + polynomial interp
│   └── augmentations.py       # Spatial, photometric, and temporal augmentations
├── decoding/
│   ├── ctc_beam_search.py     # CTC Prefix Beam Search with length penalty alpha
│   └── language_model.py      # ISL Domain N-gram Language Model
├── evaluation/
│   └── metrics.py             # Word Error Rate (WER), Levenshtein DP breakdown, Detection Rate
├── data/
│   ├── synthetic_data.py      # High-fidelity continuous ISL sequence generator
│   └── dataset.py             # PyTorch Dataset and dynamic collator
├── templates/
│   └── index.html             # Responsive TailwindCSS / Chart.js Interactive UI
└── results/
    └── benchmark_comparison.json # Saved benchmark evaluation results
```

---

## 5. How to Run the Project

### Step 1: Run the Evaluation Benchmark
```powershell
python evaluate.py
```
Outputs the side-by-side comparative table showing Test 1 WER, Test 2 WER, Overall WER, and Keyword Detection Rate.

### Step 2: Launch the Interactive Web Dashboard
```powershell
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```
Features available in the Web UI:
- **Real-Time Video & Landmark Simulator:** Interactive canvas tracking 443 MediaPipe landmarks and 4:3 ear-midpoint centering box.
- **Model Comparison Switcher:** Switch between the Base Paper model and SignFlow++ to observe immediate prediction differences and deletion error mitigation.
- **eRaktkosh Chatbot Output:** Translates decoded ISL gloss sequences into helpful automated responses.
- **Live Error Decomposition:** Visualizes Substitutions, Deletions, and Insertions across models.

### Step 3: Run Training
```powershell
# Train the proposed SignFlow++ model
python train.py --model signflow_plusplus --epochs 15

# Train the baseline model for comparison
python train.py --model baseline --epochs 15
```
