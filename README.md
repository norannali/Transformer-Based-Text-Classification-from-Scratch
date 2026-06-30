# Transformer-Based-Text-Classification-from-Scratch

An end-to-end NLP PyTorch implementation of a Transformer-based text classifier built entirely from scratch (without using high-level frameworks like HuggingFace, NLTK, or SpaCy). The model is evaluated on the **AG News** dataset for multi-class news classification.

---

## 📌 Project Overview
The core objective of this project is to implement and understand the inner workings of the Transformer architecture (specifically the Encoder block) as proposed in the seminal paper *"Attention Is All You Need"*. Every component, from tokenization and text preprocessing to advanced optimization techniques like gradient clipping, has been engineered from the ground up using core PyTorch and NumPy.

- **Dataset:** AG News (4-class news classification)
- **Task:** Multi-class classification (World / Sports / Business / Sci-Tech)
- **Modality:** Text
- **Target Accuracy:** ~90%

---

## 🏗️ Architecture & Features Built From Scratch

### 1. Tokenizer & Vocabulary
- Built a custom deterministic text preprocessing pipeline using regex.
- Vectorized tokens based on a training-set-only vocabulary (Vocabulary Size: 30,000).
- Handled special tokens manually: `<PAD> = 0` and `<UNK> = 1`.

### 2. Positional Encoding
Implemented the original sinusoidal positional encoding formula to inject sequential order into the embeddings:

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 3. Multi-Head Self-Attention
- Implemented Scaled Dot-Product Attention without using `nn.MultiheadAttention`.
- Configured dynamic padding masks to ignore padding tokens during attention weight calculation.

### 4. Custom Layer Normalization
- Developed a manual `LayerNorm` module (`ManualLayerNorm`) to stabilize internal hidden states.


### 5. Training Strategy (Gradient Clipping)
* Integrated norm-based **Gradient Clipping** to avoid exploding gradients caused by sharp softmax distributions in self-attention matrices and deep residual paths.


### 6. Manual Evaluation Metrics
- Evaluated model performance without `sklearn.metrics`. Built a scratch implementation for:
  - Confusion Matrix
  - Classification Accuracy
  - Macro Precision, Recall, and F1-Score

---

## 📊 Experimental Results & Ablation Studies

We conducted **four ablation experiments** to study the effect of various architectural hyperparameters over 10 epochs:

### Experiment A: Number of Attention Heads (1 vs 4)
- **1 Head:** Accuracy: `0.8957` | F1-Score: `0.8960`
- **4 Heads:** Accuracy: `0.8923` | F1-Score: `0.8925`

### Experiment B: Positional Encoding Impact
- **With PE:** Accuracy: `0.8946` | F1-Score: `0.8946`
- **Without PE:** Accuracy: `0.8925` | F1-Score: `0.8927`

### Experiment C: Embedding Dimension ($d_{model}$)
- **$d=64$:** Accuracy: `0.8780` | F1-Score: `0.8781`
- **$d=128$:** Accuracy: `0.8966` | F1-Score: `0.8968`
- **$d=256$:** Accuracy: `0.9057` | F1-Score: `0.9062`

### Experiment D: Depth (1 vs 2 Transformer Blocks)
- **1 Block:** Accuracy: `0.8948` | F1-Score: `0.8948`
- **2 Blocks:** Accuracy: `0.8973` | F1-Score: `0.8980`

---

## 🛠️ Environment & Reproducibility
To ensure the reproducibility of the results, the random seed was strictly fixed (`seed=42`). 

### Dependencies
```text
torch        >= 2.0.0
numpy        >= 1.24.0
pandas       >= 2.0.0
matplotlib   >= 3.7.0
kagglehub    >= 0.2.0
```
---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/noranali/transformer-text-classification-scratch.git
   cd transformer-text-classification-scratch
   ```

2. Install dependencies:
   ```bash
     pip install -r requirements.txt
    ```
3. Open and run the Jupyter notebook to see the data pipeline, training loops, and evaluation matrices:
     ```bash
      jupyter notebook transformer_scratch.ipynb
    ```
---

## 🎯 Final Baseline Performance
The optimized configuration yields highly competitive performance on the AG News test suite:

**Test Accuracy**: `89.70%`

**Macro F1-Score**: `89.71%`

```text
Confusion Matrix:

[[4005  160  199  173]

 [  67 4281   27   35]

 [ 150   74 3877  477]

 [ 158   68  265 3984]]
```

---

