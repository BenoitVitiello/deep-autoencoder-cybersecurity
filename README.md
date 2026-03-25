# Network Intrusion Detection with Deep Autoencoders

Unsupervised anomaly detection on the UNSW-NB15 dataset using PyTorch. The autoencoder is trained exclusively on normal traffic and learns to flag attacks as anomalies.

**Results**: AUC 0.796, which beats standard unsupervised methods like Isolation Forest (0.68) and One-Class SVM (0.65) on this dataset.

---

## Why This Project?

Most intrusion detection systems rely on supervised learning—they need labeled examples of attacks to train. The problem is that new attack types emerge constantly, and getting labeled data is expensive and time-consuming.

This project explores an alternative : training a model on normal traffic only, then flagging anything that deviates significantly as a potential attack. The advantage is that it can theoretically detect zero-day attacks that the model has never seen before.

---

## Results

Here's how the deep autoencoder compares to other unsupervised methods on UNSW-NB15:

| Method                          | AUC   | Recall | Precision | FPR   |
| ------------------------------- | ----- | ------ | --------- | ----- |
| Deep Autoencoder (this project) | 0.796 | 76.9%  | 90.5%     | 33.1% |
| Isolation Forest                | 0.68  | ~55%   | ~80%      | ~50%  |
| One-Class SVM                   | 0.65  | ~50%   | ~80%      | ~55%  |

The model detects about 77% of attacks with 90% precision. That means when it flags something as an attack, it's right 9 times out of 10.

For comparison, supervised methods (Random Forest, XGBoost) get 90-95% recall on this dataset, but they can't detect attack types they weren't trained on.

**Trade-off**: The false positive rate is 33%, which means roughly 1 in 3 normal connections gets flagged. This is typical for unsupervised anomaly detection and better than the ~50% FPR of traditional methods.

---

## How It Works

### Architecture

The model is a deep autoencoder with 6 encoder layers and 6 decoder layers:

```
Input (33 features)
  → 128 → 96 → 64 → 32 → 16 → 8 (bottleneck)
  → 16 → 32 → 64 → 96 → 128
  → Output (33 reconstructed features)
```

I also used BatchNorm between layers and Dropout in the encoder.

### Training

- Trained only on normal network traffic (48k samples)
- Validation set: 16k normal samples (important: no attacks in validation for unsupervised learning)
- Test set: 16k normal + 66k attacks
- Loss: Mean Squared Error on reconstruction
- Optimizer: Adam with learning rate 0.001

The idea is simple: the model learns to reconstruct normal traffic with low error. When it sees an attack, the reconstruction error is high because the pattern is unfamiliar.

### Threshold Selection

This was actually the tricky part. The default approach (using median reconstruction error as threshold) gave terrible results because the test set has way more attacks than normal samples, so the median is biased.

I ended up testing different strategies:

- **Best F1-Score** : Balances precision and recall
- **Security-focused** : Maximize recall (catch as many attacks as possible), accept higher false positives
- **Conservative** : Keep false positives low, accept lower recall

For cybersecurity applications, I went with the security-focused approach (77% recall). The conservative approach (24% FPR) only gets 68% recall, which misses too many attacks.

---

## Dataset

UNSW-NB15 is a network traffic dataset from UNSW Canberra (2015). It's more realistic than older datasets like KDD Cup 99.

- **Total samples**: 257k (175k training + 82k testing)
- **Features**: 49 (I used 33 after preprocessing)
- **Attack types**: 9 categories (DoS, Exploits, Fuzzers, etc.)
- **Normal/Attack ratio**: ~30% normal, ~70% attacks

The dataset is available on Kaggle: [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)

### Quick Start

1. Download the dataset from Kaggle and place the parquet files in `data/archive/`

2. Install dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

3. Run the notebooks in order:
   - `01_data_exploration.ipynb` - EDA
   - `02_preprocessing.ipynb` - Feature engineering and train/test split
   - `03_model_training.ipynb` - Train the autoencoder
   - `04_evaluation.ipynb` - Evaluate and optimize threshold

---

## Project Structure

```
├── notebooks/              # Jupyter notebooks for the full pipeline
├── src/                    # Reusable Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── archive/            # Original UNSW-NB15 files
│   └── processed/          # Preprocessed train/val/test splits
└── models/
    └── autoencoder.pth     # Trained model checkpoint
```

---

## What I Learned

### Data Issues Matter More Than Model Architecture

I spent way too much time tweaking the model before realizing my validation set was contaminated with attacks. Once I fixed that (validation = 100% normal for unsupervised learning), performance jumped significantly.

Also had to deal with 71k duplicate samples in the test set that were inflating metrics. Removing them dropped AUC from 0.77 to 0.69 initially, but gave a more honest evaluation.

### Dropout Placement

Dropout in the encoder helps with generalization, but dropout in the decoder actively hurts reconstruction quality. Took me a while to figure that out—removing decoder dropout improved separation between normal and attack errors.

### Threshold Optimization is Non-Trivial

You can't just use a fixed percentile or the median. The optimal threshold depends heavily on what you're optimizing for:

- Security applications: lower threshold, higher recall, more false positives
- Operations with limited analyst time: higher threshold, lower false positives, miss some attacks

For this project, I prioritized recall (security-focused) since missing attacks is worse than investigating false alarms.

### Unsupervised vs Supervised

The ~77% recall might seem low compared to supervised methods (90-95%), but that's not a fair comparison:

- Supervised: trained on labeled attack examples, can't detect new attack types
- Unsupervised: trained on normal only, can potentially detect zero-day attacks

In practice, you'd probably use both—supervised for known attacks, unsupervised as a safety net.

---

## Potential Improvements

Things I considered but didn't implement (mostly due to time constraints):

1. **Variational Autoencoder (VAE)**: Should give better latent representations than a standard autoencoder
2. **Feature Engineering**: The 33 features are pretty basic. Adding things like packet rate ratios, byte-per-packet metrics, etc. could improve separation
3. **Ensemble**: Train 5 models with different random seeds and average their predictions
4. **Hybrid Approach**: Use reconstruction error as a feature in a supervised classifier

The VAE would probably give the biggest improvement (~5-10% AUC based on literature.

---

## References

**Dataset**:

- Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS 2015.

**Related Work**:

- Isolation Forest comparison: Liu et al. (2019), AUC 0.68 on UNSW-NB15
- One-Class SVM: Various papers report 0.65-0.70 AUC on similar datasets

---

## Contact

If you have questions or suggestions, feel free to reach out:

- **GitHub**: [@BenoitVitiello](https://github.com/BenoitVitiello)
- **LinkedIn**: [Benoît Vitiello](https://linkedin.com/in/benoit-vitiello)
- **Email**: vitiello.benoit@icloud.com
