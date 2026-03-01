# Q-Fraud: Complete Project Guide for Judges

**Team: Volunteers | Hackathon 2026**

---

## 🎯 Executive Summary

Q-Fraud is a **quantum-classical hybrid machine learning system** that detects fraudulent credit card transactions using IBM Qiskit's Variational Quantum Classifier (VQC). The project demonstrates how quantum computing can potentially outperform classical algorithms on classification tasks with small, high-dimensional datasets.

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Our Solution](#-our-solution)
3. [How It Works (Simple Explanation)](#-how-it-works-simple-explanation)
4. [Technical Deep Dive](#-technical-deep-dive)
5. [Project Structure](#-project-structure)
6. [Key Files Explained](#-key-files-explained)
7. [The Data Pipeline](#-the-data-pipeline)
8. [Quantum Concepts Used](#-quantum-concepts-used)
9. [Classical vs Quantum Comparison](#-classical-vs-quantum-comparison)
10. [How to Run](#-how-to-run)
11. [Potential Judge Questions & Answers](#-potential-judge-questions--answers)
12. [What Makes This Project Special](#-what-makes-this-project-special)
13. [Challenges We Faced](#-challenges-we-faced)
14. [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

**Credit card fraud** costs the global economy over $30 billion annually. Traditional machine learning models struggle with:

1. **Extreme class imbalance**: Only 0.17% of transactions are fraudulent
2. **High dimensionality**: Many features to process
3. **Subtle patterns**: Fraudsters constantly evolve their techniques

**Our Question**: Can quantum computing detect fraud patterns that classical algorithms miss?

---

## 💡 Our Solution

We built a complete fraud detection system that:

1. **Processes real Kaggle data** (284,807 transactions)
2. **Uses quantum feature maps** to encode data into quantum states
3. **Trains a Variational Quantum Classifier** (VQC)
4. **Compares against classical Logistic Regression**
5. **Provides an interactive Streamlit dashboard**

---

## 🧠 How It Works (Simple Explanation)

### For Non-Technical Judges

Think of it like this:

1. **Classical ML** looks at transaction features one by one, like reading a book page by page
2. **Quantum ML** looks at ALL features simultaneously via **superposition**, like seeing the entire book at once
3. **Entanglement** lets the quantum model find hidden relationships between features that classical models might miss

### The Pipeline (5-Minute Overview)

```
Raw Data (284,807 transactions)
        ↓
Select Best 4 Features (Cohen's d statistic)
        ↓
Balance Data (492 fraud + 492 legit = 984)
        ↓
Scale to [0, 2π] (quantum gates need angles)
        ↓
┌─────────────────┬─────────────────────┐
│ Classical Model │   Quantum Model     │
│ (Logistic Reg)  │   (VQC)             │
└─────────────────┴─────────────────────┘
        ↓                   ↓
      Compare Results → Dashboard
```

---

## 🔬 Technical Deep Dive

### 1. Feature Selection (Cohen's d)

We use **Cohen's d** to measure how different fraud and legitimate transactions are for each feature:

```
Cohen's d = (mean_fraud - mean_legit) / pooled_std
```

- **|d| > 0.8** = Large effect (very discriminative)
- **|d| > 0.5** = Medium effect
- **|d| > 0.2** = Small effect

**Selected features**: V14, V4, V11, V12 (highest |Cohen's d|)

**Why 4 features?** Quantum computers have limited qubits. 4 qubits = 4 features = 2⁴ = 16 possible states.

### 2. Data Balancing (Undersampling)

Original data: 284,315 legit vs 492 fraud (0.17% fraud)

**Problem**: Model would just predict "legit" always and be 99.83% accurate!

**Solution**: Undersample majority class → 492 legit + 492 fraud = 984 balanced samples

### 3. Quantum Data Preparation

Classical data → Quantum angles:

```python
# Step 1: Standardize (mean=0, std=1)
X_std = StandardScaler().fit_transform(X)

# Step 2: Scale to [0, 2π] for rotation gates
X_quantum = MinMaxScaler(feature_range=(0, 2*π)).fit_transform(X_std)
```

**Why [0, 2π]?** Quantum rotation gates (RY, RZ) use angles. A full rotation = 2π radians.

### 4. Quantum Feature Map (ZZFeatureMap)

Encodes classical data into quantum states using:

```
|ψ(x)⟩ = U_Φ(x) |0...0⟩
```

**Circuit structure**:
1. **Hadamard gates**: Create superposition (|0⟩ + |1⟩)/√2
2. **P(2xᵢ) gates**: Encode feature values as phases
3. **ZZ interactions**: Entangle qubits based on feature products

```
|q₀⟩ ─ H ─ P(2x₁) ─ ZZ(x₁·x₂) ─ ...
|q₁⟩ ─ H ─ P(2x₂) ─ ZZ(x₁·x₃) ─ ...
|q₂⟩ ─ H ─ P(2x₃) ─ ZZ(x₂·x₃) ─ ...
|q₃⟩ ─ H ─ P(2x₄) ─ ZZ(x₃·x₄) ─ ...
```

**Repetitions (reps=3)**: The circuit is repeated 3 times for greater expressivity.

### 5. Ansatz (RealAmplitudes)

The **trainable** part of the quantum circuit:

```
|q₀⟩ ─ RY(θ₀) ─ CX ─ RY(θ₄) ─ CX ─ RY(θ₈) ─ ...
|q₁⟩ ─ RY(θ₁) ─ ●  ─ RY(θ₅) ─ ●  ─ RY(θ₉) ─ ...
|q₂⟩ ─ RY(θ₂) ─ CX ─ RY(θ₆) ─ CX ─ RY(θ₁₀)─ ...
|q₃⟩ ─ RY(θ₃) ─ ●  ─ RY(θ₇) ─ ●  ─ RY(θ₁₁)─ ...
```

- **RY(θ)**: Rotation gates with learnable parameters
- **CX (CNOT)**: Entanglement gates
- **Full entanglement**: Every qubit connected to every other

**Total parameters**: 4 qubits × 3 reps × some connections ≈ 16-20 learnable θ values

### 6. Optimization (COBYLA)

**COBYLA** = Constrained Optimization BY Linear Approximations

- **Derivative-free**: Doesn't need gradients
- **Works well for noisy quantum landscapes**
- **300 iterations**: Enough for convergence

**Loss function**: Binary cross-entropy (minimize prediction errors)

### 7. Measurement & Classification

1. Run quantum circuit with input features
2. Measure qubit states
3. Interpret measurement as probability: P(fraud) vs P(legit)
4. Threshold at 0.5 for final prediction

---

## 📁 Project Structure

```
q-fraud/
├── dashboard.py              # Streamlit web app
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
│
├── scripts/                  # Pipeline scripts
│   ├── run_pipeline.py       # ⭐ Run everything at once
│   ├── feature_selection.py  # Step 1: Cohen's d selection
│   ├── balance_dataset.py    # Step 2: Undersample majority
│   ├── prepare_quantum_data.py # Step 3: Scale to [0, 2π]
│   ├── classical_baseline.py # Step 4: Train Logistic Regression
│   ├── quantum_feature_map.py # Step 5: Visualize ZZFeatureMap
│   ├── quantum_vqc.py        # Step 6: Train VQC (slow!)
│   ├── compare_models.py     # Step 7: Compare metrics
│   └── visualize_results.py  # Step 8: Generate charts
│
├── data/                     # Datasets
│   ├── creditcard.csv        # Raw Kaggle data (284,807 rows)
│   ├── quantum_selected_dataset.csv  # 4 features only
│   ├── balanced_creditcard.csv       # 984 balanced samples
│   ├── X_train.npy, X_test.npy       # Processed features
│   └── y_train.npy, y_test.npy       # Labels
│
├── models/                   # Saved models
│   ├── classical_model.pkl   # Trained Logistic Regression
│   └── quantum_vqc.pkl       # VQC weights
│
├── circuits/                 # Quantum circuit diagrams
│   ├── zz_feature_map.png    # Data encoding circuit
│   └── ansatz.png            # Trainable circuit
│
├── results/                  # Outputs
│   ├── classical_metrics.json
│   ├── quantum_metrics.json
│   ├── comparison.json
│   ├── accuracy_comparison.png
│   └── metrics_comparison.png
│
├── config/                   # Configuration
│   └── selected_features.json  # ["V14", "V4", "V11", "V12"]
│
└── docs/                     # Documentation
    ├── JUDGES_GUIDE.md       # ⭐ This file!
    ├── architecture.md
    ├── technologies.md
    ├── workflow.md
    └── eda_report.md
```

---

## 📄 Key Files Explained

### `dashboard.py` (556 lines)
The main user interface with 4 pages:
- **Fraud Predictor**: Enter transaction values, get prediction
- **Quantum Circuits**: View feature map and ansatz diagrams
- **Model Comparison**: See accuracy charts
- **About**: Project information

### `quantum_vqc.py` (Most Important!)
```python
# Key configuration
N_SAMPLES = 500      # Training samples
MAXITER = 300        # Optimizer iterations
REPS = 3             # Circuit repetitions
ENTANGLEMENT = "full" # All qubits connected

# The VQC
vqc = VQC(
    sampler=StatevectorSampler(),  # Exact quantum simulation
    feature_map=ZZFeatureMap(...), # Data encoding
    ansatz=RealAmplitudes(...),    # Trainable circuit
    optimizer=COBYLA(maxiter=300), # Classical optimizer
)
```

### `feature_selection.py`
```python
def compute_cohens_d(df, feature):
    fraud = df[df["Class"] == 1][feature]
    legit = df[df["Class"] == 0][feature]
    pooled_std = np.sqrt((fraud.std()**2 + legit.std()**2) / 2)
    return (fraud.mean() - legit.mean()) / pooled_std
```

---

## 🔄 The Data Pipeline

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| 1 | creditcard.csv | quantum_selected_dataset.csv | Select 4 best features |
| 2 | quantum_selected_dataset.csv | balanced_creditcard.csv | Balance 50/50 |
| 3 | balanced_creditcard.csv | X_train.npy, etc. | Scale for quantum |
| 4 | X_train.npy | classical_model.pkl | Train baseline |
| 5 | Config | zz_feature_map.png | Visualize encoding |
| 6 | X_train.npy | quantum_vqc.pkl | Train VQC |
| 7 | Both metrics | comparison.json | Compare models |
| 8 | Results | PNG charts | Visualize |

---

## ⚛️ Quantum Concepts Used

### 1. Superposition
A qubit can be in states |0⟩ AND |1⟩ simultaneously:
```
|ψ⟩ = α|0⟩ + β|1⟩
```
**Our use**: Hadamard gates create superposition to explore all feature combinations.

### 2. Entanglement
Qubits become correlated - measuring one affects others:
```
|Φ⁺⟩ = (|00⟩ + |11⟩)/√2
```
**Our use**: ZZ gates and CNOT gates create entanglement to capture feature relationships.

### 3. Quantum Interference
Probability amplitudes can add or cancel:
```
|ψ⟩ = (|0⟩ + |1⟩)/√2 + (|0⟩ - |1⟩)/√2 = |0⟩
```
**Our use**: The circuit structure creates interference patterns that separate fraud from legit.

### 4. Parameterized Quantum Circuits
Rotation gates with learnable angles:
```
RY(θ) = [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]
```
**Our use**: COBYLA optimizer adjusts θ values to minimize classification error.

---

## 📊 Classical vs Quantum Comparison

| Aspect | Classical (LR) | Quantum (VQC) |
|--------|----------------|---------------|
| **Algorithm** | Logistic Regression | Variational Quantum Classifier |
| **Parameters** | 5 (weights + bias) | ~16-20 (rotation angles) |
| **Training time** | <1 second | 5-10 minutes |
| **Feature space** | Linear | Exponential (Hilbert space) |
| **Interpretability** | High | Low (black box) |
| **Scalability** | Excellent | Limited by qubit count |

**Why VQC might win**:
- Quantum computers access exponentially larger feature spaces
- ZZ entanglement captures nonlinear feature interactions
- Small datasets benefit from quantum's generalization

---

## 🚀 How to Run

### Quick Start (Models Already Trained)
```bash
cd q-fraud
python scripts/run_pipeline.py  # Skips training if models exist
streamlit run dashboard.py
```

### Full Pipeline (Retrain Everything)
```bash
cd q-fraud
python scripts/run_pipeline.py --force
streamlit run dashboard.py
```

### Individual Steps
```bash
python scripts/feature_selection.py
python scripts/balance_dataset.py
python scripts/prepare_quantum_data.py
python scripts/classical_baseline.py
python scripts/quantum_feature_map.py
python scripts/quantum_vqc.py        # Takes 5-10 minutes!
python scripts/compare_models.py
python scripts/visualize_results.py
```

---

## ❓ Potential Judge Questions & Answers

### Q1: "Why use quantum computing for fraud detection?"
**A**: Quantum computers can explore exponentially larger feature spaces via superposition and entanglement. For high-dimensional data with subtle patterns, this can reveal correlations that classical linear models miss.

### Q2: "What is a Variational Quantum Classifier?"
**A**: A VQC is a hybrid quantum-classical algorithm. A quantum circuit (parameterized by angles θ) processes the data, then a classical optimizer adjusts θ to minimize prediction error. It's like neural networks, but with quantum gates instead of neurons.

### Q3: "Why did you choose only 4 features?"
**A**: Current quantum simulators are limited. 4 qubits = 2⁴ = 16-dimensional Hilbert space. More qubits would be better but slower. We chose the 4 most discriminative features using Cohen's d statistic.

### Q4: "What is Cohen's d?"
**A**: A measure of effect size. It tells us how different two groups are in terms of standard deviations. |d| > 0.8 means a large, meaningful difference between fraud and legitimate transactions for that feature.

### Q5: "What is ZZFeatureMap?"
**A**: A quantum circuit that encodes classical data into quantum states. It uses:
- Hadamard gates for superposition
- Phase gates P(2x) to encode feature values
- ZZ interactions to entangle features (capturing pairwise relationships)

### Q6: "What is RealAmplitudes ansatz?"
**A**: The trainable part of our quantum circuit. It has:
- RY rotation gates with learnable angles
- CNOT gates for entanglement
- The optimizer tunes these angles to minimize classification error

### Q7: "Why scale data to [0, 2π]?"
**A**: Quantum rotation gates use angles. RY(θ) rotates by θ radians. The full circle is 2π, so we scale features to use the full expressive power of the gates.

### Q8: "What optimizer did you use?"
**A**: COBYLA (Constrained Optimization BY Linear Approximations). It's derivative-free, which is important because quantum circuits have noisy, non-smooth loss landscapes where gradient descent struggles.

### Q9: "Why undersample instead of oversample?"
**A**: Oversampling (SMOTE) creates synthetic fraud samples, which might not reflect real fraud patterns. Undersampling uses only real data, giving more authentic training. With quantum's strong generalization, we don't need many samples.

### Q10: "What's the difference between simulator and real quantum hardware?"
**A**: We use `StatevectorSampler` (exact simulation). Real hardware has:
- Noise and errors
- Limited connectivity between qubits
- Decoherence (qubits lose their state)
Our code is hardware-ready but would need error mitigation on real devices.

### Q11: "How does the dashboard work?"
**A**: Streamlit is a Python framework for data apps. Our dashboard:
1. Loads trained models
2. Takes user input (transaction features)
3. Runs the quantum circuit
4. Returns fraud/legit prediction with confidence

### Q12: "What libraries did you use?"
**A**:
- **Qiskit 2.3.0**: IBM's quantum SDK
- **qiskit-machine-learning**: VQC implementation
- **scikit-learn**: Classical baseline, preprocessing
- **Streamlit**: Web dashboard
- **pandas/numpy**: Data manipulation

### Q13: "What challenges did you face?"
**A**:
1. **Import errors**: Qiskit 2.x changed from `Sampler` to `StatevectorSampler`
2. **Training time**: VQC takes 5-10 minutes vs seconds for classical
3. **Model saving**: VQC can't be pickled directly, so we save weights separately
4. **Class imbalance**: 0.17% fraud required careful balancing

### Q14: "What would you do with more time?"
**A**:
1. Test on real IBM quantum hardware
2. Try different feature maps (PauliFeatureMap, custom)
3. Implement quantum error mitigation
4. Use more qubits (8+) with more features
5. Compare with quantum kernel methods (QSVC)

---

## ⭐ What Makes This Project Special

1. **End-to-end pipeline**: Raw data → trained models → interactive dashboard
2. **Real-world data**: Kaggle's credit card fraud dataset (284,807 transactions)
3. **Fair comparison**: Same features, same split, same metrics for classical vs quantum
4. **Production-ready**: Streamlit dashboard for non-technical users
5. **Documented**: Comprehensive docs for reproducibility
6. **Quantum advantage potential**: VQC specifically designed for small, imbalanced datasets

---

## 🚧 Challenges We Faced

| Challenge | Solution |
|-----------|----------|
| Qiskit 2.x breaking changes | Switched to `StatevectorSampler` |
| VQC training takes forever | Reduced samples (500), cached models |
| Can't pickle VQC object | Save weights dict instead |
| Class imbalance (0.17% fraud) | Undersampling to 50/50 |
| Too many features for qubits | Cohen's d selection → 4 features |
| Dashboard crashes | Added error handling, fallback values |

---

## 🔮 Future Improvements

1. **Run on real IBM Q hardware** via `IBMBackend`
2. **Quantum error mitigation** (ZNE, PEC)
3. **More qubits** (8-16) for more features
4. **Ensemble methods** (combine classical + quantum)
5. **Quantum kernel SVM** as alternative approach
6. **Real-time inference** via quantum cloud API
7. **Explainability** tools for quantum decisions

---

## 📚 References

- [Qiskit Documentation](https://qiskit.org/)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Variational Quantum Classifier Paper](https://arxiv.org/abs/1804.11326)
- [ZZFeatureMap Paper](https://arxiv.org/abs/1804.11326)

---

**Good luck with your presentation!** 🎉

*Team Volunteers - Hackathon 2026*
