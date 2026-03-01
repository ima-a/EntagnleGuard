# Workflow

## Pipeline Execution

Execute scripts in sequence from the `q-fraud/` directory:

```bash
cd q-fraud

# Step 1: Feature Selection
python scripts/feature_selection.py

# Step 2: Balance Dataset
python scripts/balance_dataset.py

# Step 3: Prepare Quantum Data
python scripts/prepare_quantum_data.py

# Step 4: Train Classical Baseline
python scripts/classical_baseline.py

# Step 5: Create Quantum Feature Map
python scripts/quantum_feature_map.py

# Step 6: Train Quantum VQC
python scripts/quantum_vqc.py

# Step 7: Compare Models
python scripts/compare_models.py

# Step 8: Generate Visualizations
python scripts/visualize_results.py

# Launch Dashboard
streamlit run dashboard.py
```

## Step-by-Step Breakdown

### Step 1: Feature Selection
**Script**: `feature_selection.py`
**Input**: `data/creditcard.csv`
**Output**: `data/quantum_selected_dataset.csv`, `config/selected_features.json`

Process:
1. Load raw credit card transaction data
2. Calculate Cohen's d for each V1-V28 feature
3. Rank features by discriminative power
4. Select top 4: V14, V4, V11, V12
5. Save reduced dataset

### Step 2: Balance Dataset
**Script**: `balance_dataset.py`
**Input**: `data/quantum_selected_dataset.csv`
**Output**: `data/balanced_creditcard.csv`

Process:
1. Count fraud cases (492)
2. Random sample 492 legitimate transactions
3. Combine into balanced 984-sample dataset
4. Shuffle and save

### Step 3: Prepare Quantum Data
**Script**: `prepare_quantum_data.py`
**Input**: `data/balanced_creditcard.csv`
**Output**: `data/X_train.npy`, `data/X_test.npy`, `data/y_train.npy`, `data/y_test.npy`

Process:
1. Split 80/20 train/test (stratified)
2. Apply StandardScaler (zero mean, unit variance)
3. Apply MinMaxScaler to [0, 2π] range
4. Save as NumPy arrays

### Step 4: Train Classical Baseline
**Script**: `classical_baseline.py`
**Input**: Processed NumPy arrays
**Output**: `models/classical_model.pkl`, `results/classical_metrics.json`

Process:
1. Load train/test data
2. Train Logistic Regression
3. Evaluate on test set
4. Save model and metrics

### Step 5: Create Quantum Feature Map
**Script**: `quantum_feature_map.py`
**Input**: None (configuration only)
**Output**: `circuits/zz_feature_map.png`

Process:
1. Create ZZFeatureMap (4 qubits, reps=3)
2. Decompose into gate-level circuit
3. Render and save visualization

### Step 6: Train Quantum VQC
**Script**: `quantum_vqc.py`
**Input**: Processed NumPy arrays
**Output**: `models/quantum_vqc.pkl`, `results/quantum_metrics.json`

Process:
1. Load train/test data
2. Create feature map (ZZFeatureMap)
3. Create ansatz (RealAmplitudes)
4. Initialize VQC with StatevectorSampler
5. Train with COBYLA (300 iterations)
6. Evaluate on test set
7. Save weights and metrics

### Step 7: Compare Models
**Script**: `compare_models.py`
**Input**: `results/classical_metrics.json`, `results/quantum_metrics.json`
**Output**: `results/comparison.json`, console output

Process:
1. Load both metric files
2. Calculate differences
3. Determine winner
4. Save comparison summary

### Step 8: Generate Visualizations
**Script**: `visualize_results.py`
**Input**: Metric JSON files
**Output**: `results/accuracy_comparison.png`, `results/metrics_comparison.png`, `circuits/ansatz.png`

Process:
1. Load metrics
2. Create accuracy bar chart
3. Create multi-metric comparison chart
4. Render ansatz circuit diagram
5. Save all visualizations

## Dashboard Launch

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501` with pages:
- **Fraud Predictor**: Input transaction features, get prediction
- **Quantum Circuits**: View ZZFeatureMap and ansatz diagrams
- **Model Comparison**: Interactive performance charts
- **About**: Team and project information

## Quick Run (All Steps)

```bash
cd q-fraud
for script in feature_selection balance_dataset prepare_quantum_data \
              classical_baseline quantum_feature_map quantum_vqc \
              compare_models visualize_results; do
    python scripts/${script}.py
done
streamlit run dashboard.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: qiskit` | `pip install qiskit qiskit-machine-learning qiskit-algorithms` |
| `FileNotFoundError: creditcard.csv` | Download from Kaggle: `kaggle datasets download mlg-ulb/creditcardfraud` |
| `Port 8501 in use` | `streamlit run dashboard.py --server.port 8502` |
| VQC training slow | Reduce `N_SAMPLES` or `MAXITER` in `quantum_vqc.py` |
