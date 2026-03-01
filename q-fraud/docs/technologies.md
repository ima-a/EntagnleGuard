# Technologies

## Core Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Programming language |
| **Qiskit** | 2.3.0 | Quantum computing framework |
| **qiskit-machine-learning** | Latest | VQC implementation |
| **qiskit-algorithms** | Latest | COBYLA optimizer |
| **scikit-learn** | 1.x | Classical ML, preprocessing |
| **Streamlit** | 1.x | Interactive dashboard |
| **pandas** | 2.x | Data manipulation |
| **NumPy** | 1.x | Numerical computing |
| **Matplotlib** | 3.x | Visualizations |

## Quantum Components

### Qiskit
IBM's open-source quantum computing SDK providing:
- Circuit construction and manipulation
- Quantum primitives (StatevectorSampler)
- Circuit visualization tools

### ZZFeatureMap
Encodes classical data into quantum states using:
- Hadamard gates for superposition
- Phase gates P(2xᵢ) for single-qubit encoding
- ZZ interactions for feature entanglement
- Multiple repetitions (reps=3) for expressivity

### RealAmplitudes Ansatz
Parameterized circuit for variational learning:
- RY rotation gates with trainable parameters
- CNOT gates for entanglement
- Full entanglement topology
- 3 repetition layers

### StatevectorSampler
Qiskit primitive for quantum circuit simulation:
- Exact statevector computation
- Deterministic results (no shot noise)
- Efficient for small circuits

### COBYLA Optimizer
Constrained Optimization BY Linear Approximations:
- Derivative-free optimization
- Suitable for noisy quantum landscapes
- 300 iterations for convergence

## Classical Components

### scikit-learn
- **LogisticRegression**: Classical baseline classifier
- **StandardScaler**: Zero mean, unit variance normalization
- **MinMaxScaler**: Range scaling to [0, 2π]
- **train_test_split**: Stratified data splitting
- **Metrics**: accuracy, precision, recall, F1

### pandas
- CSV loading and manipulation
- DataFrame operations
- Data balancing via sampling

### NumPy
- Array operations
- .npy file I/O
- Numerical computations

## Visualization

### Matplotlib
- Bar charts for model comparison
- Circuit diagrams via Qiskit integration
- Custom dark theme styling

### Streamlit
- Multi-page dashboard framework
- Interactive widgets (sliders, buttons)
- Real-time data display
- Session state management

## Data Source

### Kaggle Credit Card Fraud Dataset
- **Source**: European cardholders, September 2013
- **Size**: 284,807 transactions
- **Features**: V1-V28 (PCA transformed for privacy)
- **Imbalance**: 0.172% fraud (492 cases)
- **License**: Open Database License (ODbL)

## Development Tools

| Tool | Purpose |
|------|---------|
| **pip** | Package management |
| **venv** | Virtual environments |
| **joblib** | Model serialization |
| **JSON** | Configuration storage |
| **Git** | Version control |

## System Requirements

- **Python**: 3.10 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~500MB for dataset + models
- **OS**: Linux, macOS, or Windows

## Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install qiskit qiskit-machine-learning qiskit-algorithms
pip install scikit-learn pandas numpy matplotlib
pip install streamlit kaggle joblib
```
