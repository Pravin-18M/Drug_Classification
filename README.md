# Drug Classification — ML CI/CD Pipeline & Static ONNX Runtime Web App

Complete CI/CD implementation for a Machine Learning project using **GitHub Actions**, **CML (Continuous Machine Learning)**, and a static **ONNX Runtime Web** inference web application hosted on **Hugging Face Spaces**.

---

## 🎯 Target Architecture

- **Code & Automation**: GitHub repository with GitHub Actions CI/CD workflow.
- **Model Training & Conversion**: Automated Random Forest pipeline training with Scikit-learn and export to ONNX (`skl2onnx`).
- **Reporting**: Automated evaluation metrics and confusion matrix generated and posted as pull request comments via CML.
- **Hosting & Inference**: Serverless, static HTML application hosted on Hugging Face Static Spaces with zero-latency, zero-cost, in-browser inference powered by `onnxruntime-web`.

---

## 📁 Project Directory Structure

```
Drug_Classification/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI and CD workflow
├── app/
│   ├── index.html              # Static HTML & ONNX Runtime Web client interface
│   └── README.md               # Hugging Face Space metadata configuration
├── data/
│   └── drug200.csv             # Training dataset
├── model/
│   ├── drug_model.joblib       # Original Scikit-learn model artifact
│   └── drug_model.onnx         # Optimized portable ONNX model for browser inference
├── results/
│   ├── metrics.txt             # Accuracy, F1-score & classification report
│   └── confusion_matrix.png    # Confusion matrix visualization
├── train.py                    # Model training, evaluation, and ONNX conversion script
├── test_inference.py           # Verification script comparing Python and ONNX outputs
├── requirements.txt            # Python dependencies for training and CI runner
├── Makefile                    # Automation targets (install, format, train, eval, all)
├── .gitignore                  # Git ignore rules for Python artifacts
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Local Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Format & Lint Check
```bash
black --check train.py
```

### 3. Train Model & Export ONNX
```bash
python train.py
```

### 4. Verify Local Inference vs ONNX
```bash
python test_inference.py
```

### 5. Automated Local Pipeline with Make
```bash
make all
```

---

## 🚀 CI/CD Pipeline (GitHub Actions)

When code is pushed to `main` or in a pull request:
1. **Source Checkout**: Checks out the repository.
2. **Setup Python**: Initializes Python 3.11 environment.
3. **Setup CML**: Prepares Continuous Machine Learning tools.
4. **Install Dependencies**: Executes `make install`.
5. **Format Check**: Executes `make format` (`black --check train.py`).
6. **Model Training**: Executes `make train` (generates `drug_model.joblib` and `drug_model.onnx`).
7. **CML Report**: Creates a markdown evaluation report with metrics and the confusion matrix image and posts it to the GitHub commit/PR.
8. **Hugging Face Deployment**: On push to `main`, uploads static assets and model artifacts to Hugging Face Static Space using `huggingface_hub`.

---

## 🔑 GitHub Secrets Configuration

Under **Settings → Secrets and variables → Actions**, configure:
- `HF_TOKEN`: Hugging Face User Access Token with **write** permissions.
- `HF_SPACE_ID`: Target Space identifier (e.g. `YOUR_USERNAME/drug-classification-lite`).

---

## 🌐 Browser Inference Verification

The web interface loads `model/drug_model.onnx` directly into WebAssembly through `onnxruntime-web` to execute the full pipeline (categorical encoding + Random Forest prediction) client-side without any server requirements.
