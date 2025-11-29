# Phishing Detection Project (KnowPhish Clone)

This project collects webpage HTML and screenshots, applies adversarial perturbations, and trains a machine-learning model to classify URLs as **phishing or legitimate**.

---

## How It Works

### **1. technique1.py**
- Visits each URL
- Saves HTML + screenshot
- Applies harmless text perturbations
- Measures how much predictions change

### **2. technique2.py**
- Re-fetches URLs
- Saves HTML + screenshot
- Computes similarity-based features

### **3. train_knowphish_model.py**
- Loads collected HTML
- Extracts simple text features
- Trains a RandomForest model
- Saves model as `model.pkl`

### **4. common.py**
- Shared helper functions
- URL hashing for filenames
- HTML + screenshot saving
- Unified prediction interface

---

## How to Run

```bash
python3 technique1.py
python3 technique2.py
python3 train_knowphish_model.py


Output Files
data/html/ — saved webpage HTML
data/screenshots/ — screenshots
technique1_results.csv
technique2_results.csv
model.pkl (trained model)