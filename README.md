# GLIMMER

**Glucose Level Indicator Model with Modified Error Rate**

Managing Type 1 Diabetes (T1D) demands constant vigilance as individuals strive to regulate their blood glucose levels to avert the dangers of dysglycemia (i.e., hyperglycemia and hypoglycemia). Despite the advent of sophisticated technologies such as automated insulin delivery (AID) systems, achieving optimal glycemic control remains a formidable task. AID systems integrate data from wearable devices, including continuous subcutaneous insulin infusion (CSII) pumps and continuous glucose monitors (CGMs), offering promise in reducing variability and improving time-in-range.

However, these systems often fail to prevent dysglycemia, partly due to limitations in prediction algorithms that lack the precision to anticipate impending glycemic excursions. This gap highlights the need for more advanced blood glucose forecasting methods.

We address this need with **GLIMMER** — *Glucose Level Indicator Model with Modified Error Rate* — an architecture-agnostic and modular machine learning framework for improved forecasting accuracy. GLIMMER integrates a custom loss function that emphasizes accuracy during dysglycemia by optimizing region-specific penalties through a genetic algorithm.

We evaluate GLIMMER on two datasets: the publicly available **OhioT1DM** dataset and a newly collected dataset (**AZT1D**) involving 25 individuals with T1D. Our extensive analyses show that GLIMMER consistently improves glucose forecasting performance over baseline architectures, enhancing RMSE (Root-Mean-Square Error) and MAE (Mean-Absolute-Error) by up to **24.6%** and **29.6%**, respectively. Additionally, GLIMMER achieves a **recall of 98.4%** and an **F1-score of 86.8%** in predicting dysglycemic events, demonstrating its effectiveness in high-risk regions.

Compared to state-of-the-art models with millions of parameters—such as TimesNet (18.7M), BG-BERT (2.1M), and Gluformer (11.2M)—GLIMMER achieves comparable accuracy while using only **10K parameters**, demonstrating its efficiency as a lightweight, architecture-agnostic solution for glycemic forecasting.

---

## 📁 Dataset Setup

### 1. OhioT1DM Dataset

- **Option A (official):**  
  Request the dataset from the official source:  
  https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html

- **Option B (Kaggle):**  
  Download from Kaggle:  
  https://www.kaggle.com/datasets/ryanmouton/ohiot1dm

- **Directory structure:**  
  Place the `.xlsx` files in:

  ```
  OhioT1DM/raw_data/{2018 or 2020}/{train or test}/
  ```

- **Preprocess the dataset:**

  ```bash
  python dataset/OhioT1DM/preprocess/main.py
  ```

---

### 2. AZT1D Dataset

- **Download from Mendeley:**  
  https://data.mendeley.com/datasets/gk9m674wcx/1

- **Directory structure:**  
  Place the files in the already created folder:

  ```
  AZT1D/
  ```

---

## ⚙️ Environment Setup

- **Python version:** `3.10`

- **Install dependencies:**

  Create a virtual environment (optional but recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

  Then install required packages:

  ```bash
  pip install -r requirements.txt
  ```

  `requirements.txt` includes:

  ```
  matplotlib==3.9.2
  numpy==2.1.3
  pandas==2.2.3
  scikit_learn==1.5.2
  scipy==1.14.1
  seaborn==0.13.2
  tensorflow==2.18.0
  ```

---

## 🚀 Run the Code

To start training and evaluating the model, run:

```bash
python main.py
```

Before running, make sure to manually set the model type in `main.py` (e.g., `"transformer"` or `"cnn_lstm"`).

---

## 📖 Citation

If you use GLIMMER in your work, please cite:

```bibtex
Khamesian, S., Arefeen, A., Grando, M. et al. Glycemic-aware and architecture-agnostic training framework for blood glucose forecasting in type 1 diabetes. BMC Artif. Intell. 2, 13 (2026). https://doi.org/10.1186/s44398-026-00032-x
```

---
