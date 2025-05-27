# MBTI Personality Prediction using NLP

Predict Myers‑Briggs Type Indicator (MBTI) personality types from a user’s text posts with classic Natural Language Processing and a Logistic Regression classifier.

---

## 📚 Project Overview

This notebook‑based project shows an *end‑to‑end* NLP workflow:

1. **Data Ingestion** – Upload the popular MBTI‑500 dataset (or any CSV with `posts` & `type` columns).
2. **Pre‑processing** – Lower‑casing, link/emoji removal, stop‑word filtering.
3. **Feature Extraction** – TF‑IDF (top 5 000 n‑grams).
4. **Modeling** – Multiclass Logistic Regression.
5. **Evaluation** – Accuracy & detailed classification report.
6. **Inference** – `predict_mbti()` helper for new text.
7. **Visualization** – Distribution of personality types.

> **Why Logistic Regression?**  It trains fast, is easy to interpret, and performs surprisingly well on sparse TF‑IDF features. You can swap in SVM, XGBoost, or a transformer (e.g. BERT) later.

---

## 🗂 Dataset

* **Source:** [Kaggle – MBTI 500](https://www.kaggle.com/datasnaek/mbti-type) (50 k+ Reddit/Forum posts labelled with 16 MBTI classes).
* **Columns**
  `type` – MBTI label (e.g. INTP)
  `posts` – User’s concatenated social‑media posts (pipe‑separated).

Feel free to use any dataset with the same two columns – the code is agnostic.

---

## 🏗 Quick Start (Google Colab)

```python
!pip install nltk scikit-learn seaborn pandas

from google.colab import files
uploaded = files.upload()            # ⬆️ upload your CSV

# run the rest of the notebook cells …
```

The notebook prints training metrics, a sample prediction, and a bar‑chart of class counts.

> **Tip:** Colab has a free GPU, but this model is CPU‑friendly.

---

## 🔨 Code Walk‑through

| Step | Purpose              | Key Functions                                        |
| ---- | -------------------- | ---------------------------------------------------- |
| 1    | **Import libs**      | `pandas`, `nltk`, `sklearn`, `matplotlib`, `seaborn` |
| 2    | **Load data**        | `pd.read_csv()`                                      |
| 3    | **Clean text**       | `re.sub`, NLTK `stopwords`                           |
| 4    | **Encode labels**    | `LabelEncoder()`                                     |
| 5    | **Vectorize**        | `TfidfVectorizer(max_features=5000)`                 |
| 6    | **Train/Test split** | `train_test_split(test_size=0.2)`                    |
| 7    | **Train model**      | `LogisticRegression(max_iter=1000)`                  |
| 8    | **Evaluate**         | `classification_report`, `accuracy_score`            |
| 9    | **Predict new text** | `predict_mbti()` helper                              |
| 10   | **Visualize**        | `seaborn.countplot()`                                |

---

## 🚀 Using `predict_mbti()`

```python
sample = "I enjoy deep conversations about abstract topics and often reflect on my thoughts."
print(predict_mbti(sample))  # ➜ e.g. "INFJ"
```

Embed this function into a Flask / FastAPI / Streamlit app to serve live predictions.

---

## 🔄 Extending the Project

* **Better Features** – N‑grams, character‑grams, or transformer embeddings.
* **Hyper‑parameter Tuning** – `GridSearchCV` for C, penalty, and class weights.
* **Model Zoo** – Compare SVM, Random Forest, or fine‑tuned BERT.
* **Explainability** – Show top TF‑IDF weights per class with `eli5`.
* **Web App** – Add a React or Next.js front‑end (Shashank’s specialty!).

---

## 🖼 Example Output

<div align="center">
  <img src="" width="480" ![WhatsApp Image 2025-05-27 at 12 46 58 PM](https://github.com/user-attachments/assets/84d7a652-d190-46fe-b636-38b312cc29dd)

</div>

---

## 📑 Requirements

* Python 3.8+
* pandas, numpy, scikit‑learn, nltk, seaborn, matplotlib

> ```bash
> pip install -r requirements.txt
> ```

Create *requirements.txt* via `pip freeze > requirements.txt` after testing.

---
