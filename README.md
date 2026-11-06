# 📰 Fake News Detection System

An end-to-end **Natural Language Processing (NLP)** and **Machine Learning** project that classifies news articles as **Fake** or **Real** using **TF-IDF** and **Scikit-learn**, with an interactive **Streamlit** web application.

---

## 🚀 Features

- Text preprocessing using NLTK
- TF-IDF feature extraction (Unigrams + Bigrams)
- Comparison of multiple ML algorithms
- Hyperparameter tuning using GridSearchCV
- Interactive Streamlit web application
- Single article prediction
- Batch CSV prediction
- Performance dashboard
- Download prediction results
- Clean and modular project structure

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│
├── models/
│   ├── model.pkl
│   ├── tfidf.pkl
│   └── reports/
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

Dataset used:

**ISOT Fake News Dataset**

- Fake News Articles
- True News Articles

Total Articles: **~45,000**

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ANUMANDLA-GANESH/Fake-News-Detection.git
```

Move into the project

```bash
cd Fake-News-Detection
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Preprocess Dataset

```bash
python src/preprocess.py
```

### Train Model

```bash
python src/train.py
```

### Evaluate Model

```bash
python src/evaluate.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🤖 Models Compared

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 99.17% |
| Naive Bayes | 95.63% |
| Linear SVM | 99.68% |
| **Random Forest** | **99.73%** |

### Final Model

- Algorithm: Random Forest
- Accuracy: **99.73%**
- Precision: **99.67%**
- Recall: **99.83%**
- F1-Score: **99.75%**
- ROC-AUC: **99.97%**

---


## 🔮 Future Improvements

- Transformer-based models (BERT/DistilBERT)
- Explainable AI using SHAP/LIME
- Docker support
- CI/CD with GitHub Actions
- Cloud deployment
- Better generalization with larger datasets

---

