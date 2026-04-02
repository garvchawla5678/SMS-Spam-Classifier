# 🛡️ SMS Spam Classifier

A comprehensive machine learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and ensemble tree-based models.

After **8 iterations** of progressive optimization — spanning baseline models, feature engineering, hyperparameter tuning (Optuna), ensemble methods, and cross-validation — the final model achieves **98.16% accuracy** with **100% precision** on the test set.

---

## 📊 Dataset

- **Source:** [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total messages:** 5,572 → **5,169** after removing duplicates
- **Class distribution:** 87% Ham / 13% Spam (imbalanced)

---

## 🔧 Text Preprocessing Pipeline

Every SMS message goes through the following transformations before being fed to the model:

```
Raw Text → Lowercase → Tokenize (NLTK) → Keep Alphabetic Only → Remove Stopwords → Porter Stemming → Cleaned Text
```

**Example:**
```
Input:  "WINNER!! You have been selected to receive a £900 prize reward!"
Output: "winner select receiv prize reward"
```

---

## 🧪 Iterative Optimization Journey

### Iteration 1 — Baseline (11 Classifiers)
Trained 11 classifiers on raw TF-IDF features with default parameters.

| Model | Accuracy | Precision |
|-------|----------|-----------|
| ExtraTreesClassifier | 97.58% | 100% |
| RandomForest | 96.90% | 100% |
| XGBClassifier | 97.20% | 94.6% |
| MultinomialNB | 93.52% | 100% |
| SVM (sigmoid kernel) | 85.98% | 0% ❌ |

> **Finding:** ExtraTrees dominated. SVM with sigmoid kernel completely failed (predicted everything as ham).

---

### Iteration 2 — N-grams + Class Balancing
- Added **bigrams** to TF-IDF (`ngram_range=(1,2)`, `max_features=3000`)
- Set `class_weight='balanced'` for tree-based models to handle the 87/13 imbalance
- Started tracking **Recall** alongside Precision

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| ExtraTreesClassifier | 97.78% | 98.4% | **85.5%** |
| RandomForest | 97.49% | 100% | 82.1% |
| XGBClassifier | 96.52% | 91.0% | 83.4% |

> **Finding:** Class balancing significantly improved recall (from ~0% to 85%+) while maintaining high precision.

---

### Iteration 3 — Feature Engineering
Added hand-crafted features alongside TF-IDF:
- `characters_length` — character count of the message
- `words_length` — word count
- `sentence_length` — sentence count

Used `MaxAbsScaler` to normalize engineered features to the same scale as TF-IDF values.

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| ExtraTreesClassifier | 97.87% | 97.7% | **86.9%** |
| MultinomialNB | 97.87% | 100% | 84.8% |
| LogisticRegression | 97.39% | 100% | 81.4% |

> **Finding:** Marginal improvement. Recall pushed to 86.9% for ExtraTrees.

---

### Iteration 4 — Hyperparameter Tuning (Optuna)
Used **Optuna** (20 trials per model) with 3-fold cross-validation to tune:

**ExtraTreesClassifier** — Best config found:
```python
ExtraTreesClassifier(
    n_estimators=500, min_samples_split=10, min_samples_leaf=1,
    max_features='log2', criterion='gini', max_depth=None,
    class_weight={0: 1, 1: 5}
)
# Optuna best CV score: 0.9824
```

**XGBClassifier** — Best config found:
```python
XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.2,
    subsample=1, colsample_bytree=0.9, scale_pos_weight=7
)
# Optuna best CV score: 0.9681
```

**RandomForestClassifier** — Best config found:
```python
RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=8,
    min_samples_leaf=2, max_features='log2', criterion='entropy',
    class_weight={0: 1, 1: 5}
)
# Optuna best CV score: 0.9783
```

> **Finding:** ExtraTrees consistently outperformed RF and XGBoost across all tuning rounds.

---

### Iteration 5 — Feature Selection (SelectKBest)
Applied `SelectKBest(chi2)` to reduce the 3000-dimensional TF-IDF space. Tested `k=1000` and `k=2000`.

| k | Accuracy | Precision | Recall |
|---|----------|-----------|--------|
| 3000 (all) | 98.07% | 97.7% | 88.3% |
| 2000 | 97.39% | 92.1% | 89.0% |
| 1000 | 98.07% | 97.7% | 88.3% |

> **Finding:** `k=1000` matched full feature performance while being 3× faster to train.

---

### Iteration 6 — Threshold Tuning
Analyzed **Precision-Recall curves** to find the optimal decision threshold instead of using the default 0.5.

| Threshold | Precision | Recall | Accuracy |
|-----------|-----------|--------|----------|
| 0.50 | 98.5% | 88.3% | 98.07% |
| **0.58** | **100%** | **86.9%** | **98.16%** |
| 0.61 | 100% | 84.8% | 97.87% |
| 0.79 | 100% | 64.1% | 94.97% |

> **Finding:** Threshold = 0.58 achieved **100% precision** (zero false positives) while keeping recall at 86.9%.

---

### Iteration 7 — Ensemble Methods

**Voting Classifier** (soft voting, weights: ETC=3, XGB=1, RF=2):

| Metric | Score |
|--------|-------|
| Accuracy | 97.29% |
| Precision | 88.7% |
| Recall | 92.4% |

**Stacking Classifier** (ETC as final estimator):

| Metric | Score |
|--------|-------|
| Accuracy | 98.26% |
| Precision | 99.2% |
| Recall | 88.3% |

Also tried **TruncatedSVD** (500 components) for dimensionality reduction — abandoned due to terrible recall (46%).

> **Finding:** Ensembles did not meaningfully beat the standalone ExtraTrees. Stacking was competitive but more complex with no clear gain.

---

### Iteration 8 — Cross-Validation (Final Verification)
Ran **5-fold cross-validation** on the full dataset to verify the model was not overfit.

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
|-------|-------------|-------------|-----------|-------|
| **ExtraTrees (standalone)** | **98.10 ± 0.37%** | **98.45 ± 1.57%** | **86.38 ± 2.73%** | **91.99 ± 1.63%** |
| Voting (ETC+XGB+RF) | 98.01 ± 0.50% | 98.40 ± 1.03% | 85.61 ± 3.24% | 91.54 ± 2.21% |
| Stacking (ETC meta) | 97.95 ± 0.25% | 92.24 ± 2.54% | 91.58 ± 1.59% | 91.87 ± 0.89% |
| ETC + 3 eng. features | 98.19 ± 0.31% | 98.44 ± 1.13% | 86.60 ± 2.35% | 92.13 ± 1.44% |

> **Finding:** CV scores closely match test set scores → **no overfitting**. Engineered features add negligible value. Standalone ExtraTrees is optimal.

---

## 🏆 Final Model

```python
# Vectorizer
TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

# Classifier
ExtraTreesClassifier(
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='log2',
    criterion='gini',
    max_depth=None,
    class_weight={0: 1, 1: 5},
    n_jobs=-1,
    random_state=42
)

# Decision Threshold: 0.58
```

### Final Performance

| Metric | Test Set | 5-Fold CV |
|--------|----------|-----------|
| **Accuracy** | **98.16%** | 98.10 ± 0.37% |
| **Precision** | **100%** | 98.45 ± 1.57% |
| **Recall** | **86.9%** | 86.38 ± 2.73% |
| **F1 (spam)** | **~0.93** | 91.99 ± 1.63% |

> 💡 **100% precision** means zero legitimate messages are ever falsely flagged as spam — critical for a spam filter where false positives are more costly than missed spam.

---

## 📁 Project Structure

```
SMS-Spam-Classifier/
├── sms-spam-detection.ipynb   # Full EDA + modeling notebook (208 cells)
├── spam.csv                   # Dataset
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| NLP | NLTK (tokenization, stopwords, stemming) |
| Vectorization | scikit-learn TfidfVectorizer |
| Models | ExtraTreesClassifier, RandomForest, XGBoost, + 8 others |
| Tuning | Optuna (Bayesian optimization) |
| Feature Selection | SelectKBest (chi-squared) |
| Ensembles | VotingClassifier, StackingClassifier |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| GPU | XGBoost with CUDA acceleration |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/garvchawla5678/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier

# Install dependencies
pip install pandas numpy scikit-learn xgboost nltk optuna matplotlib seaborn plotly wordcloud

# Open the notebook
jupyter notebook sms-spam-detection.ipynb
```

---

## 📈 Key Takeaways

1. **Class imbalance handling is critical** — Without `class_weight`, tree models ignored the minority spam class entirely.
2. **Threshold tuning > model swapping** — Moving from 0.50 → 0.58 threshold gave 100% precision, a bigger gain than switching models.
3. **Ensembles aren't always better** — For this dataset, the standalone ExtraTrees outperformed both Voting and Stacking ensembles.
4. **Feature engineering had diminishing returns** — Character/word/sentence length features added ~0.1% improvement, not worth the complexity.
5. **SVD destroys sparse signal** — TruncatedSVD crushed recall from 87% → 46%, proving that sparse TF-IDF features carry important discriminative information.
6. **Cross-validation confirms robustness** — CV scores matched test scores within 0.1%, confirming no overfitting.
