# SMS Spam Classifier — Analysis & Improvement Guide

## Your Current Pipeline Summary

| Step | What You Did |
|---|---|
| **Data** | 5572 rows → dropped duplicates → **5169 rows** (87.37% ham, 12.63% spam — **imbalanced**) |
| **Features** | `characters_length`, `words_length`, `sentence_length` (engineered but **never used in model**) |
| **Text Preprocessing** | lowercase → tokenize → keep only alpha → remove stopwords → **PorterStemmer** |
| **Vectorizer** | Tried CountVectorizer (5849 features) → then **TF-IDF** (5849 → then limited to 3000) |
| **Models** | 11 classifiers + VotingClassifier + StackingClassifier |
| **Split** | 80/20, `random_state=42` |

---

## Your Current Results (TF-IDF, max_features=3000)

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| **ExtraTreesClassifier** | **0.9758** | **0.9762** | **0.8483** |
| XGBClassifier | 0.9710 | 0.9600 | 0.8276 |
| RF | 0.9681 | 0.9912 | 0.7793 |
| BaggingClassifier | 0.9603 | 0.8714 | 0.8414 |
| GradientBoosting | 0.9594 | 0.9558 | 0.7448 |
| LogisticRegression | 0.9555 | 0.9459 | 0.7241 |
| MultinomialNB | 0.9555 | 1.0000 | 0.6828 |
| KNN | 0.9197 | 1.0000 | 0.4276 |

> [!NOTE]
> **ExtraTreesClassifier** is indeed your best overall model — highest accuracy (97.58%) with excellent precision (97.62%) and the best recall (84.83%) among the top-precision models. Good call!

---

## Issues I Noticed in Your Notebook

> [!WARNING]
> ### 1. Engineered features are NEVER used
> You created `characters_length`, `words_length`, `sentence_length` but your model only trains on the TF-IDF matrix of `transformed_text`. These features are wasted — combining them with TF-IDF can boost performance.

> [!WARNING]
> ### 2. SVM with sigmoid kernel gets 0% precision
> `SVC(kernel='sigmoid')` is predicting everything as "ham" (class 0). Sigmoid kernel is almost never the right choice for text. Use `kernel='rbf'` or `kernel='linear'` instead.

> [!NOTE]
> ### 3. Class imbalance (87% vs 13%)
> With only 12.63% spam, precision can appear artificially high. Some models (KNN, MultinomialNB) get 100% precision but terrible recall (42-68%), meaning they miss most spam.

---

## 7 Ways to Improve Performance

### Strategy 1: Hyperparameter Tuning with GridSearchCV / RandomizedSearchCV

This is the **highest-impact change** you can make right now. Here's ready-to-use code:

#### ExtraTreesClassifier — RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced', {0: 1, 1: 5}, {0: 1, 1: 7}],
    'criterion': ['gini', 'entropy'],
}

etc = ExtraTreesClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    etc,
    param_distributions=param_dist,
    n_iter=50,  # try 50 random combinations
    cv=5,
    scoring='precision',  # optimize for precision
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Precision (CV):", random_search.best_score_)

# Evaluate on test set
y_pred = random_search.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred))
print("Test Recall:", recall_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

#### XGBClassifier — GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 5, 7],  # handles class imbalance
}

xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

grid_search = GridSearchCV(
    xgb,
    param_grid=param_grid,
    cv=5,
    scoring='precision',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.best_estimator_.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred))
print("Test Recall:", recall_score(y_test, y_pred))
```

> [!TIP]
> **GridSearchCV** tries every combination — use when the param grid is small. **RandomizedSearchCV** samples random combos — much faster for large search spaces. For ExtraTreesClassifier, use RandomizedSearchCV (the grid above has 1440 combos!).

---

### Strategy 2: Combine TF-IDF with Your Engineered Features

You already created useful features but never used them! Stack them with TF-IDF:

```python
from scipy.sparse import hstack, csr_matrix

# Re-create TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(df['transformed_text'])

# Get your engineered features
extra_features = df[['characters_length', 'words_length', 'sentence_length']].values
extra_features_sparse = csr_matrix(extra_features)

# Combine them
X_combined = hstack([X_tfidf, extra_features_sparse])

# Now split and train
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)
```

This gives the model both word-level signals AND structural signals (spam tends to be longer, more sentences, etc.).

---

### Strategy 3: Try TF-IDF with N-grams

Your current TF-IDF only uses unigrams (single words). Adding bigrams captures phrases like "free call", "claim prize":

```python
tfidf_ngram = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = tfidf_ngram.fit_transform(df['transformed_text']).toarray()
```

You can also tune this inside GridSearchCV using a Pipeline (see Strategy 5).

---

### Strategy 4: Handle Class Imbalance with `class_weight='balanced'`

For tree-based models, pass `class_weight='balanced'` to give the minority class (spam) more weight:

```python
etc = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # <-- adds this
)
```

This alone can improve recall without hurting precision much. It's already included as a tunable parameter in the RandomizedSearchCV code above.

---

### Strategy 5: Use a Pipeline for End-to-End Tuning

Tune the **vectorizer + classifier together** in one GridSearchCV:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ExtraTreesClassifier(random_state=42, n_jobs=-1))
])

param_dist = {
    'tfidf__max_features': [2000, 3000, 5000, None],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__sublinear_tf': [True, False],
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 20, 30],
    'clf__class_weight': [None, 'balanced'],
    'clf__criterion': ['gini', 'entropy'],
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=40,
    cv=5,
    scoring='precision',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(df['transformed_text'], y)  # use full data, CV handles splitting
print("Best params:", search.best_params_)
print("Best precision:", search.best_score_)
```

> [!IMPORTANT]
> This is the most powerful approach because it tunes the vectorizer AND classifier jointly — the best `max_features` might differ depending on the classifier's settings.

---

### Strategy 6: Threshold Tuning (Free Precision Boost)

Instead of using the default 0.5 threshold for classification, you can raise it to get higher precision:

```python
# Get probability predictions
y_proba = etc.predict_proba(X_test)[:, 1]  # probability of being spam

# Try different thresholds
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.title('Precision-Recall vs Threshold')
plt.show()

# Example: use threshold of 0.6 instead of 0.5
threshold = 0.6
y_pred_custom = (y_proba >= threshold).astype(int)
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))
print("Accuracy:", accuracy_score(y_test, y_pred_custom))
```

---

### Strategy 7: Improved Voting/Stacking with Tuned Models

After tuning individual models, combine them:

```python
from sklearn.ensemble import VotingClassifier

# Use your BEST tuned models
voting = VotingClassifier(
    estimators=[
        ('etc', best_etc),   # tuned ExtraTreesClassifier
        ('xgb', best_xgb),   # tuned XGBClassifier
        ('rf', best_rf),      # tuned RandomForest
    ],
    voting='soft',            # uses probabilities — better than 'hard'
    weights=[3, 2, 1]         # weight the best model more
)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
```

---

## Recommended Action Plan (Priority Order)

| Priority | Strategy | Expected Impact | Effort |
|---|---|---|---|
| 🥇 **1st** | Hyperparameter tune ExtraTreesClassifier (Strategy 1) | High | Medium |
| 🥈 **2nd** | Add engineered features to TF-IDF (Strategy 2) | Medium-High | Low |
| 🥉 **3rd** | Try N-grams in TF-IDF (Strategy 3) | Medium | Low |
| 4th | Pipeline-based tuning (Strategy 5) | High | Medium |
| 5th | Threshold tuning (Strategy 6) | Medium | Low |
| 6th | Class weight balancing (Strategy 4) | Medium | Very Low |
| 7th | Ensemble tuned models (Strategy 7) | Medium | Medium |

> [!TIP]
> Start with **Strategy 1** (RandomizedSearchCV on ExtraTreesClassifier) — it's the single biggest lever you have. Then try **Strategy 2** (combining your unused features). These two together can reasonably push precision to ~98-99% and accuracy above 97.5%.
