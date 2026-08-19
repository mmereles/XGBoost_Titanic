# XGBoost on Titanic 🚢

Binary classification of survival on the Titanic dataset using XGBoost, focused on **model diagnostics beyond accuracy** — class imbalance, early stopping, feature importance, and explainability with SHAP.

## Results

| Model | Test Accuracy |
|---|---|
| Decision Tree | 75.98% |
| Random Forest | 80.45% |
| XGBoost (no early stopping) | 80.45% |
| Logistic Regression | 81.01% |
| **XGBoost (early stopping, 27 trees)** | **79.89%** |

> A simple linear model (Logistic Regression) matched or outperformed the tree-based models on this dataset — evidence that a more complex algorithm doesn't always win, especially on small datasets with relatively linear relationships.

## Key findings

**1. Accuracy alone hid a real problem.**
With 80% overall accuracy, the model was good at detecting passengers who died (recall 0.92) but missed 39% of actual survivors (recall 0.61) — a direct consequence of class imbalance (61.6% died / 38.4% survived).

**2. Early stopping outperformed GridSearchCV.**
The model that decided on its own when to stop (monitoring test logloss in real time, 27 trees) generalized better than the best result from an exhaustive search over 180 hyperparameter combinations. `GridSearchCV` optimizes against cross-validation on train, not against the real test set — it isn't a drop-in replacement for early stopping.

**3. `gain` is more reliable than `weight` for feature importance.**
`age` and `fare` (continuous variables) are used in far more splits, but each individual split contributes little. `sex` is used far less often, but each split is much more decisive — confirmed by gain (28.75 vs 2.42), and consistent with historical reality ("women and children first").

**4. SHAP explains individual predictions, not just global trends.**
Beyond which feature matters "in general," SHAP answers why the model predicted something specific for a given passenger — key when a model is used for real, auditable decisions.

## Notebook contents

- Loading and exploring the dataset (`seaborn.load_dataset('titanic')`)
- Handling missing values: when to impute, when to drop, when to let XGBoost handle NaN natively
- Encoding categorical variables
- Training with `XGBClassifier` + early stopping
- Evaluation: accuracy, classification report, confusion matrix
- Feature importance (weight vs gain)
- Explainability with SHAP (summary plot + individual waterfall plot)
- Tuning with `GridSearchCV`
- Comparison against Decision Tree, Logistic Regression, and Random Forest

Run in Google Colab o same platforms

## Stack

`xgboost` · `scikit-learn` · `pandas` · `seaborn` · `matplotlib` · `shap`
