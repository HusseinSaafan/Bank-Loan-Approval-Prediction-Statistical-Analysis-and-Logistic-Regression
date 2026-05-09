# Best Model Evaluation Results

## Model Selection

All three candidate models (Logistic Regression, Random Forest, XGBoost) were trained and evaluated using cross-validation F1 scores. All three models achieved very close CV F1 score of **0.8792**, and **XGBoost** was selected as the best model (as the last evaluated with that score).

| Model               | CV F1 Score |
|---------------------|-------------|
| Logistic Regression | 0.8792      |
| Random Forest       | 0.8792      |
| **XGBoost**         | **0.8792**  |

---

## Test Set Performance

The selected XGBoost model was evaluated on the held-out test set (112 samples).

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 80.36% |
| F1 Score     | 0.8706 |
| ROC AUC      | 0.7317 |

---

## Confusion Matrix

|                     | Predicted: Rejected (0) | Predicted: Approved (1) |
|---------------------|-------------------------|-------------------------|
| **Actual: Rejected (0)** | 16 (True Negatives)  | 20 (False Positives)    |
| **Actual: Approved (1)** | 2  (False Negatives) | 74 (True Positives)     |

- **True Positives (74):** Correctly predicted loan approvals
- **True Negatives (16):** Correctly predicted loan rejections
- **False Positives (20):** Loans predicted as approved but actually rejected — the model is over-approving rejections
- **False Negatives (2):** Loans predicted as rejected but actually approved — very few missed approvals

---

## Per-Class Metrics

| Class             | Precision | Recall | F1 Score | Support |
|-------------------|-----------|--------|----------|---------|
| Rejected (0)      | 88.9%     | 44.4%  | 0.593    | 36      |
| **Approved (1)**  | **78.7%** | **97.4%** | **0.871** | 76   |
| Weighted Average  | 82.0%     | 80.4%  | 0.781    | 112     |

---

## Interpretation

### Strengths

- **High recall on loan approvals (97.4%):** The model correctly identifies nearly all applicants who should be approved. Only 2 out of 76 approved applicants were missed. This is the most commercially important outcome — the bank does not want to turn away creditworthy applicants.

- **Strong F1 score on approved class (0.871):** The model performs well on the majority class overall, balancing both precision and recall.

- **High precision on rejections (88.9%):** When the model does predict a rejection, it is correct 88.9% of the time.

### Weaknesses

- **Poor recall on loan rejections (44.4%):** The model only correctly identifies 16 out of 36 applicants who should be rejected. This means **20 risky applicants are being approved**, which introduces credit risk for the bank.

- **Class imbalance effect:** The dataset is skewed (76 approved vs 36 rejected in test set). The model has learned to favor the "Approved" prediction, which inflates approval recall but hurts rejection recall.

- **ROC AUC of 0.7317:** This is moderate. A perfect classifier scores 1.0 and a random classifier scores 0.5. The model has reasonable discriminative ability but there is clear room for improvement — particularly in separating the two classes at the decision boundary.

### Business Implications

The current model **prioritizes minimizing missed approvals** (False Negatives = 2) at the cost of **incorrectly approving risky applicants** (False Positives = 20). Whether this tradeoff is acceptable depends on the bank's risk appetite:

- If the bank prioritizes **customer acquisition**, this bias toward approvals is acceptable.
- If the bank prioritizes **credit risk management**, the decision threshold should be raised (above 0.5) to reduce false positives, which will improve rejection recall at the expense of some approval recall.

### Recommended Next Steps

1. **Threshold tuning:** Adjust the classification threshold from 0.5 to ~0.6–0.7 to reduce false positives and improve rejection identification.
2. **Class balancing:** Apply SMOTE or class weighting during training to make the model less biased toward approvals.
3. **Feature engineering:** Explore additional features that may better discriminate between creditworthy and risky applicants.
