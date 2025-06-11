###  Model Comparison
Trained and evaluated 4 classification algorithms using `caret::train()` and 3–5 fold cross-validation:

| Model         | Best Accuracy |
|---------------|----------------|
| Random Forest | **82.01%**  |
| SVM (Linear)  | 81.94%         |
| k-NN          | ~81.3%         |
| Naive Bayes   | ~81.2%         |

---

## Final Outcome
- Random Forest performed best overall with 82.01% accuracy (3-fold CV)
- Top predictors included: `PAY_0`, `LIMIT_BAL`, `AGE`, `BILL_AMT1`, `EDUCATION`

---
