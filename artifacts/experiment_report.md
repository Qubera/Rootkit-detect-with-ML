# Syscall ML Experiment Report

## Dataset Summary
- Rows: 5452
- Unique PIDs: 13
- Class counts (rows): {0: 1932, 1: 3520}
- Class counts (PIDs): {0: 7, 1: 6}

## Training Configuration
- Candidate window sizes: [3, 5, 10]
- Candidate top bigrams: [20, 50, 100]
- CV splits: 3
- Holdout ratio: 0.2

## Selected Configuration
- Model: XGBoost
- Window size: 3
- Top bigrams: 20
- Feature count: 33
- CV F1: 0.2051
- CV AUC: 0.5000

## Holdout Metrics
- Accuracy: 0.9674
- Balanced accuracy: 0.9720
- Precision: 0.9994
- Recall: 0.9672
- F1-score: 0.9830
- ROC-AUC: 0.9898

## Classification Report
```text
              precision    recall  f1-score   support

           0       0.43      0.98      0.60        43
           1       1.00      0.97      0.98      1676

    accuracy                           0.97      1719
   macro avg       0.72      0.97      0.79      1719
weighted avg       0.99      0.97      0.97      1719
```
