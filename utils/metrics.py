import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report




def compute_metrics(y_true, y_pred, average='binary'):
acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
return {
'accuracy': float(acc),
'precision': float(precision),
'recall': float(recall),
'f1_score': float(f1)
}




def detailed_report(y_true, y_pred, target_names=None):
return classification_report(y_true, y_pred, target_names=target_names, digits=4)