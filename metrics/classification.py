import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def classification_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    # AUROC macro (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "macro_f1": macro_f1, "auroc": auc}
