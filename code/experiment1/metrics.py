# https://github.com/huggingface/transformers/blob/9e9a1fb8c75e2ef00fea9c4c0dc511fc0178081c/src/transformers/data/metrics/__init__.py

try:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        matthews_corrcoef,
    )
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
    )

    from scipy.special import expit
    from scipy.stats import pearsonr, spearmanr

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


# ---------------------------------------------------------------------------


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


# ---------------------------------------------------------------------------


def sameness_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    # return {"mcc": matthews_corrcoef(labels, preds)}
    # return acc_and_f1(preds, labels)

    if task_name == "sent-5":
        return {
            "acc": simple_accuracy(preds, labels),
            "f1-micro": f1_score(y_true=labels, y_pred=preds, average="micro"),
            "f1-macro": f1_score(y_true=labels, y_pred=preds, average="macro"),
            "f1-weighted": f1_score(y_true=labels, y_pred=preds, average="weighted"),
            **pearson_and_spearman(preds, labels),
        }
    elif task_name in ("sent-b", "same-b"):
        return {
            **acc_and_f1(preds, labels),
            **pearson_and_spearman(preds, labels),
            "class_report": classification_report(
                y_true=labels,
                y_pred=preds,
                output_dict=True,
                labels=[0, 1],
                target_names=["not same", "same"],
            ),
        }
    elif task_name in ("sent-r", "same-r"):
        # TODO: how to better do this ...
        # preds2 = expit(preds).round().astype("int32")
        preds2 = preds.round().astype("int32")
        # labels = labels.astype("float")
        # preds = preds.astype("float")
        # float can not use average="binary" in f1_score
        return {
            **acc_and_f1(preds2, labels),
            **pearson_and_spearman(preds, labels),
        }
    else:
        raise KeyError(task_name)


# ---------------------------------------------------------------------------
