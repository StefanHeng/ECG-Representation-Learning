import torch
import sklearn.metrics as metrics

from ecg_transformer.util import *


def get_accuracy(
        preds: torch.Tensor, labels: torch.Tensor,
        return_auc: bool = True
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    :param preds: Probability distribution for each class
    :param labels: Ground truth labels
    :param return_auc: If true, macro AUROC and (valid) per-class AUROC is returned
    """
    if not hasattr(get_accuracy, 'id2code'):
        get_accuracy.id2code = config('datasets.PTB_XL.code.id2code')
    preds_bin = (preds >= 0.5).int()  # for binary classifications per class

    macro_auc, code2auroc = None, dict()
    if return_auc:
        # which of the 71 classes in this batch/entire dataset of labels, has both positive and negative samples
        # filter out those with only pos/neg labels, otherwise AUROC breaks
        msk_2_class = torch.any(labels != labels[0], dim=0)
        if msk_2_class.sum() > 0:
            idxs_2_class = msk_2_class.nonzero().flatten().tolist()
            labels, preds_prob = labels.float().cpu().numpy(), preds.float().cpu().numpy()
            # macro-average auroc, as in *Self-supervised representation learning from 12-lead ECG data*
            aucs: List[float] = metrics.roc_auc_score(labels[:, msk_2_class], preds_prob[:, msk_2_class], average=None)
            code2auroc = {get_accuracy.id2code[idx]: auc for idx, auc in zip(idxs_2_class, aucs)}
            macro_auc = np.array(list(code2auroc.values())).mean()
        # should rarely be not the case, unless, the positive labels for all samples is exactly the same

    preds_bin, labels = preds_bin.flatten(), labels.flatten()  # aggregate all classes
    report = metrics.classification_report(  # suppresses the warning
        preds_bin, labels, labels=[0, 1], target_names=['neg', 'pos'], output_dict=True, zero_division=0
    )
    rec_pos, rec_neg = (report[k]['recall'] for k in ('neg', 'pos'))
    return dict(
        binary_accuracy=metrics.accuracy_score(labels, preds_bin),
        weighted_binary_accuracy=metrics.balanced_accuracy_score(labels, preds_bin),
        binary_negative_recall=rec_neg,
        binary_positive_recall=rec_pos,
        macro_auc=macro_auc, per_class_auc=code2auroc
    )


def _pretty_single(key: str, val, ref: Dict = None):
    if key in ['step', 'epoch']:
        k = next(iter(k for k in ref.keys() if key in k))
        lim = ref[k]
        return f'{val:>{len(str(lim))}}/{lim}'  # Pad integer
    elif 'loss' in key:
        return f'{round(val, 4):7.4f}'
    elif any(k in key for k in ('acc', 'recall', 'auc')):
        def _single(v):
            return f'{round(v * 100, 2):6.2f}' if v is not None else '-'

        if isinstance(val, list):
            return [_single(v) for v in val]
        elif isinstance(val, dict):
            return {k: _single(v) for k, v in val.items()}
        else:
            return _single(val)
    elif 'learning_rate' in key or 'lr' in key:
        return f'{round(val, 7):.3e}'
    else:
        return val


def pretty_log_dict(d_log: Dict, ref: Dict = None):
    return {k: _pretty_single(k, v, ref=ref) for k, v in d_log.items()}
