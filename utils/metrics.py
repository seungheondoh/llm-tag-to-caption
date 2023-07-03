"""Placeholder for metrics."""
from functools import partial
import evaluate
import numpy as np
import torch
import torchmetrics.retrieval as retrieval_metrics
# CAPTIONING METRICS
def bleu(predictions, ground_truths, order):
    bleu_eval = evaluate.load("bleu")
    return bleu_eval.compute(
        predictions=predictions, references=ground_truths, max_order=order
    )["bleu"]


def meteor(predictions, ground_truths):
    meteor_eval = evaluate.load("meteor")
    return meteor_eval.compute(predictions=predictions, references=ground_truths)[
        "meteor"
    ]


def rouge(predictions, ground_truths):
    rouge_eval = evaluate.load("rouge")
    return rouge_eval.compute(predictions=predictions, references=ground_truths)[
        "rougeL"
    ]


def bertscore(predictions, ground_truths):
    bertscore_eval = evaluate.load("bertscore")
    score = bertscore_eval.compute(
        predictions=predictions, references=ground_truths, lang="en"
    )["f1"]
    return np.mean(score)


def metric_1(predictions, ground_truths) -> float:
    """Computes metric_1 score.
    Args:
        predictions: A list of predictions.
        ground_truths: A list of ground truths.
    Returns:
        metric_1: A float number, the metric_1 score.
    """
    return 0.0


# RETRIEVAL METRICS
def _prepare_torchmetrics_input(scores, query2target_idx):
    target = [
        [i in target_idxs for i in range(len(scores[0]))]
        for query_idx, target_idxs in query2target_idx.items()
    ]
    indexes = torch.arange(len(scores)).unsqueeze(1).repeat((1, len(target[0])))
    return torch.as_tensor(scores), torch.as_tensor(target), indexes


def _call_torchmetrics(
    metric: retrieval_metrics.RetrievalMetric, scores, query2target_idx, **kwargs
):
    preds, target, indexes = _prepare_torchmetrics_input(scores, query2target_idx)
    return metric(preds, target, indexes=indexes, **kwargs).item()


def recall(predicted_scores, query2target_idx, k: int) -> float:
    """Compute retrieval recall score at cutoff k.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
        k: number of top-k results considered
    Returns:
        average score of recall@k
    """
    recall_metric = retrieval_metrics.RetrievalRecall(k=k)
    return _call_torchmetrics(recall_metric, predicted_scores, query2target_idx)


def mean_average_precision(predicted_scores, query2target_idx) -> float:
    """Compute retrieval mean average precision (MAP) score at cutoff k.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
    Returns:
        MAP@k score
    """
    map_metric = retrieval_metrics.RetrievalMAP()
    return _call_torchmetrics(map_metric, predicted_scores, query2target_idx)


def mean_reciprocal_rank(predicted_scores, query2target_idx) -> float:
    """Compute retrieval mean reciprocal rank (MRR) score.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
    Returns:
        MRR score
    """
    mrr_metric = retrieval_metrics.RetrievalMRR()
    return _call_torchmetrics(mrr_metric, predicted_scores, query2target_idx)