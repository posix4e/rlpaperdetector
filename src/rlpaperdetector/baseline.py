from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def row_to_document(row: dict[str, str]) -> str:
    pieces = [
        clean_text(row.get("title")),
        clean_text(row.get("abstract")),
        f"journal:{clean_text(row.get('journal')).lower()}",
        f"year:{clean_text(row.get('publication_year'))}",
    ]
    return "\n".join(piece for piece in pieces if piece)


def row_to_tokens(row: dict[str, str]) -> list[str]:
    return tokenize(row_to_document(row))


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[clean_text(row.get("split")) or "unspecified"].append(row)
    return buckets


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def roc_auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positives_in_group = sum(label for _, label in ranked[index:end])
        rank_sum += positives_in_group * average_rank
        index = end

    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision_score(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / index
    return precision_sum / positives


def classification_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float | None]:
    if not labels:
        return {
            "count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "brier": None,
            "auroc": None,
            "average_precision": None,
        }

    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 1)
    fp = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 1)
    tn = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 0)
    fn = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 0)

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)

    return {
        "count": len(labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "auroc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
    }


class NaiveBayesRetractionModel:
    def __init__(
        self,
        class_doc_counts: dict[int, int],
        class_token_totals: dict[int, int],
        class_token_counts: dict[int, dict[str, int]],
        vocabulary_size: int,
        alpha: float = 1.0,
    ) -> None:
        self.class_doc_counts = {int(key): int(value) for key, value in class_doc_counts.items()}
        self.class_token_totals = {int(key): int(value) for key, value in class_token_totals.items()}
        self.class_token_counts = {
            int(label): {token: int(count) for token, count in token_counts.items()}
            for label, token_counts in class_token_counts.items()
        }
        self.vocabulary_size = int(vocabulary_size)
        self.alpha = float(alpha)

    @classmethod
    def train(cls, rows: list[dict[str, str]], alpha: float = 1.0) -> "NaiveBayesRetractionModel":
        class_doc_counts = Counter()
        class_token_totals = Counter()
        class_token_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
        vocabulary: set[str] = set()

        for row in rows:
            label = int(row["label"])
            tokens = row_to_tokens(row)
            class_doc_counts[label] += 1
            class_token_totals[label] += len(tokens)
            class_token_counts[label].update(tokens)
            vocabulary.update(tokens)

        return cls(
            class_doc_counts=dict(class_doc_counts),
            class_token_totals=dict(class_token_totals),
            class_token_counts={label: dict(counter) for label, counter in class_token_counts.items()},
            vocabulary_size=len(vocabulary),
            alpha=alpha,
        )

    def predict_score(self, row: dict[str, str]) -> float:
        tokens = row_to_tokens(row)
        total_docs = sum(self.class_doc_counts.values())
        if total_docs == 0:
            raise ValueError("Model has no training data")

        log_probs: dict[int, float] = {}
        for label in (0, 1):
            doc_count = self.class_doc_counts.get(label, 0)
            prior = (doc_count + self.alpha) / (total_docs + self.alpha * 2)
            log_prob = math.log(prior)
            token_total = self.class_token_totals.get(label, 0)
            denominator = token_total + self.alpha * max(self.vocabulary_size, 1)
            for token in tokens:
                token_count = self.class_token_counts.get(label, {}).get(token, 0)
                log_prob += math.log((token_count + self.alpha) / denominator)
            log_probs[label] = log_prob

        return sigmoid(log_probs[1] - log_probs[0])

    def predict(self, row: dict[str, str], threshold: float = 0.5) -> dict[str, float | int]:
        score = self.predict_score(row)
        return {
            "score": score,
            "label": 1 if score >= threshold else 0,
        }

    def evaluate(self, rows: list[dict[str, str]], threshold: float = 0.5) -> dict[str, float | None]:
        labels = [int(row["label"]) for row in rows]
        scores = [self.predict_score(row) for row in rows]
        metrics = classification_metrics(labels, scores, threshold=threshold)
        metrics["positive_rate"] = sum(labels) / len(labels) if labels else None
        return metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "multinomial_naive_bayes",
            "alpha": self.alpha,
            "class_doc_counts": self.class_doc_counts,
            "class_token_totals": self.class_token_totals,
            "class_token_counts": self.class_token_counts,
            "vocabulary_size": self.vocabulary_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NaiveBayesRetractionModel":
        return cls(
            class_doc_counts=payload["class_doc_counts"],
            class_token_totals=payload["class_token_totals"],
            class_token_counts=payload["class_token_counts"],
            vocabulary_size=int(payload["vocabulary_size"]),
            alpha=float(payload.get("alpha", 1.0)),
        )


def save_model(path: Path, model: NaiveBayesRetractionModel, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "model": model.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model(path: Path) -> tuple[NaiveBayesRetractionModel, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NaiveBayesRetractionModel.from_dict(payload["model"]), payload["metadata"]
