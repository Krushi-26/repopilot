"""Okapi BM25 with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

The classic Okapi IDF goes to zero or negative for terms in half the corpus or more,
which silently drops matches in small repos. Lucene's form is always positive.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        terms = [t for t in set(query) if t in self.idf]
        scores = []
        for tf, dl in zip(self.tfs, self.lens):
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append(s)
        return scores
