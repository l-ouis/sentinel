"""Rubric schema, v0 — placeholder shape until the full spec lands.

A rubric is a TOML file (default: verifier/rubric.toml) of scored items plus
multiplicative penalties:

    score = sum(item.weight * item_score) * prod(p.factor for triggered p)

Invariants:
- item weights sum to exactly 1.0, so an unpenalized perfect run scores 1.0
- item scores are judge-emitted in [0, 1]
- a penalty, when triggered, multiplies the total by its factor in [0, 1)

The judge produces the per-item scores and the triggered-penalty set; the
harness computes the final score with `compute_score` — arithmetic is never
delegated to the judge model.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RubricItem(_Model):
    id: str
    weight: float = Field(gt=0.0, le=1.0)
    desc: str = ""


class RubricPenalty(_Model):
    id: str
    factor: float = Field(ge=0.0, lt=1.0)  # applied multiplicatively when triggered
    desc: str = ""


class Rubric(_Model):
    items: list[RubricItem] = Field(min_length=1)
    penalties: list[RubricPenalty] = Field(default_factory=list)

    @model_validator(mode="after")
    def _invariants(self) -> "Rubric":
        ids = [x.id for x in self.items] + [p.id for p in self.penalties]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate rubric ids: {ids}")
        total = sum(i.weight for i in self.items)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"item weights sum to {total}, must sum to 1.0")
        return self

    def compute_score(
        self, item_scores: dict[str, float], triggered: Iterable[str]
    ) -> float:
        item_ids = {i.id for i in self.items}
        if set(item_scores) != item_ids:
            raise ValueError(
                f"item scores {sorted(item_scores)} != rubric items {sorted(item_ids)}"
            )
        for k, s in item_scores.items():
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"item {k!r} score {s} outside [0, 1]")
        factors = {p.id: p.factor for p in self.penalties}
        unknown = set(triggered) - set(factors)
        if unknown:
            raise ValueError(f"unknown penalties triggered: {sorted(unknown)}")
        base = sum(i.weight * item_scores[i.id] for i in self.items)
        for pid in set(triggered):
            base *= factors[pid]
        return base


def load_rubric(path: str | Path) -> Rubric:
    with open(path, "rb") as f:
        return Rubric.model_validate(tomllib.load(f))
