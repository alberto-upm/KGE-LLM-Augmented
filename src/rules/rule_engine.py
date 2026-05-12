"""
Motor de inferencia en cascada basado en reglas AnyBURL.

Carga las reglas del fichero TSV generado por AnyBURL, las indexa por condición de
cuerpo para lookup O(1) y expone `infer()` para obtener la sugerencia de mayor
confianza para un predicado objetivo dado el estado conocido de la incidencia.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Rule:
    rule_id: str                     # e.g. "r_00001"
    head_pred: str                   # predicate en la cabeza
    head_value: str                  # valor constante en la cabeza
    body: list[tuple[str, str]]      # [(pred, valor), ...] — todas las condiciones del cuerpo
    confidence: float
    support: int


# Patrón para átomos del tipo  predicado(X, valor)
_ATOM_RE = re.compile(r'(\w+)\(X,([^)]+)\)')


def _is_variable(token: str) -> bool:
    """Devuelve True si el token es una variable AnyBURL (una sola letra mayúscula)."""
    return len(token) == 1 and token.isupper()


def _parse_rule(rule_text: str) -> Optional[tuple[str, str, list[tuple[str, str]]]]:
    """
    Parsea una regla AnyBURL del tipo:
        head_pred(X, head_val) <= body_pred1(X, val1), body_pred2(X, val2)

    Devuelve (head_pred, head_val, body_list) o None si la regla no es aplicable
    (variable en cabeza, variable en cuerpo, o formato no reconocido).
    """
    if "<=" not in rule_text:
        return None

    head_part, body_part = rule_text.split("<=", 1)

    head_m = _ATOM_RE.match(head_part.strip())
    if not head_m:
        return None

    head_pred = head_m.group(1)
    head_val = head_m.group(2).strip()

    if _is_variable(head_val):
        return None

    body_atoms = _ATOM_RE.findall(body_part)
    if not body_atoms:
        return None

    body: list[tuple[str, str]] = []
    for bp, bv in body_atoms:
        bv = bv.strip()
        if _is_variable(bv):
            return None
        body.append((bp, bv))

    return head_pred, head_val, body


class RuleEngine:
    """
    Motor de reglas AnyBURL para inferencia en cascada.

    Uso:
        engine = RuleEngine(Path("data/reglas/rules-1000-3"))
        hint = engine.infer({"int_hasCustomer": "company_149..."}, "incident_hasOrigin")
        # → {"value": "incidentOrigin__3", "source": "RULE", "rule_id": "r_00001", "confidence": 1.0}
        # → None  si ninguna regla aplica
    """

    def __init__(self, rules_path: Path, min_confidence: float = 0.0):
        self._rules: list[Rule] = []
        # Índice (body_pred, body_value) → reglas que tienen esa condición en el cuerpo
        self._index: dict[tuple[str, str], list[Rule]] = {}
        # Conjunto de predicados de cabeza presentes (para diagnóstico)
        self._rules_by_head: dict[str, list[Rule]] = {}

        if rules_path.exists():
            self._load(rules_path, min_confidence)

    # ------------------------------------------------------------------
    # Carga y parseo
    # ------------------------------------------------------------------

    def _load(self, path: Path, min_confidence: float) -> None:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                try:
                    support = int(parts[0])
                    confidence = float(parts[2])
                except ValueError:
                    continue

                if confidence < min_confidence:
                    continue

                parsed = _parse_rule(parts[3])
                if parsed is None:
                    continue

                head_pred, head_val, body = parsed
                rule = Rule(
                    rule_id=f"r_{line_no:05d}",
                    head_pred=head_pred,
                    head_value=head_val,
                    body=body,
                    confidence=confidence,
                    support=support,
                )
                self._rules.append(rule)
                self._rules_by_head.setdefault(head_pred, []).append(rule)
                for bp, bv in body:
                    self._index.setdefault((bp, bv), []).append(rule)

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def infer(
        self,
        known_props: dict[str, Optional[str]],
        target_pred: str,
    ) -> Optional[dict]:
        """
        Intenta disparar la regla de mayor confianza que:
          1. Tiene `target_pred` como predicado de cabeza.
          2. Todas sus condiciones de cuerpo están satisfechas en `known_props`.

        Devuelve un dict de trazabilidad o None si ninguna regla aplica:
            {"value": str, "source": "RULE", "rule_id": str, "confidence": float}
        """
        best: Optional[dict] = None
        seen: set[str] = set()

        for (bp, bv), rules in self._index.items():
            # Saltar grupos cuya condición clave no está satisfecha
            if known_props.get(bp) != bv:
                continue

            for rule in rules:
                if rule.rule_id in seen or rule.head_pred != target_pred:
                    continue
                seen.add(rule.rule_id)

                # Comprobar que TODAS las condiciones del cuerpo están satisfechas
                if not all(known_props.get(cb) == cv for cb, cv in rule.body):
                    continue

                if best is None or rule.confidence > best["confidence"]:
                    best = {
                        "value":      rule.head_value,
                        "source":     "RULE",
                        "rule_id":    rule.rule_id,
                        "confidence": rule.confidence,
                    }

        return best

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "total_rules":  len(self._rules),
            "head_preds":   list(self._rules_by_head.keys()),
            "index_keys":   len(self._index),
        }
