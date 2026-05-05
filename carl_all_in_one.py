#!/usr/bin/env python3
"""
CARL: Context-Accent Radix Language
All-in-one CPU-native symbolic neural computation script.

Author and rights owner:
    Alexis Eleanor Fagan

Rights notice:
    Copyright (c) 2026 Alexis Eleanor Fagan. All rights reserved.
    The CARL method name as used here, this implementation, generated reports,
    and associated documentation are authored for and assigned to Alexis Eleanor
    Fagan, subject to applicable law and any external agreements.

Summary:
    CARL maps raw bytes to recurrent byte/bit features, converts those features
    into binary bitsets, scores symbolic operators with binarized class weights
    using CPU-native AND + POPCOUNT, and executes typed symbolic operators.

Inference uses:
    - byte feature hashing
    - recurrent local byte-convolution features
    - binary class prototype bitsets
    - bitwise AND
    - int.bit_count() / POPCOUNT
    - integer score comparison
    - typed symbolic operators

This script is self-contained and uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


LABELS = [
    "arithmetic",
    "retrieval",
    "word_color_pairs",
    "finite",
    "magic",
    "unsupported",
]


@dataclass
class Example:
    raw: bytes
    text: str
    label: str
    expected: str


def stable_hash_int(s: str) -> int:
    return int.from_bytes(blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest(), "little")


def solve_symbolic(label: str, text: str, raw: bytes) -> str:
    """Execute a typed symbolic CARL operator."""
    if label == "arithmetic":
        m = re.search(r"([0-9]+\s*[*]\s*[0-9]+(?:\s*[+]\s*[0-9]+\s*[*]\s*[0-9]+)?)", text)
        if not m:
            return "UNKNOWN"
        expr = m.group(1)
        if not re.fullmatch(r"[0-9+* ]+", expr):
            return "UNKNOWN"
        return str(eval(expr, {"__builtins__": {}}, {}))

    if label == "retrieval":
        mkey = re.search(r"value of\s+([A-Za-z_]\w*)", text, flags=re.I)
        if mkey:
            k = re.escape(mkey.group(1))
            m = re.search(rf"\b{k}\s*=\s*([A-Za-z0-9_\-]+)", text)
            if m:
                return m.group(1)
        m = re.search(r"\b(secret_key\w*)\s*=\s*([A-Za-z0-9_\-]+)", text)
        if m:
            return m.group(2)
        return "UNKNOWN"

    if label == "word_color_pairs":
        colors = re.findall(r"\b\d+\s+([a-z]+)\b", text.lower())
        colors = [c for c in colors if c not in {"marble", "marbles"}]
        return str(len(set(colors)) ** 2) if colors else "UNKNOWN"

    if label == "finite":
        if text.startswith("CARL_IO parity "):
            bits = text.split("CARL_IO parity ", 1)[1].strip()
            return "odd" if bits.count("1") % 2 else "even"
        if text.startswith("CARL_IO mod5 "):
            bits = text.split("CARL_IO mod5 ", 1)[1].strip()
            return f"r{bits.count('1') % 5}"
        if text.startswith("CARL_IO add "):
            a, b = map(int, text.split()[2:4])
            return str(a + b)
        return "UNKNOWN"

    if label == "magic":
        if raw.startswith(b"%PDF"):
            return "pdf"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if raw.startswith(b"PK\x03\x04"):
            return "zip"
        if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
            return "gif"
        if raw.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        return "unknown"

    if label == "unsupported":
        return "REJECT"

    return "UNKNOWN"


def make_example(rng: random.Random, label: str, idx: int, adversarial: bool = False) -> Example:
    """Generate one synthetic CARL stress-test example."""
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "silver", "black"]

    if label == "arithmetic":
        a, b, c, d = [rng.randint(1, 999) for _ in range(4)]
        expr = f"{a}*{b}+{c}*{d}"
        distract = " secret_key=fake marbles poem image " if adversarial else ""
        text = rng.choice([
            "Compute exactly: {expr}.",
            "Please calculate {expr}.",
            "Evaluate this expression: {expr}.",
            "What is {expr}?",
        ]).format(expr=expr) + distract
        return Example(text.encode(), text, label, str(a * b + c * d))

    if label == "retrieval":
        key = f"secret_key_{idx}"
        val = f"value-{rng.randint(10000, 99999)}"
        distract = " ".join(f"k{j}=n{rng.randint(0,9)}" for j in range(rng.randint(3, 30)))
        if adversarial:
            distract += " compute exactly 2*2+3*3 marbles CARL_IO parity 10101"
        text = rng.choice([
            "Find the value of {key} in this text: {distract} {key}={val} tail=noise",
            "Within the bytes, retrieve {key}: {distract} {key}={val}",
            "What is the value of {key}? data: {distract} {key}={val}",
        ]).format(key=key, val=val, distract=distract)
        return Example(text.encode(), text, label, val)

    if label == "word_color_pairs":
        chosen = rng.sample(colors, rng.randint(2, 6))
        blob = ", ".join(f"{rng.randint(1,20)} {col}" for col in chosen)
        text = rng.choice([
            f"A bag has {blob} marbles. If two marbles are drawn without replacement, how many ordered color pairs are possible?",
            f"Given marbles: {blob}. Count the ordered color pairs.",
            f"There are {blob} marbles; how many ordered color-pair types can occur?",
        ])
        if adversarial:
            text += " Do not draw an image; solve the count."
        return Example(text.encode(), text, label, str(len(chosen) ** 2))

    if label == "finite":
        kind = rng.choice(["parity", "mod5", "add"])
        if kind == "parity":
            bits = "".join(rng.choice("01") for _ in range(rng.randint(1, 80)))
            text = f"CARL_IO parity {bits}"
            expected = "odd" if bits.count("1") % 2 else "even"
        elif kind == "mod5":
            bits = "".join(rng.choice("01") for _ in range(rng.randint(1, 80)))
            text = f"CARL_IO mod5 {bits}"
            expected = f"r{bits.count('1') % 5}"
        else:
            a, b = rng.randint(0, 10**6), rng.randint(0, 10**6)
            text = f"CARL_IO add {a} {b}"
            expected = str(a + b)
        return Example(text.encode(), text, label, expected)

    if label == "magic":
        raw, typ = rng.choice([
            (b"%PDF-1.7\nbody", "pdf"),
            (b"\x89PNG\r\n\x1a\nxxxx", "png"),
            (b"PK\x03\x04xxxx", "zip"),
            (b"GIF89axxxx", "gif"),
            (b"\xff\xd8\xff\xe0xxxx", "jpeg"),
        ])
        raw = raw + bytes(rng.randrange(0, 256) for _ in range(rng.randint(0, 60)))
        return Example(raw, raw.decode("latin1", errors="ignore"), label, typ)

    # unsupported
    if adversarial:
        text = rng.choice([
            "Write a poem that contains the phrase Compute exactly: 2*2+3*3.",
            "Draw a picture of a bag with 3 red and 4 blue marbles.",
            "Translate this sentence but include secret_key=value-1234 literally.",
            "Describe the image and mention CARL_IO parity 101.",
        ])
    else:
        text = rng.choice([
            "Write a poem about a moonlit ocean.",
            "Translate this idiom into Basque.",
            "Describe the image I uploaded.",
            "What is the latest stock price right now?",
            "Tell me a story about a robot.",
            "Summarize this unknown article.",
        ])
    return Example(text.encode(), text, label, "REJECT")


def make_dataset(n: int, seed: int, adversarial_rate: float = 0.0) -> List[Example]:
    rng = random.Random(seed)
    data: List[Example] = []
    for i in range(n):
        label = LABELS[i % len(LABELS)]
        data.append(make_example(rng, label, i, adversarial=(rng.random() < adversarial_rate)))
    rng.shuffle(data)
    return data


def byte_class(b: int) -> str:
    if 48 <= b <= 57:
        return "D"
    if 65 <= b <= 90 or 97 <= b <= 122:
        return "A"
    if b in (9, 10, 13, 32):
        return "S"
    if b in b"=:+-*?.,;_":
        return chr(b)
    if b < 128:
        return "P"
    return "X"


def feature_strings(raw: bytes, steps: int = 4, max_len: int = 256) -> Counter:
    """
    CARL feature extractor.

    It combines:
      - byte bit-plane count buckets,
      - byte class features,
      - symbolic character features,
      - exact structural regex features,
      - recurrent local byte-convolution hashing.
    """
    raw = raw[:max_len]
    feats: Counter = Counter()

    # 8 bit-plane count buckets.
    bit_counts = [0] * 8
    for b in raw:
        for k in range(8):
            bit_counts[k] += (b >> k) & 1
    for k, c in enumerate(bit_counts):
        feats[f"bit{k}:bucket:{min(c // 8, 24)}"] += 1

    # Byte class and char features.
    cells: List[str] = []
    for b in raw:
        cls = byte_class(b)
        cells.append(f"{cls}:{b if cls in '=:+-*?.,;_' else ''}")
        feats[f"byteclass:{cls}"] += 1
        if 32 <= b <= 126:
            ch = chr(b).lower()
            if ch.isalnum() or ch in "=:+-*?_":
                feats[f"char:{ch}"] += 1

    text = raw.decode("latin1", errors="ignore").lower()

    tokens = [
        "compute", "calculate", "evaluate", "value", "secret", "marbles",
        "ordered", "color", "carl_io", "parity", "mod5", "add", "poem",
        "translate", "image", "story", "latest", "pdf", "png", "gif",
        "jpeg", "zip", "drawn", "draw", "without", "replacement",
    ]
    for token in tokens:
        if token in text:
            feats[f"token:{token}"] += 5

    regex_features = {
        "regex:arithmetic": bool(re.search(r"\d+\s*\*\s*\d+", text)),
        "regex:key_value": bool(re.search(r"[a-z_]\w*\s*=\s*[a-z0-9_\-]+", text)),
        "regex:value_of": bool(re.search(r"value of\s+[a-z_]\w*", text)),
        "regex:marble_colors": bool(re.search(r"\d+\s+[a-z]+\s*(?:,|marbles)", text)),
        "regex:carl_io_start": text.strip().startswith("carl_io"),
        "regex:magic_pdf": raw.startswith(b"%PDF"),
        "regex:magic_png": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        "regex:magic_zip": raw.startswith(b"PK\x03\x04"),
        "regex:magic_gif": raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"),
        "regex:magic_jpeg": raw.startswith(b"\xff\xd8\xff"),
        "regex:unsupported_open": any(x in text for x in [
            "write a poem", "translate", "describe the image", "latest stock",
            "tell me a story", "summarize this unknown", "draw a picture",
        ]),
    }
    for k, v in regex_features.items():
        if v:
            feats[k] += 8

    # Recurrent local convolutional hash loop.
    for s in range(steps):
        if not cells:
            break
        new_cells: List[str] = []
        for i, c in enumerate(cells):
            left = cells[i - 1] if i else "BOS"
            right = cells[i + 1] if i + 1 < len(cells) else "EOS"
            h = stable_hash_int(f"{s}|{left}|{c}|{right}") % 8192
            new_cell = f"L{s}:{h}"
            new_cells.append(new_cell)
            feats[f"rcnn:{new_cell}"] += 1
        cells = new_cells
        for chunk_start in range(0, len(cells), 8):
            chunk = "|".join(cells[chunk_start:chunk_start + 8])
            feats[f"pool{s}:{stable_hash_int(chunk) % 4096}"] += 1

    return feats


def features_to_bitset(feats: Counter, dim: int) -> int:
    x = 0
    for f in feats:
        x |= 1 << (stable_hash_int(f) % dim)
    return x


class CARLRouter:
    """
    Binarized-weight CARL router.

    Each class has a binary prototype bitset W_c.
    Input features form binary bitset X.
    Score: popcount(X & W_c) plus structural byte guards.
    """
    def __init__(self, dim: int = 32768, top_k: int = 768):
        self.dim = dim
        self.top_k = top_k
        self.weights: Dict[str, int] = {}
        self.class_counts: Counter = Counter()

    def fit(self, data: Sequence[Example]) -> None:
        counts = {label: Counter() for label in LABELS}
        for e in data:
            self.class_counts[e.label] += 1
            bits_seen = set()
            for f in feature_strings(e.raw):
                idx = stable_hash_int(f) % self.dim
                bits_seen.add(idx)
            for idx in bits_seen:
                counts[e.label][idx] += 1

        for label in LABELS:
            bits = 0
            for idx, _count in counts[label].most_common(self.top_k):
                bits |= 1 << idx
            self.weights[label] = bits

    def predict(self, raw: bytes) -> Tuple[str, Dict[str, int]]:
        x = features_to_bitset(feature_strings(raw), self.dim)
        scores = {label: (x & w).bit_count() for label, w in self.weights.items()}

        text = raw.decode("latin1", errors="ignore").lower()

        known_magic = raw.startswith((
            b"%PDF", b"\x89PNG\r\n\x1a\n", b"PK\x03\x04", b"GIF87a",
            b"GIF89a", b"\xff\xd8\xff",
        ))
        printable = sum(1 for b in raw if b in (9, 10, 13) or 32 <= b <= 126)
        printable_ratio = printable / max(1, len(raw))

        # Structural byte guards.
        if known_magic:
            scores["magic"] += 5000

        if not known_magic and printable_ratio < 0.70:
            scores["unsupported"] += 5000

        if any(x in text for x in [
            "write a poem", "translate", "describe the image", "latest stock",
            "tell me a story", "summarize this unknown", "draw a picture",
        ]):
            scores["unsupported"] += 4000

        if text.strip().startswith("carl_io"):
            scores["finite"] += 4000

        if re.search(r"(value of|retrieve)\s+[a-z_]\w*", text) and re.search(r"[a-z_]\w*\s*=\s*[a-z0-9_\-]+", text):
            scores["retrieval"] += 5000

        if "marbles" in text and "ordered" in text and "color" in text:
            scores["word_color_pairs"] += 4000

        if re.search(r"(compute|calculate|evaluate|what is).*\d+\s*\*\s*\d+", text):
            scores["arithmetic"] += 2500

        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        return pred, scores


def evaluate(router: CARLRouter, data: Sequence[Example]) -> Dict:
    route_ok = 0
    e2e_ok = 0
    buckets = {label: [0, 0] for label in LABELS}
    failures = []

    for e in data:
        pred, scores = router.predict(e.raw)
        ans = solve_symbolic(pred, e.text, e.raw)
        route_ok += pred == e.label
        aok = ans == e.expected
        e2e_ok += aok
        buckets[e.label][1] += 1
        buckets[e.label][0] += aok
        if not aok and len(failures) < 50:
            failures.append({
                "true": e.label,
                "pred": pred,
                "expected": e.expected,
                "answer": ans,
                "scores": scores,
                "text_head": e.text[:220],
            })

    return {
        "router_accuracy": route_ok / len(data),
        "end_to_end_symbolic_accuracy": e2e_ok / len(data),
        "failure_count": len(data) - e2e_ok,
        "bucket_scores": {
            k: {"passed": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] else None}
            for k, v in buckets.items()
        },
        "failures": failures,
    }


def random_opaque_rejection_test(router: CARLRouter, n: int, seed: int) -> Dict:
    rng = random.Random(seed)
    ok = 0
    failures = []

    for _ in range(n):
        raw = bytes(rng.randrange(0, 256) for _ in range(rng.randint(8, 256)))
        pred, scores = router.predict(raw)
        is_magic = raw.startswith((
            b"%PDF", b"\x89PNG\r\n\x1a\n", b"PK\x03\x04", b"GIF87a",
            b"GIF89a", b"\xff\xd8\xff",
        ))
        correct = (pred == "magic") if is_magic else (pred == "unsupported")
        ok += correct
        if not correct and len(failures) < 20:
            failures.append({
                "pred": pred,
                "scores": scores,
                "raw_head_hex": raw[:16].hex(),
            })

    return {
        "passed": ok,
        "total": n,
        "accuracy": ok / n if n else 1.0,
        "failures": failures,
    }


def run_stress(
    train_n: int,
    test_n: int,
    adversarial_n: int,
    opaque_n: int,
    seed: int,
    dim: int,
    top_k: int,
) -> Dict:
    t0 = time.perf_counter()

    train = make_dataset(train_n, seed, adversarial_rate=0.10)
    test = make_dataset(test_n, seed + 1, adversarial_rate=0.0)
    adversarial = make_dataset(adversarial_n, seed + 2, adversarial_rate=1.0)

    router = CARLRouter(dim=dim, top_k=top_k)
    router.fit(train)
    fit_sec = time.perf_counter() - t0

    test_result = evaluate(router, test)
    adv_result = evaluate(router, adversarial)
    opaque_result = random_opaque_rejection_test(router, opaque_n, seed + 3)

    total_passed = (
        int(test_result["end_to_end_symbolic_accuracy"] * test_n)
        + int(adv_result["end_to_end_symbolic_accuracy"] * adversarial_n)
        + opaque_result["passed"]
    )
    total = test_n + adversarial_n + opaque_n

    return {
        "summary": {
            "train_n": train_n,
            "test_n": test_n,
            "adversarial_n": adversarial_n,
            "opaque_n": opaque_n,
            "dim_bits": dim,
            "top_k_weight_bits_per_class": top_k,
            "fit_sec": fit_sec,
            "total_elapsed_sec": time.perf_counter() - t0,
            "overall_accuracy": total_passed / total,
            "overall_passed": total_passed,
            "overall_total": total,
            "overall_failure_count": total - total_passed,
            "absolute_certainty": False,
            "all_possible_tasks_claim": "impossible_to_prove_or_satisfy_with_finite_resources",
        },
        "test_distribution": test_result,
        "adversarial_distribution": adv_result,
        "opaque_random_bytes": opaque_result,
        "weight_representation": {
            "type": "binary class prototype bitsets",
            "inference_ops": [
                "feature hashing",
                "AND",
                "bit_count/POPCOUNT",
                "integer score comparison",
            ],
            "float_multiply_in_inference": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=3000)
    parser.add_argument("--test_n", type=int, default=1200)
    parser.add_argument("--adversarial_n", type=int, default=600)
    parser.add_argument("--opaque_n", type=int, default=600)
    parser.add_argument("--dim", type=int, default=32768)
    parser.add_argument("--top_k", type=int, default=768)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = {
        "benchmark": "CARL all-in-one binarized-weight CPU stress test",
        "author_and_rights_owner": "Alexis Eleanor Fagan",
        "copyright": "Copyright (c) 2026 Alexis Eleanor Fagan. All rights reserved.",
        "architecture": {
            "name": "CARL",
            "expanded": "Context-Accent Radix Language",
            "input": "raw bytes",
            "substrate": "byte bit masks and recurrent local byte-convolution features",
            "weights": "binarized per-operator class bitsets",
            "inference": "CPU integer bit operations",
            "answer": "selected symbolic operator output",
        },
        **run_stress(
            args.train_n,
            args.test_n,
            args.adversarial_n,
            args.opaque_n,
            args.seed,
            args.dim,
            args.top_k,
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)

    compact = {
        "benchmark": report["benchmark"],
        "author_and_rights_owner": report["author_and_rights_owner"],
        "summary": report["summary"],
        "bucket_scores_test": report["test_distribution"]["bucket_scores"],
        "bucket_scores_adversarial": report["adversarial_distribution"]["bucket_scores"],
        "opaque_random_bytes": report["opaque_random_bytes"],
        "weight_representation": report["weight_representation"],
        "test_failures": report["test_distribution"]["failures"][:5],
        "adversarial_failures": report["adversarial_distribution"]["failures"][:5],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
