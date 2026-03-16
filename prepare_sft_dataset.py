import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Record:
    grade: str
    color: str
    properties_raw: str
    formula_raw: str


_ENTRY_RE = re.compile(r"产品牌号：(?P<grade>[^，。]+)，产品色号：(?P<color>[^，。]+)，(?P<body>.*)", re.DOTALL)
_FORMULA_SPLIT_RE = re.compile(r"(?:的推荐配方为：|该产品的推荐配方为：)")


def _clean_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _iter_entries(text: str) -> Iterable[str]:
    # Split by newline first; each line generally contains a full entry.
    for line in text.splitlines():
        line = _clean_text(line)
        if not line:
            continue
        # Some lines may contain multiple entries; split on "产品牌号：" occurrences.
        if line.count("产品牌号：") <= 1:
            yield line
            continue

        parts = line.split("产品牌号：")
        for part in parts:
            part = _clean_text(part)
            if not part:
                continue
            yield "产品牌号：" + part


def parse_records(text: str) -> List[Record]:
    records: List[Record] = []
    for entry in _iter_entries(text):
        m = _ENTRY_RE.match(entry)
        if not m:
            # Skip unmatched lines rather than failing hard.
            continue
        grade = _clean_text(m.group("grade"))
        color = _clean_text(m.group("color"))
        body = _clean_text(m.group("body"))

        parts = _FORMULA_SPLIT_RE.split(body, maxsplit=1)
        if len(parts) != 2:
            continue
        properties_raw = _clean_text(parts[0]).rstrip("。")
        formula_raw = _clean_text(parts[1]).rstrip("。")

        records.append(
            Record(
                grade=grade,
                color=color,
                properties_raw=properties_raw,
                formula_raw=formula_raw,
            )
        )
    return records


def _extract_key_metrics(properties_raw: str) -> Dict[str, str]:
    # Keep a small set of metrics that are most commonly queried.
    keys = [
        "拉伸强度",
        "断裂伸长率",
        "弯曲强度",
        "弯曲模量",
        "悬臂梁缺口冲击强度",
        "密度",
        "熔体流动速率",
        "热变形",
        "维卡",
        "拉伸模量",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        # Example: 拉伸强度为26.8 MPa
        # Some values may be "无数据".
        m = re.search(rf"{re.escape(k)}为(?P<v>[^，。]+)", properties_raw)
        if not m:
            continue
        v = _clean_text(m.group("v"))
        if v == "无数据":
            continue
        out[k] = v
    return out


def _split_formula_items(formula_raw: str) -> List[str]:
    # Separate by Chinese comma and standard comma.
    raw_items = re.split(r"[，,]", formula_raw)
    items = []
    for it in raw_items:
        it = _clean_text(it)
        if not it:
            continue
        items.append(it)
    return items


def _normalize_formula(formula_raw: str) -> str:
    # Normalize formula to compare similarity across same grade.
    items = _split_formula_items(formula_raw)
    norm_items = []
    for it in items:
        it = it.replace("：", ":")
        it = re.sub(r"\s+", "", it)
        norm_items.append(it)
    # Keep order to avoid over-normalizing.
    return ";".join(norm_items)


def _canonical_per_grade(records: List[Record]) -> Dict[str, Record]:
    by_grade: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        by_grade[r.grade].append(r)

    canonical: Dict[str, Record] = {}
    for grade, rs in by_grade.items():
        if len(rs) == 1:
            canonical[grade] = rs[0]
            continue

        formula_counter = Counter(_normalize_formula(r.formula_raw) for r in rs)
        best_norm, _ = formula_counter.most_common(1)[0]
        # Pick the first record that matches the most common formula.
        best = next(r for r in rs if _normalize_formula(r.formula_raw) == best_norm)
        canonical[grade] = best

    return canonical


def _answer_formula(record: Record) -> str:
    metrics = _extract_key_metrics(record.properties_raw)
    lines = [f"牌号：{record.grade}", f"色号：{record.color}", "", "推荐配方："]
    for it in _split_formula_items(record.formula_raw):
        lines.append(f"- {it}")
    if metrics:
        lines.append("")
        lines.append("关键性能(供参考)：")
        for k, v in metrics.items():
            lines.append(f"- {k}：{v}")
    return "\n".join(lines).strip()


def _answer_properties(record: Record) -> str:
    metrics = _extract_key_metrics(record.properties_raw)
    lines = [f"牌号：{record.grade}", f"色号：{record.color}", ""]
    if metrics:
        lines.append("关键性能：")
        for k, v in metrics.items():
            lines.append(f"- {k}：{v}")
    else:
        lines.append("该条记录未抽取到关键性能字段（原始数据可能缺失或格式不一致）。")
    lines.append("")
    lines.append("推荐配方(如需)：")
    for it in _split_formula_items(record.formula_raw):
        lines.append(f"- {it}")
    return "\n".join(lines).strip()


def _answer_compare(a: Record, b: Record) -> str:
    am = _extract_key_metrics(a.properties_raw)
    bm = _extract_key_metrics(b.properties_raw)

    keys = [
        "拉伸强度",
        "断裂伸长率",
        "弯曲模量",
        "悬臂梁缺口冲击强度",
        "密度",
        "熔体流动速率",
        "热变形",
    ]

    lines = [
        f"对比牌号：{a.grade} vs {b.grade}",
        "",
        "关键性能对比(供参考)：",
    ]
    for k in keys:
        av = am.get(k, "无数据")
        bv = bm.get(k, "无数据")
        lines.append(f"- {k}：{a.grade}={av}；{b.grade}={bv}")

    lines.append("")
    lines.append("配方要点：")
    lines.append(f"- {a.grade}：" + ", ".join(_split_formula_items(a.formula_raw)[:6]) + ("..." if len(_split_formula_items(a.formula_raw)) > 6 else ""))
    lines.append(f"- {b.grade}：" + ", ".join(_split_formula_items(b.formula_raw)[:6]) + ("..." if len(_split_formula_items(b.formula_raw)) > 6 else ""))

    lines.append("")
    lines.append("结论建议：")
    lines.append("- 若更关注韧性/冲击：优先比较悬臂梁缺口冲击强度与断裂伸长率。")
    lines.append("- 若更关注刚性/尺寸稳定：优先比较弯曲模量、热变形温度与填料占比。")
    return "\n".join(lines).strip()


def _format_formula_block(record: Record) -> str:
    lines = ["配方："]
    for it in _split_formula_items(record.formula_raw):
        lines.append(f"- {it}")
    return "\n".join(lines).strip()


def _answer_predicted_properties(record: Record) -> str:
    metrics = _extract_key_metrics(record.properties_raw)
    if not metrics:
        return "该配方在原始数据中未提供可抽取的关键性能指标，无法基于现有样本给出预测。"

    lines = ["预测关键性能(基于已有样本的对应关系)："]
    for k, v in metrics.items():
        lines.append(f"- {k}：{v}")
    lines.append("")
    lines.append("说明：以上为数据集中该配方对应的性能记录，用于配方-性能映射训练；真实结果需实验验证。")
    return "\n".join(lines).strip()


def _format_target_metrics(record: Record, rng: random.Random, min_k: int = 3, max_k: int = 6) -> Tuple[str, Dict[str, str]]:
    metrics = _extract_key_metrics(record.properties_raw)
    if not metrics:
        return "目标性能：无可用指标", {}

    keys = list(metrics.keys())
    rng.shuffle(keys)
    k = min(len(keys), rng.randint(min_k, max_k))
    chosen = {kk: metrics[kk] for kk in keys[:k]}

    lines = ["目标性能(期望/约束)："]
    for kk, vv in chosen.items():
        lines.append(f"- {kk}：约 {vv}")
    return "\n".join(lines).strip(), chosen


def _answer_recommended_formula(record: Record, target_metrics: Dict[str, str]) -> str:
    lines = ["建议方案(从已有样本中匹配)：", f"推荐参考牌号：{record.grade}", f"参考色号：{record.color}", "", "推荐配方："]
    for it in _split_formula_items(record.formula_raw):
        lines.append(f"- {it}")

    if target_metrics:
        lines.append("")
        lines.append("对应样本的关键性能(供核对)：")
        full = _extract_key_metrics(record.properties_raw)
        for kk in target_metrics.keys():
            vv = full.get(kk)
            if vv:
                lines.append(f"- {kk}：{vv}")

    lines.append("")
    lines.append("说明：该建议来源于历史配方-性能样本的近似匹配；如需严格达标，建议做小试并按冲击/模量/流动等指标微调填料与增韧体系。")
    return "\n".join(lines).strip()


def build_sft_samples(
    records: List[Record],
    seed: int = 7,
    formula_to_properties_per_record: int = 2,
    target_to_formula_per_record: int = 2,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)

    canonical = _canonical_per_grade(records)
    grade_list = sorted(canonical.keys())

    samples: List[Dict[str, str]] = []

    # Task A: formula lookup per grade (canonical)
    for grade in grade_list:
        r = canonical[grade]
        samples.append(
            {
                "instruction": "请给出指定牌号的推荐配方，并补充关键性能指标（若数据中有）。",
                "input": f"牌号：{grade}\n问题：{grade} 的配方是什么？",
                "output": _answer_formula(r),
            }
        )

    # Task B: properties lookup (use some variants for robustness)
    for r in records:
        samples.append(
            {
                "instruction": "请查询并整理该牌号的关键性能指标（若有），并给出推荐配方。",
                "input": f"牌号：{r.grade}\n色号：{r.color}",
                "output": _answer_properties(r),
            }
        )

    # Task C: compare (random pairs)
    pairs = set()
    target_pairs = min(2000, len(grade_list) * 3)
    if len(grade_list) >= 2:
        while len(pairs) < target_pairs:
            a, b = rng.sample(grade_list, 2)
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in pairs:
                continue
            pairs.add(key)

        for a, b in sorted(pairs):
            ra, rb = canonical[a], canonical[b]
            samples.append(
                {
                    "instruction": "请对比两个材料牌号的关键性能差异，并给出选型建议。",
                    "input": f"需要对比的牌号：{a} vs {b}",
                    "output": _answer_compare(ra, rb),
                }
            )

    # Task D: formula -> predict properties
    # Use only records with extractable metrics.
    if formula_to_properties_per_record > 0:
        eligible = [r for r in records if _extract_key_metrics(r.properties_raw)]
        for r in eligible:
            for _ in range(formula_to_properties_per_record):
                samples.append(
                    {
                        "instruction": "已知一个材料的配方，请预测其关键性能指标（如拉伸强度、弯曲模量、冲击强度等）。如果无法判断，请明确说明原因。",
                        "input": _format_formula_block(r),
                        "output": _answer_predicted_properties(r),
                    }
                )

    # Task E: target properties -> recommend formula
    if target_to_formula_per_record > 0:
        eligible = [r for r in records if _extract_key_metrics(r.properties_raw)]
        for r in eligible:
            for _ in range(target_to_formula_per_record):
                target_text, chosen = _format_target_metrics(r, rng=rng)
                samples.append(
                    {
                        "instruction": "已知目标性能需求，请从已有样本中给出最接近的推荐配方，并说明该配方与目标的匹配点与可能的取舍。",
                        "input": target_text,
                        "output": _answer_recommended_formula(r, chosen),
                    }
                )

    rng.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("task/input/shuju.txt"))
    parser.add_argument("--out_dir", type=Path, default=Path("finetune/data"))
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--formula_to_properties_per_record",
        type=int,
        default=2,
        help="每条记录生成多少条‘配方->预测性能’样本；设为0可关闭。",
    )
    parser.add_argument(
        "--target_to_formula_per_record",
        type=int,
        default=2,
        help="每条记录生成多少条‘目标性能->推荐配方’样本；设为0可关闭。",
    )
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    records = parse_records(raw)
    if not records:
        raise SystemExit("No records parsed. Please check input format.")

    samples = build_sft_samples(
        records,
        seed=args.seed,
        formula_to_properties_per_record=args.formula_to_properties_per_record,
        target_to_formula_per_record=args.target_to_formula_per_record,
    )
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    val_n = max(1, int(len(samples) * args.val_ratio))
    val = samples[:val_n]
    train = samples[val_n:]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    (args.out_dir / "records.jsonl").write_text(
        "\n".join(
            json.dumps(r.__dict__, ensure_ascii=False) for r in records
        )
        + "\n",
        encoding="utf-8",
    )

    (args.out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in train) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in val) + "\n",
        encoding="utf-8",
    )

    print(f"Parsed records: {len(records)}")
    print(f"SFT samples: {len(samples)} (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()
