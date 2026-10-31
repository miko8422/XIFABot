"""Score paired transcripts and report warm full-utterance latency."""
import argparse
import csv
import json
import unicodedata
from pathlib import Path

import numpy as np
from pykakasi import kakasi
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent
KANA = kakasi()


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                   if not unicodedata.category(c).startswith(("P", "S", "Z", "C")))


def reading(text):
    return normalize("".join(x["hira"] for x in KANA.convert(unicodedata.normalize("NFKC", text))))


def metrics(rows):
    if not rows:
        return {}
    refs = sum(r["ref_chars"] for r in rows)
    kana_refs = sum(r["kana_chars"] for r in rows)
    return {"n": len(rows), "cer_pct": round(100 * sum(r["edits"] for r in rows) / refs, 3) if refs else None,
        "kana_cer_pct": round(100 * sum(r["kana_edits"] for r in rows) / kana_refs, 3) if kana_refs else None,
        "exact_pct": round(100 * sum(r["edits"] == 0 for r in rows) / len(rows), 2),
        "errors": sum(bool(r["error"]) for r in rows),
        "p50_ms": round(float(np.median([r["latency_ms"] for r in rows])), 1),
        "p95_ms": round(float(np.percentile([r["latency_ms"] for r in rows], 95)), 1),
        "rtf": round(sum(r["latency_ms"] for r in rows) / (1000 * sum(r["seconds"] for r in rows)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    folder = ROOT / "runs" / args.tag
    all_rows = {}
    summary = {}
    for file in sorted(folder.glob("*.jsonl")):
        model = file.stem
        rows = [json.loads(x) for x in file.read_text().splitlines()]
        for r in rows:
            ref, hyp = normalize(r["text"]), normalize(r["hypothesis"])
            kr, kh = reading(r["text"]), reading(r["hypothesis"])
            r.update(edits=Levenshtein.distance(ref, hyp), ref_chars=len(ref),
                     kana_edits=Levenshtein.distance(kr, kh), kana_chars=len(kr))
        first = [r for r in rows if r["repeat"] == 0]
        clean = [r for r in first if r["group"] in ["fleurs", "dialogue"]]
        probes = [r for r in rows if r["latency_probe"]]
        all_rows[model] = {r["id"]: r for r in clean}
        summary[model] = {"clean": metrics(clean),
            "groups": {g: metrics([r for r in first if r["group"] == g]) for g in sorted({r['group'] for r in first})},
            "short_latency": metrics(probes),
            "short_latency_by_group": {g: metrics([r for r in probes if r["group"] == g])
                                       for g in ["fleurs", "dialogue"]},
            "controls": [{"id":r["id"], "hypothesis":r["hypothesis"], "ms":r["latency_ms"], "error":r["error"]}
                         for r in first if r["group"] == "control"],
            "calls": len(rows), "errors": sum(bool(r["error"]) for r in rows)}
    baseline = all_rows.get("omlx_auto", {})
    for model, keyed in all_rows.items():
        ids = sorted(baseline.keys() & keyed.keys())
        if not ids:
            continue
        b = np.array([baseline[i]["edits"] for i in ids])
        c = np.array([keyed[i]["edits"] for i in ids])
        n = np.array([baseline[i]["ref_chars"] for i in ids])
        indices = np.random.default_rng(20261027).integers(0, len(ids), size=(2000, len(ids)))
        diffs = 100 * (c[indices].sum(axis=1) - b[indices].sum(axis=1)) / n[indices].sum(axis=1)
        summary[model]["paired_vs_production"] = {"n":len(ids), "improved_clips":int((c < b).sum()),
            "worsened_clips":int((c > b).sum()), "equal_clips":int((c == b).sum()),
            "cer_difference_pp": round(float(100 * (c.sum() - b.sum()) / n.sum()), 3),
            "bootstrap_95pct_pp": [round(float(v), 3) for v in np.percentile(diffs, [2.5,97.5])]}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Japanese ASR experiment — measurements", "", "Lower CER and latency are better.", "",
             "| Model | Clean CER % | Kana CER % | FLEURS CER % | Dialogue CER % | Short p50 ms | Short p95 ms | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        c, l, g = s["clean"], s["short_latency"], s["groups"]
        lines.append(f"| {name} | {c.get('cer_pct')} | {c.get('kana_cer_pct')} | {g.get('fleurs',{}).get('cer_pct')} | {g.get('dialogue',{}).get('cer_pct')} | {l.get('p50_ms')} | {l.get('p95_ms')} | {s['errors']} |")
    (folder / "measurements.md").write_text("\n".join(lines) + "\n")
    with (folder / "per_clip.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["model","id","group","seconds","text","hypothesis","edits","ref_chars","kana_edits","kana_chars","latency_ms","error"], extrasaction="ignore")
        writer.writeheader()
        for keyed in all_rows.values():
            writer.writerows(keyed.values())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
