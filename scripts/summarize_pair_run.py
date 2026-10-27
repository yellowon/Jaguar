#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(
    r"\[Jaguar\]\[Layer\] idx=(?P<idx>\d+) kind=(?P<kind>\S+) name=(?P<name>\S+) "
    r"total_latency_us=(?P<total>\d+) compute_latency_us=(?P<compute>\d+) "
    r"comm_bytes=(?P<comm>\d+)(?: comm_bytes_compat=(?P<comm_compat>\d+) comm_bytes_actual=(?P<comm_actual>\d+))? "
    r"rounds=(?P<rounds>\d+) details=(?P<details>.*)"
)

TEXT_PATTERNS = [
    ("encrypt", re.compile(r"Runtime for client encrypt w/ padding & new encoding \(us\): (?P<us>\d+)")),
    ("decrypt_text", re.compile(r"Runtime for client decryption \(us\): (?P<us>\d+)")),
    ("avg_pool_text", re.compile(r"Runtime for avg pooling \(us\): (?P<us>\d+)")),
    ("fc_precompute", re.compile(r"Runtime for FC precompute \(us\): (?P<us>\d+)")),
    ("fc_client_encrypt", re.compile(r"Runtime for FC client encrypt \(us\): (?P<us>\d+)")),
    ("fc_cloud", re.compile(r"Runtime for FC cloud \(us\): (?P<us>\d+)")),
    ("fc_client_decrypt", re.compile(r"Runtime for FC client decrypt \(us\): (?P<us>\d+)")),
    ("fc_total_text", re.compile(r"Runtime for FC total \(us\): (?P<us>\d+)")),
]


def is_rollup_kind(kind: str) -> bool:
    return kind.endswith("_total")


def effective_comm_bytes_sum(row: dict) -> int:
    if row["kind"] == "conv_compute":
        return 0
    return row["comm_bytes_party_sum"]


def effective_comm_bytes_max(row: dict) -> int:
    if row["kind"] == "conv_compute":
        return 0
    return row["comm_bytes_party_max"]


def parse_layer_lines(path: Path, party: str):
    events = []
    text_events = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            m = LAYER_RE.search(line)
            if m:
                data = m.groupdict()
                events.append(
                    {
                        "party": party,
                        "source_file": str(path),
                        "line_no": line_no,
                        "idx": int(data["idx"]),
                        "kind": data["kind"],
                        "name": data["name"],
                        "total_latency_us": int(data["total"]),
                        "compute_latency_us": int(data["compute"]),
                        "comm_bytes": int(data["comm"]),
                        "comm_bytes_compat": int(data["comm_compat"] or data["comm"]),
                        "comm_bytes_actual": int(data["comm_actual"] or data["comm"]),
                        "rounds": int(data["rounds"]),
                        "details": data["details"].strip(),
                    }
                )
            for kind, regex in TEXT_PATTERNS:
                mt = regex.search(line)
                if mt:
                    text_events.append(
                        {
                            "party": party,
                            "source_file": str(path),
                            "line_no": line_no,
                            "kind": kind,
                            "name": kind,
                            "total_latency_us": int(mt.group("us")),
                            "details": line.strip(),
                        }
                    )
    return events, text_events


def normalize_name(name: str) -> str:
    aliases = {
        "relu-ring-client": "relu-ring",
        "relu-ring-cloud": "relu-ring",
        "relu6-ring-client": "relu6-ring",
        "relu6-ring-cloud": "relu6-ring",
        "activation-total-client": "activation-total",
        "activation-total-cloud": "activation-total",
        "fc_cheetah_client": "fc_cheetah",
        "fc_cheetah_cloud": "fc_cheetah",
    }
    if name in aliases:
        return aliases[name]
    if name.endswith("-client"):
        return name[:-7]
    if name.endswith("-cloud"):
        return name[:-6]
    if name.endswith("_client"):
        return name[:-7]
    if name.endswith("_cloud"):
        return name[:-6]
    return name


def merge_events(client_events, cloud_events):
    by_kind_client = defaultdict(list)
    by_kind_cloud = defaultdict(list)
    for ev in client_events:
        by_kind_client[(ev["kind"], normalize_name(ev["name"]))].append(ev)
    for ev in cloud_events:
        by_kind_cloud[(ev["kind"], normalize_name(ev["name"]))].append(ev)

    merged = []
    keys = sorted(set(by_kind_client) | set(by_kind_cloud))
    for key in keys:
        client_list = by_kind_client.get(key, [])
        cloud_list = by_kind_cloud.get(key, [])
        count = max(len(client_list), len(cloud_list))
        for i in range(count):
            c = client_list[i] if i < len(client_list) else None
            s = cloud_list[i] if i < len(cloud_list) else None
            merged.append(
                {
                    "sequence": i + 1,
                    "kind": key[0],
                    "name": key[1],
                    "client_idx": c["idx"] if c else "",
                    "cloud_idx": s["idx"] if s else "",
                    "client_time_us": c["total_latency_us"] if c else 0,
                    "cloud_time_us": s["total_latency_us"] if s else 0,
                    "e2e_time_us": max(c["total_latency_us"] if c else 0, s["total_latency_us"] if s else 0),
                    "client_comm_bytes": c["comm_bytes"] if c else 0,
                    "cloud_comm_bytes": s["comm_bytes"] if s else 0,
                    "comm_bytes_party_sum": (c["comm_bytes"] if c else 0) + (s["comm_bytes"] if s else 0),
                    "comm_bytes_party_max": max(c["comm_bytes"] if c else 0, s["comm_bytes"] if s else 0),
                    "client_comm_bytes_compat": c["comm_bytes_compat"] if c else 0,
                    "cloud_comm_bytes_compat": s["comm_bytes_compat"] if s else 0,
                    "comm_bytes_compat_party_sum": (c["comm_bytes_compat"] if c else 0) + (s["comm_bytes_compat"] if s else 0),
                    "comm_bytes_compat_party_max": max(c["comm_bytes_compat"] if c else 0, s["comm_bytes_compat"] if s else 0),
                    "client_comm_bytes_actual": c["comm_bytes_actual"] if c else 0,
                    "cloud_comm_bytes_actual": s["comm_bytes_actual"] if s else 0,
                    "comm_bytes_actual_party_sum": (c["comm_bytes_actual"] if c else 0) + (s["comm_bytes_actual"] if s else 0),
                    "comm_bytes_actual_party_max": max(c["comm_bytes_actual"] if c else 0, s["comm_bytes_actual"] if s else 0),
                    "client_rounds": c["rounds"] if c else 0,
                    "cloud_rounds": s["rounds"] if s else 0,
                    "details": c["details"] if c else (s["details"] if s else ""),
                }
            )
    merged.sort(key=lambda x: (x["kind"], x["sequence"], x["name"]))
    return merged


def remap_kind_for_oc_basis(row):
    kind = row["kind"]
    name = row["name"]
    details = row.get("details", "")
    seq = int(row.get("sequence", 0) or 0)
    if kind == "encrypt" and name == "client_enc_zero_padding_quant":
        return "fc_total" if ("width=1" in details and "ct_count=1" in details) else "conv_total"
    if kind == "decrypt" and name == "client_lwe_dec_quant":
        return "conv_total"
    if kind == "decrypt" and name == "client_dec_quant":
        return "fc_total" if seq == 1 else "argmax"
    return kind


def summarize_by_kind(merged):
    sums = defaultdict(
        lambda: {
            "layers": 0,
            "e2e_time_us": 0,
            "client_time_us": 0,
            "cloud_time_us": 0,
            "comm_bytes_party_sum": 0,
            "comm_bytes_party_max": 0,
            "comm_bytes_compat_party_sum": 0,
            "comm_bytes_compat_party_max": 0,
            "comm_bytes_actual_party_sum": 0,
            "comm_bytes_actual_party_max": 0,
        }
    )
    for row in merged:
        mapped_kind = remap_kind_for_oc_basis(row)
        s = sums[mapped_kind]
        s["layers"] += 1
        s["e2e_time_us"] += row["e2e_time_us"]
        s["client_time_us"] += row["client_time_us"]
        s["cloud_time_us"] += row["cloud_time_us"]
        s["comm_bytes_party_sum"] += row["comm_bytes_party_sum"]
        s["comm_bytes_party_max"] += row["comm_bytes_party_max"]
        s["comm_bytes_compat_party_sum"] += row["comm_bytes_compat_party_sum"]
        s["comm_bytes_compat_party_max"] += row["comm_bytes_compat_party_max"]
        s["comm_bytes_actual_party_sum"] += row["comm_bytes_actual_party_sum"]
        s["comm_bytes_actual_party_max"] += row["comm_bytes_actual_party_max"]
    return sums


def summarize_client_only(merged):
    sums = defaultdict(
        lambda: {
            "layers": 0,
            "client_time_us": 0,
            "client_comm_bytes": 0,
            "client_comm_bytes_compat": 0,
            "client_comm_bytes_actual": 0,
        }
    )
    for row in merged:
        mapped_kind = remap_kind_for_oc_basis(row)
        if mapped_kind == "conv_compute":
            continue
        s = sums[mapped_kind]
        s["layers"] += 1
        s["client_time_us"] += row["client_time_us"]
        s["client_comm_bytes"] += row["client_comm_bytes"]
        s["client_comm_bytes_compat"] += row["client_comm_bytes_compat"]
        s["client_comm_bytes_actual"] += row["client_comm_bytes_actual"]
    return sums


def summarize_text_events(text_events):
    sums = defaultdict(lambda: {"count": 0, "total_latency_us": 0})
    for ev in text_events:
        sums[ev["kind"]]["count"] += 1
        sums[ev["kind"]]["total_latency_us"] += ev["total_latency_us"]
    return sums


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    if len(sys.argv) != 2:
        print("Usage: summarize_pair_run.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    client_log = run_dir / "client.log"
    cloud_log = run_dir / "cloud.log"
    if not client_log.exists() or not cloud_log.exists():
        print(f"missing client.log or cloud.log under {run_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = run_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    client_layers, client_text = parse_layer_lines(client_log, "client")
    cloud_layers, cloud_text = parse_layer_lines(cloud_log, "cloud")
    merged = merge_events(client_layers, cloud_layers)
    by_kind = summarize_by_kind(merged)
    by_kind_client = summarize_client_only(merged)
    text_summary = summarize_text_events(client_text + cloud_text)

    write_csv(
        out_dir / "raw_layers.csv",
        client_layers + cloud_layers,
        [
            "party",
            "source_file",
            "line_no",
            "idx",
            "kind",
            "name",
            "total_latency_us",
            "compute_latency_us",
            "comm_bytes",
            "comm_bytes_compat",
            "comm_bytes_actual",
            "rounds",
            "details",
        ],
    )
    write_csv(
        out_dir / "merged_layers.csv",
        merged,
        [
            "sequence",
            "kind",
            "name",
            "client_idx",
            "cloud_idx",
            "client_time_us",
            "cloud_time_us",
            "e2e_time_us",
            "client_comm_bytes",
            "cloud_comm_bytes",
            "comm_bytes_party_sum",
            "comm_bytes_party_max",
            "client_comm_bytes_compat",
            "cloud_comm_bytes_compat",
            "comm_bytes_compat_party_sum",
            "comm_bytes_compat_party_max",
            "client_comm_bytes_actual",
            "cloud_comm_bytes_actual",
            "comm_bytes_actual_party_sum",
            "comm_bytes_actual_party_max",
            "client_rounds",
            "cloud_rounds",
            "details",
        ],
    )

    kind_rows = []
    for kind in sorted(by_kind):
        row = {"kind": kind}
        row.update(by_kind[kind])
        row["oc_sent_only_party_sum"] = row["comm_bytes_compat_party_sum"]
        row["oc_sent_recv_est_party_sum"] = row["comm_bytes_compat_party_sum"] * 2
        kind_rows.append(row)
    write_csv(
        out_dir / "kind_summary.csv",
        kind_rows,
        [
            "kind",
            "layers",
            "e2e_time_us",
            "client_time_us",
            "cloud_time_us",
            "comm_bytes_party_sum",
            "comm_bytes_party_max",
            "comm_bytes_compat_party_sum",
            "comm_bytes_compat_party_max",
            "comm_bytes_actual_party_sum",
            "comm_bytes_actual_party_max",
            "oc_sent_only_party_sum",
            "oc_sent_recv_est_party_sum",
        ],
    )

    client_kind_rows = []
    for kind in sorted(by_kind_client):
        row = {"kind": kind}
        row.update(by_kind_client[kind])
        row["oc_sent_only_client"] = row["client_comm_bytes_compat"]
        row["oc_sent_recv_est_client"] = row["client_comm_bytes_compat"] * 2
        client_kind_rows.append(row)
    write_csv(
        out_dir / "kind_summary_client_only.csv",
        client_kind_rows,
        [
            "kind",
            "layers",
            "client_time_us",
            "client_comm_bytes",
            "client_comm_bytes_compat",
            "client_comm_bytes_actual",
            "oc_sent_only_client",
            "oc_sent_recv_est_client",
        ],
    )

    text_rows = []
    for kind in sorted(text_summary):
        text_rows.append(
            {
                "kind": kind,
                "count": text_summary[kind]["count"],
                "total_latency_us": text_summary[kind]["total_latency_us"],
            }
        )
    write_csv(out_dir / "text_timing_summary.csv", text_rows, ["kind", "count", "total_latency_us"])

    present_kinds = {row["kind"] for row in merged}
    rollup_children = {
        "conv_total": {"conv_compute"},
        "activation_total": {"relu-ring", "relu_ring", "relu6-ring", "relu6_ring"},
        "fc_total": set(),
    }
    excluded_leaf_kinds = set()
    for rollup, children in rollup_children.items():
        if rollup in present_kinds:
            excluded_leaf_kinds.update(children)

    excluded_total_kinds = set(excluded_leaf_kinds)
    excluded_total_kinds.add("excluded_overhead")

    structured_rows = [
        row for row in merged
        if row["kind"] not in excluded_total_kinds
    ]

    totals = {
        "run_dir": str(run_dir),
        "raw_layer_event_count": len(client_layers) + len(cloud_layers),
        "merged_layer_count": len(merged),
        "structured_total_e2e_time_us": sum(row["e2e_time_us"] for row in structured_rows),
        # These setup costs remain included in the aggregate above.
        "structured_rerandomization_key_setup_time_us": sum(
            row["e2e_time_us"] for row in structured_rows if row["kind"] == "rerandomization_setup"),
        "structured_rerandomization_precompute_time_us": sum(
            row["e2e_time_us"] for row in structured_rows if row["kind"] == "rerandomization_precompute"),
        "structured_total_comm_bytes_party_sum": sum(effective_comm_bytes_sum(row) for row in structured_rows),
        "structured_total_comm_bytes_party_max": sum(effective_comm_bytes_max(row) for row in structured_rows),
        "structured_total_comm_bytes_compat_party_sum": sum(row["comm_bytes_compat_party_sum"] for row in structured_rows if row["kind"] != "conv_compute"),
        "structured_total_comm_bytes_compat_party_max": sum(row["comm_bytes_compat_party_max"] for row in structured_rows if row["kind"] != "conv_compute"),
        "structured_total_comm_bytes_actual_party_sum": sum(row["comm_bytes_actual_party_sum"] for row in structured_rows if row["kind"] != "conv_compute"),
        "structured_total_comm_bytes_actual_party_max": sum(row["comm_bytes_actual_party_max"] for row in structured_rows if row["kind"] != "conv_compute"),
        "structured_total_comm_bytes_compat_client_only": sum(row["client_comm_bytes_compat"] for row in structured_rows if row["kind"] != "conv_compute"),
        "structured_total_comm_bytes_actual_client_only": sum(row["client_comm_bytes_actual"] for row in structured_rows if row["kind"] != "conv_compute"),
        "structured_rollup_kinds_excluded": sorted({row["kind"] for row in merged if is_rollup_kind(row["kind"])}),
        "structured_leaf_kinds_excluded": sorted(excluded_leaf_kinds),
        "structured_total_kinds_excluded": sorted(excluded_total_kinds),
        "kind_summary": kind_rows,
        "kind_summary_client_only": client_kind_rows,
        "text_timing_summary": text_rows,
        "structured_total_oc_sent_only_party_sum": sum(
              row["comm_bytes_compat_party_sum"]
              for row in structured_rows
              if row["kind"] != "conv_compute"
          ),
        "structured_total_oc_sent_recv_est_party_sum": 2 * sum(
              row["comm_bytes_compat_party_sum"]
              for row in structured_rows
              if row["kind"] != "conv_compute"
          ),
        "structured_total_oc_sent_only_client_only": sum(
              row["client_comm_bytes_compat"]
              for row in structured_rows
              if row["kind"] != "conv_compute"
          ),
        "structured_total_oc_sent_recv_est_client_only": 2 * sum(
              row["client_comm_bytes_compat"]
              for row in structured_rows
              if row["kind"] != "conv_compute"
          ),
    }
    (out_dir / "summary.json").write_text(json.dumps(totals, indent=2))

    print(f"summary_dir={out_dir}")
    print(f"structured_total_e2e_time_us={totals['structured_total_e2e_time_us']}")
    print(f"structured_total_comm_bytes_compat_client_only={totals['structured_total_comm_bytes_compat_client_only']}")
    print("kind_summary_client_only:")
    for row in client_kind_rows:
        print(
            f"  {row['kind']}: layers={row['layers']} "
            f"client_time_us={row['client_time_us']} "
            f"client_comm_compat={row['client_comm_bytes_compat']}"
        )


if __name__ == "__main__":
    main()
