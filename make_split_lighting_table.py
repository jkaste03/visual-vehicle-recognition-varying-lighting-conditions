from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LIGHTING_ORDER = ["Light", "Medium", "Dark"]
SPLIT_ORDER = ["train", "val", "test"]

# Ønsket rekkefølge slik den gamle tabellen viste
PREDEFINED_LVL2_ORDER = [
    "Other",
    "3 2017–2023",
    "3 2024–nå",
    "S 2012–2015",
    "S 2016–nå",
    "X",
    "Y 2020–2024",
    "Y 2025–nå",
]


def read_split_csv(split_dir: Path, split_name: str) -> pd.DataFrame:
    path = split_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fant ikke fil: {path}")
    return pd.read_csv(path)


def normalize_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col not in df.columns:
        raise ValueError(f"Mangler kolonnen '{label_col}' i split-filene")

    labels = df[label_col].astype("string")

    if label_col == "lvl2":
        # Other har typisk tom lvl2 og må derfor fylles inn eksplisitt
        if "lvl1" in df.columns:
            lvl1 = df["lvl1"].astype("string")
            labels = labels.mask(lvl1.eq("Other") & labels.isna(), "Other")

        labels = labels.fillna("Other")
    else:
        labels = labels.fillna("Ukjent")

    return labels


def make_one_split_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if "lighting" not in df.columns:
        raise ValueError("Mangler kolonnen 'lighting' i split-filene")

    work = df.copy()
    work[label_col] = normalize_labels(work, label_col)
    work["lighting"] = work["lighting"].astype("string")

    grouped = (
        work.groupby([label_col, "lighting"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LIGHTING_ORDER, fill_value=0)
    )

    grouped["Sum"] = grouped.sum(axis=1)
    return grouped


def get_label_order(split_tables: dict[str, pd.DataFrame], label_col: str) -> list[str]:
    found_labels = []
    for split_name in SPLIT_ORDER:
        found_labels.extend(split_tables[split_name].index.tolist())

    # fjern duplikater, behold første forekomst
    found_labels = list(dict.fromkeys(found_labels))

    if label_col == "lvl2":
        ordered = [x for x in PREDEFINED_LVL2_ORDER if x in found_labels]
        rest = [x for x in found_labels if x not in PREDEFINED_LVL2_ORDER]
        return ordered + sorted(rest)

    return sorted(found_labels)


def make_split_table(split_dir: Path, label_col: str) -> pd.DataFrame:
    split_tables: dict[str, pd.DataFrame] = {}
    base_total = 0

    for split_name in SPLIT_ORDER:
        df = read_split_csv(split_dir, split_name)
        base_total += len(df)
        split_tables[split_name] = make_one_split_table(df, label_col)

    label_order = get_label_order(split_tables, label_col)

    parts = []
    for split_name in SPLIT_ORDER:
        part = split_tables[split_name].reindex(label_order, fill_value=0)
        part.columns = pd.MultiIndex.from_product([[split_name], part.columns])
        parts.append(part)

    result = pd.concat(parts, axis=1)

    result[("Total", "")] = (
        result[("train", "Sum")]
        + result[("val", "Sum")]
        + result[("test", "Sum")]
    )

    total_row = {}
    for split_name in SPLIT_ORDER:
        for col in LIGHTING_ORDER + ["Sum"]:
            total_row[(split_name, col)] = int(result[(split_name, col)].sum())
    total_row[("Total", "")] = int(result[("Total", "")].sum())

    result.loc["Total"] = pd.Series(total_row)

    df_total = int(result.loc["Total", ("Total", "")])
    print(f"[CHECK] sum(base)={base_total}  rows(df)={df_total}  OK={base_total == df_total}")

    return result


def flatten_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        f"{top}_{sub}".rstrip("_") if top != "Total" else "Total"
        for top, sub in out.columns
    ]
    out = out.reset_index().rename(columns={"index": "label"})
    return out

def render_split_table_html(df: pd.DataFrame) -> str:
    top_headers = [c[0] for c in df.columns]
    sub_headers = [c[1] for c in df.columns]

    html = []

    html.append("""
    <style>
    .split-table {
        border-collapse: collapse;
        font-size: 15px;
        border: none;
        color: #e8e8e8;
    }
    .split-table th, .split-table td {
        padding: 4px 10px;
        border: none;
    }
    .split-table thead th {
        background-color: #3b3b3b;
        color: white;
        text-align: center;
        font-weight: 600;
    }
    .split-table tbody th {
        text-align: right;
        font-weight: 400;
        color: #e8e8e8;
    }
    .split-table tbody td {
        text-align: right;
        color: #e8e8e8;
    }
    .split-table tbody tr:nth-child(odd) {
        background-color: #242424;
    }
    .split-table tbody tr:nth-child(even) {
        background-color: #2b2b2b;
    }
    .split-table .sumcell {
        color: #f0a020;
        font-weight: 700;
    }
    .split-table .group-end {
        border-right: 2px solid #6a6a6a;
        padding-right: 14px;
    }
    .split-table .totalrow th,
    .split-table .totalrow td {
        color: #f0a020;
        font-weight: 700;
        border-top: 2px solid #6a6a6a;
    }
    </style>
    """)

    html.append('<table class="split-table">')

    html.append("<thead>")

    html.append("<tr>")
    html.append("<th></th>")

    i = 0
    while i < len(top_headers):
        top = top_headers[i]
        colspan = 1
        j = i + 1
        while j < len(top_headers) and top_headers[j] == top:
            colspan += 1
            j += 1

        classes = []
        if top in {"train", "val", "test"}:
            classes.append("group-end")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""

        html.append(f'<th colspan="{colspan}"{class_attr}>{top}</th>')
        i = j
    html.append("</tr>")

    html.append("<tr>")
    html.append("<th></th>")
    for col in df.columns:
        classes = []
        if col[0] in {"train", "val", "test"} and col[1] == "Sum":
            classes.append("group-end")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        html.append(f"<th{class_attr}>{col[1]}</th>")
    html.append("</tr>")

    html.append("</thead>")

    html.append("<tbody>")
    for idx in df.index:
        row_class = "totalrow" if idx == "Total" else ""
        html.append(f'<tr class="{row_class}">')
        html.append(f"<th>{idx}</th>")

        for col in df.columns:
            val = df.loc[idx, col]
            classes = []

            is_sum_col = (col[1] == "Sum") or (col[0] == "Total")
            if is_sum_col:
                classes.append("sumcell")

            if col[0] in {"train", "val", "test"} and col[1] == "Sum":
                classes.append("group-end")

            td_class = " ".join(classes)
            class_attr = f' class="{td_class}"' if td_class else ""
            html.append(f"<td{class_attr}>{val}</td>")

        html.append("</tr>")
    html.append("</tbody>")

    html.append("</table>")
    return "".join(html)

def main() -> None:
    parser = argparse.ArgumentParser(description="Lag split-fordeling per klasse og lyskategori.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Mappe som inneholder train.csv, val.csv og test.csv",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="lvl2",
        help="Kolonnen som brukes som klasseetikett, f.eks. lvl1 eller lvl2",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Lagre tabellen som CSV i split-mappen",
    )
    args = parser.parse_args()

    split_dir = args.split_dir.resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"Fant ikke split-dir: {split_dir}")

    table = make_split_table(split_dir, args.label_col)

    table = table.copy()
    table.index.name = None
    table.columns.names = [None, None]

    print("\nSplit-fordeling per klasse og lyskategori (antall)\n")

    from IPython.display import HTML, display
    display(HTML(render_split_table_html(table)))

    if args.save_csv:
        out_path = split_dir / f"split_lighting_distribution_{args.label_col}.csv"
        flatten_for_csv(table).to_csv(out_path, index=False)
        print(f"\nLagret CSV: {out_path}")


if __name__ == "__main__":
    main()