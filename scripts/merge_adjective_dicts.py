"""合并形容词词典

为TOXICN和COLD数据集分别创建合并词典：
- TOXICN: toxic_adjectives_new.csv (177) + implicit_toxic_concepts.csv (20) = 197概念
- COLD:   discovered_concepts.csv (56) + implicit_toxic_concepts.csv (20) = 76概念

使用方式:
    python scripts/merge_adjective_dicts.py
"""

import csv
from pathlib import Path

project_root = Path(__file__).parent.parent
adj_dir = project_root / "data" / "raw" / "adjective"


def read_csv_rows(filepath):
    """读取CSV文件，返回(列名列表, 行字典列表)"""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    return headers, rows


def write_csv_rows(filepath, headers, rows):
    """写入CSV文件"""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def merge_for_toxicn():
    """合并TOXICN词典: 177 + 20 = 197概念

    输出格式: chinese, definition（与generate脚本兼容）
    """
    # 读取原始177个形容词
    _, rows_original = read_csv_rows(adj_dir / "toxic_adjectives_new.csv")
    base_rows = [{"chinese": r["chinese"], "definition": r["definition"]} for r in rows_original]

    # 读取新增20个隐式毒性概念
    _, rows_new = read_csv_rows(adj_dir / "implicit_toxic_concepts.csv")
    new_rows = [{"chinese": r["name"], "definition": r["definition"]} for r in rows_new]

    # 合并
    merged = base_rows + new_rows

    # 去重（基于chinese列，保留第一个）
    seen = set()
    deduped = []
    for r in merged:
        if r["chinese"] not in seen:
            seen.add(r["chinese"])
            deduped.append(r)

    # 保存
    output_path = adj_dir / "merged_toxicn_concepts.csv"
    write_csv_rows(output_path, ["chinese", "definition"], deduped)
    print(f"TOXICN合并词典: {len(base_rows)} + {len(new_rows)} -> {len(deduped)} (去重前{len(merged)})")
    print(f"保存至: {output_path}")


def merge_for_cold():
    """合并COLD词典: 56 + 20 = 76概念

    输出格式: name, definition（与discovered_concepts.csv格式一致）
    """
    # 读取原始56个概念
    _, rows_original = read_csv_rows(adj_dir / "discovered_concepts.csv")
    base_rows = [{"name": r["name"], "definition": r["definition"]} for r in rows_original]

    # 读取新增20个隐式毒性概念
    _, rows_new = read_csv_rows(adj_dir / "implicit_toxic_concepts.csv")
    new_rows = [{"name": r["name"], "definition": r["definition"]} for r in rows_new]

    # 合并
    merged = base_rows + new_rows

    # 去重（基于name列，保留第一个）
    seen = set()
    deduped = []
    for r in merged:
        if r["name"] not in seen:
            seen.add(r["name"])
            deduped.append(r)

    # 保存
    output_path = adj_dir / "merged_cold_concepts.csv"
    write_csv_rows(output_path, ["name", "definition"], deduped)
    print(f"COLD合并词典: {len(base_rows)} + {len(new_rows)} -> {len(deduped)} (去重前{len(merged)})")
    print(f"保存至: {output_path}")


if __name__ == "__main__":
    merge_for_toxicn()
    merge_for_cold()
