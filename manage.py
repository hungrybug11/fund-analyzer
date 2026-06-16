#!/usr/bin/env python
"""Fund holdings manager — no more editing YAML by hand.

Usage:
    python manage.py                  # Interactive menu
    python manage.py list             # Show current holdings
    python manage.py add              # Add a fund interactively
    python manage.py remove CODE      # Remove a fund
    python manage.py sync             # Sync CSV → config.yaml
    python manage.py check            # Validate holdings.csv

Your holdings live in holdings.csv — edit it directly in Excel,
VS Code, or any text editor. Then run `python manage.py sync`
to update config.yaml. Or use the interactive add/remove commands.
"""

import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CSV_PATH = Path(__file__).parent / "holdings.csv"
CONFIG_PATH = Path(__file__).parent / "config.yaml"

FUND_TYPES = {
    "1": ("a_share_mf",  "A股公募基金（如 110020）"),
    "2": ("a_share_etf", "A股场内ETF（如 510300）"),
    "3": ("overseas_etf","海外ETF（如 SPY / QQQ）"),
}

CURRENCIES = {"1": "CNY", "2": "USD", "3": "HKD"}


# ── CSV read/write ──────────────────────────────────────────────

def read_holdings() -> list[dict]:
    """Read holdings from CSV. Returns list of dicts."""
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("code", "").strip()]


def write_holdings(rows: list[dict]) -> None:
    """Write holdings to CSV."""
    fieldnames = ["code", "name", "type", "weight", "benchmark", "currency"]
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 已保存 {len(rows)} 只基金到 {CSV_PATH.name}")


# ── Commands ────────────────────────────────────────────────────

def cmd_list():
    """Show current holdings."""
    rows = read_holdings()
    if not rows:
        print("📭 持仓为空。运行 `python manage.py add` 添加基金。")
        return

    total = sum(float(r["weight"]) for r in rows)
    print(f"\n📋 当前持仓（{len(rows)} 只基金，权重合计 {total:.1%}）：\n")
    print(f"{'代码':<8} {'名称':<30} {'类型':<14} {'权重':>6} {'货币':>4}")
    print("-" * 66)
    for r in rows:
        print(f"{r['code']:<8} {r['name']:<30} {r['type']:<14} {float(r['weight']):>5.1%} {r['currency']:>4}")

    if abs(total - 1.0) > 0.02:
        print(f"\n⚠️  权重合计 {total:.1%}，不等于 100%，请调整！")


def cmd_add():
    """Interactive add a fund."""
    print("\n➕ 添加基金\n" + "=" * 40)

    code = input("基金代码（6位数字或美股代码如 SPY）: ").strip().upper()
    if not code:
        print("❌ 代码不能为空")
        return

    name = input("基金简称（如 永赢半导体C）: ").strip()
    if not name:
        print("❌ 名称不能为空")
        return

    print("\n基金类型：")
    for k, (_, desc) in FUND_TYPES.items():
        print(f"  {k}. {desc}")
    t = input("选类型 [1/2/3，默认1]: ").strip() or "1"
    ftype = FUND_TYPES.get(t, FUND_TYPES["1"])[0]

    weight_str = input("持仓占比（如 0.15 表示15%，默认0.10）: ").strip()
    try:
        weight = float(weight_str) if weight_str else 0.10
    except ValueError:
        print("❌ 权重格式不对")
        return

    bench = input("对标基准代码（如 000300=沪深300，^GSPC=标普500，默认000300）: ").strip() or "000300"

    print("\n货币：1. CNY(人民币)  2. USD(美元)  3. HKD(港币)")
    c = input("选货币 [1/2/3，默认1]: ").strip() or "1"
    currency = CURRENCIES.get(c, "CNY")

    rows = read_holdings()
    rows.append({
        "code": code, "name": name, "type": ftype,
        "weight": str(weight), "benchmark": bench, "currency": currency,
    })
    write_holdings(rows)

    # Show weight check
    total = sum(float(r["weight"]) for r in rows)
    if abs(total - 1.0) > 0.02:
        print(f"⚠️  当前总权重 {total:.1%}，建议调整为 100%。")
    cmd_sync()


def cmd_remove(code: str):
    """Remove a fund by code."""
    rows = read_holdings()
    before = len(rows)
    rows = [r for r in rows if r["code"] != code]
    if len(rows) == before:
        print(f"❌ 未找到代码 {code}")
        return
    write_holdings(rows)
    cmd_sync()


def cmd_sync():
    """Sync holdings.csv → config.yaml (funds section only)."""
    rows = read_holdings()
    if not rows:
        print("❌ holdings.csv 为空，请先添加基金")
        return

    # Read existing config.yaml
    if not CONFIG_PATH.exists():
        print("❌ config.yaml 不存在")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the funds: section and replace everything between funds: and the next top-level key
    funds_start = None
    funds_end = None
    in_funds = False
    indent_level = None

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped.startswith("funds:"):
            funds_start = i
            in_funds = True
            indent_level = len(line) - len(line.lstrip())
            continue
        if in_funds:
            # Check if this is a new top-level key (same indent as "funds:")
            if stripped and not stripped.startswith("#") and not stripped.startswith("-") and not stripped.startswith(" "):
                # Check indent
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and stripped:
                    funds_end = i
                    break
            # Also stop at next top-level key that starts at column 0
            if stripped and not line[0].isspace() and not stripped.startswith("-"):
                if i > funds_start + 1:
                    funds_end = i
                    break

    if funds_start is None:
        print("❌ config.yaml 中未找到 funds: 段")
        return

    if funds_end is None:
        funds_end = len(lines)

    # Build new funds section
    new_funds_lines = ["funds:\n"]
    for r in rows:
        new_funds_lines.append(f'  - code: "{r["code"]}"\n')
        new_funds_lines.append(f'    name: "{r["name"]}"\n')
        new_funds_lines.append(f'    type: "{r["type"]}"\n')
        new_funds_lines.append(f'    weight: {float(r["weight"]):.4f}\n')
        new_funds_lines.append(f'    benchmark: "{r["benchmark"]}"\n')
        new_funds_lines.append(f'    currency: "{r["currency"]}"\n')
    new_funds_lines.append("\n")

    # Replace
    new_lines = lines[:funds_start] + new_funds_lines + lines[funds_end:]
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    total = sum(float(r["weight"]) for r in rows)
    print(f"✅ 已将 {len(rows)} 只基金同步到 config.yaml（总权重 {total:.1%}）")


def cmd_check():
    """Validate holdings.csv."""
    rows = read_holdings()
    if not rows:
        print("❌ 持仓为空")
        return

    errors = []
    codes_seen = set()
    for i, r in enumerate(rows, 1):
        # Required fields
        for field in ["code", "name", "type", "weight", "currency"]:
            if not r.get(field, "").strip():
                errors.append(f"第{i}行: {field} 为空")
        # Weight
        try:
            w = float(r["weight"])
            if w <= 0 or w > 1:
                errors.append(f"第{i}行 ({r['code']}): 权重 {w} 不在 0~1 之间")
        except (ValueError, TypeError):
            errors.append(f"第{i}行 ({r['code']}): 权重格式错误")
        # Type
        if r["type"] not in ("a_share_mf", "a_share_etf", "overseas_etf"):
            errors.append(f"第{i}行 ({r['code']}): 未知类型 {r['type']}")
        # Duplicate code
        if r["code"] in codes_seen:
            errors.append(f"第{i}行 ({r['code']}): 重复代码")
        codes_seen.add(r["code"])

    total = sum(float(r["weight"]) for r in rows)
    if abs(total - 1.0) > 0.02:
        errors.append(f"总权重 {total:.1%}，不等于 100%")

    if errors:
        print("❌ 发现问题：")
        for e in errors:
            print(f"  • {e}")
    else:
        print(f"✅ 持仓验证通过：{len(rows)} 只基金，总权重 {total:.1%}")


# ── Menu ────────────────────────────────────────────────────────

def interactive_menu():
    """Display interactive menu."""
    while True:
        print("\n" + "=" * 40)
        print("  📊 基金持仓管理")
        print("=" * 40)
        print("  1. 查看持仓")
        print("  2. 添加基金")
        print("  3. 删除基金")
        print("  4. 同步到 config.yaml")
        print("  5. 验证持仓")
        print("  0. 退出")
        print("-" * 40)

        choice = input("请选择 [0-5]: ").strip()

        if choice == "1":
            cmd_list()
        elif choice == "2":
            cmd_add()
        elif choice == "3":
            code = input("要删除的基金代码: ").strip().upper()
            cmd_remove(code)
        elif choice == "4":
            cmd_sync()
        elif choice == "5":
            cmd_check()
        elif choice == "0":
            print("👋 再见")
            break
        else:
            print("❌ 无效选项")


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_menu()
    elif sys.argv[1] == "list":
        cmd_list()
    elif sys.argv[1] == "add":
        cmd_add()
    elif sys.argv[1] == "remove" and len(sys.argv) > 2:
        cmd_remove(sys.argv[2])
    elif sys.argv[1] == "sync":
        cmd_sync()
    elif sys.argv[1] == "check":
        cmd_check()
    else:
        print("用法: python manage.py [list|add|remove CODE|sync|check]")
        print("不带参数进入交互菜单")
