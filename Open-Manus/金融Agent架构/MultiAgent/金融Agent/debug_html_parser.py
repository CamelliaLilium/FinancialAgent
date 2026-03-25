"""调试HTML表格解析器"""
import sys
sys.path.insert(0, 'd:\\OpenManus\\金融Agent')

import re
from app.skill.finance_extraction import _apply_unit_conversion, _row_matches_keywords, _parse_variable_semantics


def debug_html_table_parser():
    """调试HTML表格解析器"""
    html_table = """}<s* scatter}y \) \( <r) + 1\)y ert)N+1

<table><tr><td>Statement of Income</td><td>October 29, 2011</td><td>October 30, 2010</td><td></td><td></td><td></td><td></td></tr><tr><td>Classification</td><td>Loss on Swaps</td><td>Gain on Note</td><td>Net Income Effect</td><td>Gain on Swaps</td><td>Loss on Note</td><td>Net Income Effect</td></tr><tr><td>Other income</td><td>$(4,614)</td><td>$4,614</td><td>$—</td><td>$20,692</td><td>$(20,692)</td><td>$—</td></tr></table>"""

    row_keyword = "gain swaps"
    col_keyword = "october 29, 2011"

    print("=" * 70)
    print("调试HTML表格解析器")
    print("=" * 70)
    print(f"\n行关键词: '{row_keyword}'")
    print(f"列关键词: '{col_keyword}'")

    # 1. 提取所有表格
    tables = re.findall(r'<table.*?>.*?</table>', html_table, re.IGNORECASE | re.DOTALL)
    print(f"\n找到 {len(tables)} 个表格")

    for table_idx, table in enumerate(tables):
        print(f"\n--- 表格 {table_idx + 1} ---")

        # 2. 提取所有行
        rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.IGNORECASE | re.DOTALL)
        print(f"找到 {len(rows)} 行")

        headers = []
        col_index = -1

        # 准备列关键词
        col_keywords = [col_keyword.lower()]
        year_match = re.search(r'(20\d{2})', col_keyword)
        if year_match:
            col_keywords.append(year_match.group(1))
        print(f"列关键词列表: {col_keywords}")

        for row_idx, row in enumerate(rows):
            # 3. 提取单元格
            cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
            cells = [re.sub(r'<.*?>', '', c).strip() for c in cells]

            print(f"\n行 {row_idx}: {cells}")

            if not cells:
                continue

            # 4. 找表头
            if not headers and any(cells):
                headers = cells
                print(f"  表头: {headers}")

                for kw in col_keywords:
                    for i, h in enumerate(headers):
                        h_lower = h.lower()
                        print(f"    检查: kw='{kw}' vs header='{h}' -> h_lower='{h_lower}'")
                        if kw in h_lower or h_lower in kw:
                            col_index = i
                            print(f"    ✅ 列匹配成功: index={i}")
                            break
                    if col_index != -1:
                        break
                continue

            # 5. 找数据行
            if col_index == -1:
                print(f"  警告: 未找到目标列")
                continue

            if len(cells) <= col_index:
                print(f"  警告: 单元格数量不足 ({len(cells)} <= {col_index})")
                continue

            print(f"  检查行匹配: '{cells[0]}' vs '{row_keyword}'")
            if _row_matches_keywords(cells[0], row_keyword):
                val_str = cells[col_index]
                print(f"  ✅ 行匹配成功!")
                print(f"  值字符串: '{val_str}'")
                num = _apply_unit_conversion(val_str)
                print(f"  转换结果: {num}")
                if num is not None:
                    print(f"  ✅ 提取成功: {num}")
                    return num, val_str
            else:
                print(f"  ❌ 行不匹配")

    return None, ""


if __name__ == "__main__":
    debug_html_table_parser()
