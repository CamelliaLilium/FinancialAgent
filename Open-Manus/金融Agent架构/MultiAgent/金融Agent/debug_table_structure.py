"""分析表格结构"""
import sys
sys.path.insert(0, 'd:\\OpenManus\\金融Agent')

import re


def analyze_table_structure():
    """分析表格结构"""
    html_table = """}<s* scatter}y \) \( <r) + 1\)y ert)N+1

<table><tr><td>Statement of Income</td><td>October 29, 2011</td><td>October 30, 2010</td><td></td><td></td><td></td><td></td></tr><tr><td>Classification</td><td>Loss on Swaps</td><td>Gain on Note</td><td>Net Income Effect</td><td>Gain on Swaps</td><td>Loss on Note</td><td>Net Income Effect</td></tr><tr><td>Other income</td><td>$(4,614)</td><td>$4,614</td><td>$—</td><td>$20,692</td><td>$(20,692)</td><td>$—</td></tr></table>"""

    print("=" * 70)
    print("分析表格结构")
    print("=" * 70)

    tables = re.findall(r'<table.*?>.*?</table>', html_table, re.IGNORECASE | re.DOTALL)
    for table in tables:
        rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.IGNORECASE | re.DOTALL)

        print(f"\n表格有 {len(rows)} 行:")

        # 构建二维数组
        table_data = []
        for row_idx, row in enumerate(rows):
            cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
            cells = [re.sub(r'<.*?>', '', c).strip() for c in cells]
            table_data.append(cells)
            print(f"  行{row_idx}: {cells}")

        # 分析列结构
        print("\n列分析:")
        max_cols = max(len(row) for row in table_data)
        for col_idx in range(max_cols):
            col_values = []
            for row_idx, row in enumerate(table_data):
                if col_idx < len(row):
                    col_values.append(f"行{row_idx}:{row[col_idx]}")
                else:
                    col_values.append(f"行{row_idx}:N/A")
            print(f"  列{col_idx}: {' | '.join(col_values)}")

        # 理解表格逻辑
        print("\n表格逻辑理解:")
        print("这是一个复杂的二维表格，行1是分类，行2是实际数据")
        print("列1-3是2011年数据，列4-6是2010年数据")
        print("")
        print("要提取 'Gain on Swaps for October 29, 2011':")
        print("  - 找到 'October 29, 2011' 对应的列组 (列1-3)")
        print("  - 在这一组中找到 'Gain on Swaps' 对应的值")
        print("  - 但行1显示列2是 'Gain on Note'，列4才是 'Gain on Swaps'")
        print("  - 所以 'Gain on Swaps for 2011' 应该在列4，值是 $20,692")


if __name__ == "__main__":
    analyze_table_structure()
