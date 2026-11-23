"""Tworzy ECDF_AlgorytmEwolucyjny.xlsx z pliku ecdf_data.json.

Uruchom po process_ecdf.py: python build_ecdf_python.py
Wymagana biblioteka: pip install openpyxl
"""

import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ecdf_data.json"
OUTPUT = ROOT / "outputs" / "ecdf_ae_20261123" / "ECDF_AlgorytmEwolucyjny.xlsx"


def populate(workbook, name, headings, values, percent=False, scientific=False):
    ws = workbook.create_sheet(name)
    ws.append(headings)
    for row in values:
        ws.append(row)
    ws.freeze_panes = "B2"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 25
    for j in range(2, len(headings) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 15
    ws.row_dimensions[1].height = 30
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="183153")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")
        if scientific and cell.column > 1 and isinstance(cell.value, (int, float)):
            cell.number_format = "0.00E+00"
    ws.auto_filter.ref = ws.dimensions
    for row in ws.iter_rows(min_row=2, min_col=2):
        for cell in row:
            if percent:
                cell.number_format = "0.0%"
            elif scientific:
                cell.number_format = "0.000000E+00"
    return ws


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {DATA}. Najpierw uruchom: python process_ecdf.py"
        )
    d = json.loads(DATA.read_text(encoding="utf-8"))
    wb = Workbook()
    wb.remove(wb.active)
    description = [
        ("Temat", "ECDF algorytmu ewolucyjnego – F1 i F2"),
        ("Reprezentacje", "Gray (binarna), real (rzeczywistoliczbowa)"),
        ("Liczba serii ECDF", 51),
        ("Liczba progów budżetu", 41),
        ("Eksperymenty na zestaw", "200 = 100 dla F1 + 100 dla F2"),
        ("F1 – wartość minimum", d["optima"]["F1"]),
        ("F2 – wartość minimum", d["optima"]["F2"]),
        ("Warunek sukcesu", "|F_best - F*| < próg jakości"),
        ("Wynik ECDF", "Liczba sukcesów / 200"),
        ("Źródło danych", "Wyniki_AlgorytmEwolucyjny.xlsx"),
        ("Zakres budżetu", "10–10000; dopasowano do istniejących danych"),
        ("Różnica względem PDF", "PDF wymaga 10–100000 dla n=10"),
    ]
    info = populate(wb, "START", ["Parametr", "Opis"], description)
    info.column_dimensions["B"].width = 76

    headings = ["Wywołania F", *d["thresholds"]]
    for rep in ("gray", "real"):
        ecdf = d["ecdf"][rep]["combined"]
        populate(
            wb,
            f"ECDF_{rep.upper()}",
            headings,
            [[budget, *[value / 200 for value in row]]
             for budget, row in zip(d["budget"], ecdf)],
            percent=True,
            scientific=False,
        )
        populate(
            wb,
            f"Liczby_{rep}",
            headings,
            [[budget, *row] for budget, row in zip(d["budget"], ecdf)],
        )
        for fn in ("F1", "F2"):
            samples = d["sampled"][f"{fn}_{rep}"]
            populate(
                wb,
                f"Surowe_{fn}_{rep}",
                ["Wywołania F", *[f"Seria {i}" for i in range(1, 101)]],
                [[budget, *row] for budget, row in zip(d["budget"], samples)],
                scientific=True,
            )
    populate(
        wb, "PROGI", ["Lp.", "Budżet F", "Próg jakości"],
        [[i + 1, d["budget"][i] if i < 41 else None, v]
         for i, v in enumerate(d["thresholds"])],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)
    print(f"Gotowe! Zapisano Excel: {OUTPUT}")


if __name__ == "__main__":
    main()
