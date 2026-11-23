import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from AlgorytmEwolucyjny import GeneticAlgorithm

NUMBER_OF_RUNS = 100
FUNCTION_CALLS = 10000
POPULATION_SIZE = 100
DIMENSIONS = 10
BITS_PER_DIMENSION = 16
OUTPUT_EXCEL = "Wyniki_AlgorytmEwolucyjny.xlsx"


# Funkcje podane w PDF (minimalizacja, n=10).
def function_1(x):
    s = sum(v * v for v in x)
    t = math.exp(-5.0 / (1.0 + s))
    return -5.0 / (1.0 + s) + math.sin(1.0 / math.tan(t))


def function_2(x):
    n = len(x)
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    s = sum(v * v for v in x) / n
    cos_mean = sum(math.cos(c * v) for v in x) / n
    return -a * math.exp(-b * math.sqrt(s)) - math.exp(cos_mean) + a + math.e


class EvaluationLimit(Exception):
    """Przerywa obliczenia dokładnie przy wymaganej liczbie ocen."""


def run_one(function, lower, upper, representation):
    ga = GeneticAlgorithm(
        dimensions=DIMENSIONS,
        bits_per_dimension=BITS_PER_DIMENSION,
        lower_bound=lower,
        upper_bound=upper,
        start_pop_size=POPULATION_SIZE,
        representation=representation,
    )
    history = []
    best = float("inf")
    # Pamiec ocen osobnikow: sortowanie tych samych rodzicow nie zuzywa budzetu.
    # Przechowujemy tez referencje do obiektow, aby id() nie zostalo ponownie uzyte.
    evaluation_cache = {}

    # Podmieniamy funkcję oceny tylko dla tego obiektu, bez edycji kodu
    # Liczymy faktyczne obliczenia funkcji; powtorne odczyty z cache nie sa ocenami.
    def tracked_objective(individual):
        nonlocal best
        key = id(individual)
        cached = evaluation_cache.get(key)
        if cached is not None and cached[0] is individual:
            return cached[1]

        x = ga.decode_solution(individual["x"]) if representation == "gray" else individual["x"]
        value = function(x)
        evaluation_cache[key] = (individual, value)
        if value < best:
            best = value
        history.append(best)  # najlepsza dotychczas, czyli seria nierosnąca
        if len(history) >= FUNCTION_CALLS:
            raise EvaluationLimit
        return value

    ga.objective_value = tracked_objective
    try:
        # Limit obliczen przerwie algorytm przed osiagnieciem n_iter.
        ga.run(n_iter=FUNCTION_CALLS, verbose=False)
    except EvaluationLimit:
        pass

    if len(history) != FUNCTION_CALLS:
        raise RuntimeError(f"Oczekiwano {FUNCTION_CALLS} ocen, uzyskano {len(history)}")
    return history


def add_excel_sheet(workbook, name, representation, runs):
    sheet_name = f"{name}_{representation}"
    sheet = workbook.create_sheet(sheet_name)
    # Tryb write_only: oszczędza pamięć przy ~4 mln komórek w całym pliku.
    sheet.freeze_panes = "B2"
    sheet.column_dimensions["A"].width = 17
    for col in range(2, NUMBER_OF_RUNS + 3):
        sheet.column_dimensions[get_column_letter(col)].width = 18

    from openpyxl.cell import WriteOnlyCell
    headers = ["Wywolanie"] + [f"Uruchomienie_{i + 1}" for i in range(len(runs))] + ["Srednia"]
    formatted = []
    for header in headers:
        cell = WriteOnlyCell(sheet, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="4472C4")
        cell.alignment = Alignment(horizontal="center")
        formatted.append(cell)
    sheet.append(formatted)

    averages = []
    for i in range(FUNCTION_CALLS):
        values = [run[i] for run in runs]
        avg = sum(values) / len(values)
        averages.append(avg)
        sheet.append([i + 1] + values + [avg])

    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{FUNCTION_CALLS + 1}"

    chart = LineChart()
    chart.title = f"Sredni przebieg algorytmu ewolucyjnego - {sheet_name}"
    chart.y_axis.title = "Srednia najlepsza wartosc funkcji"
    chart.x_axis.title = "Numer wywolania funkcji oceny"
    chart.height = 9
    chart.width = 20
    avg_col = len(runs) + 2
    chart.add_data(Reference(sheet, min_col=avg_col, min_row=1, max_row=FUNCTION_CALLS + 1),
                   titles_from_data=True)
    chart.set_categories(Reference(sheet, min_col=1, min_row=2, max_row=FUNCTION_CALLS + 1))
    chart.display_blanks = "gap"
    sheet.add_chart(chart, "CZ2")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, FUNCTION_CALLS + 1), averages)
    ax.set_xlabel("Numer wywolania funkcji oceny")
    ax.set_ylabel("Srednia najlepsza wartosc funkcji")
    ax.set_title(f"Algorytm ewolucyjny - {sheet_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"wykres_AE_{sheet_name}.png", dpi=150)
    plt.close(fig)
    print(f"Utworzono arkusz: {sheet_name}; srednia koncowa: {averages[-1]:.8f}", flush=True)


def main():
    tests = [
        ("F1", function_1, -3.0, 3.0),
        ("F2", function_2, -32.768, 32.768),
    ]
    workbook = Workbook(write_only=True)
    for name, function, lower, upper in tests:
        for representation in ("real", "gray"):
            print(f"\nEksperyment: {name}_{representation}", flush=True)
            runs = []
            for run_number in range(1, NUMBER_OF_RUNS + 1):
                runs.append(run_one(function, lower, upper, representation))
                if run_number % 10 == 0:
                    print(f"  Wykonano {run_number}/{NUMBER_OF_RUNS} uruchomien", flush=True)
            add_excel_sheet(workbook, name, representation, runs)
    workbook.save(OUTPUT_EXCEL)
    print(f"\nGOTOWE! Zapisano: {os.path.abspath(OUTPUT_EXCEL)}", flush=True)


if __name__ == "__main__":
    main()
