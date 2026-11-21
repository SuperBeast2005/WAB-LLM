"""Zeigt Kategorie-Anteile und Exact-Match-Raten aller MMLU-Pro-Runs.

Aufruf vom Repository-Root aus: python src/analyze_mmlu_categories.py
"""

import json
import math
import statistics
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DATA_DIR = Path(__file__).resolve().parents[1] / "bench-data"
BENCHMARK_DIR_NAME = "MMLU-Pro-500"
SAMPLE_FILE_PREFIX = "samples_mmlu_pro_custom"
OUTPUT_JSON_PATH = Path(__file__).resolve().parents[1] / "mmlu_pro_500_category_analysis.json"
CHART_DATA_DIR = Path(__file__).resolve().parents[1] / "mmlu_chart_data"
CATEGORY_COLORS = {
    "biology": "#2ca02c",
    "business": "#a66a45",
    "chemistry": "#78b87a",
    "computer science": "#9a9a9a",
    "economics": "#4e79a7",
    "engineering": "#666666",
    "health": "#59a14f",
    "history": "#e6ab02",
    "law": "#8c564b",
    "math": "#1f77b4",
    "other": "#9467bd",
    "philosophy": "#ffc000",
    "physics": "#76b7e5",
    "psychology": "#5b3427",
}


def read_samples(sample_file: Path) -> list[dict[str, object]]:
    """Liest JSONL-Samples oder ein JSON-Objekt/eine JSON-Liste."""
    samples = []
    with sample_file.open(encoding="utf-8-sig") as file:
        if sample_file.suffix == ".jsonl":
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Ungültiges JSON in '{sample_file}' "
                        f"(Zeile {line_number})."
                    ) from error
                if not isinstance(sample, dict):
                    raise ValueError(
                        f"Erwartet wurde ein JSON-Objekt in '{sample_file}' "
                        f"(Zeile {line_number})."
                    )
                samples.append(sample)
        else:
            try:
                payload = json.load(file)
            except json.JSONDecodeError as error:
                raise ValueError(f"Ungültiges JSON in '{sample_file}'.") from error
            records = payload if isinstance(payload, list) else [payload]
            if not all(isinstance(sample, dict) for sample in records):
                raise ValueError(
                    f"Erwartet wurden JSON-Objekte in '{sample_file}'."
                )
            samples.extend(records)

    if not samples:
        raise ValueError(f"Keine Samples in '{sample_file}' gefunden.")
    return samples


def summarize_samples(
    samples: list[dict[str, object]], sample_file: Path
) -> dict[str, list[int]]:
    """Sammelt je Kategorie die exakten Treffer pro Aufgabe (0 oder 1)."""
    categories: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(samples, start=1):
        document = sample.get("doc")
        if not isinstance(document, dict):
            raise ValueError(
                f"Das Feld 'doc' fehlt oder ist ungültig in '{sample_file}' "
                f"(Sample {sample_index})."
            )
        category = document.get("category")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(
                f"Die Kategorie fehlt oder ist ungültig in '{sample_file}' "
                f"(Sample {sample_index})."
            )

        exact_match = sample.get("exact_match")
        if (
            not isinstance(exact_match, (int, float))
            or not math.isfinite(exact_match)
            or exact_match not in (0, 1)
        ):
            raise ValueError(
                f"'exact_match' muss 0 oder 1 sein in '{sample_file}' "
                f"(Sample {sample_index})."
            )

        categories.setdefault(category, []).append(int(exact_match))
    return categories


def calculate_statistics(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("Für die Deviation werden mindestens zwei Runs benötigt.")

    mean = statistics.mean(values)
    deviation = statistics.stdev(values)
    if mean == 0:
        if deviation == 0:
            relative_dispersion = 0.0
        else:
            raise ValueError(
                "Die relative Streuung ist bei einem Mittelwert von null "
                "und positiver Deviation nicht definiert."
            )
    else:
        relative_dispersion = deviation / abs(mean) * 100
    return mean, deviation, relative_dispersion


def summarize_model_runs(
    model_name: str,
    runs: list[tuple[str, int, dict[str, list[int]]]],
) -> tuple[int, dict[str, dict[str, int | float]]]:
    if len(runs) < 2:
        raise ValueError(
            f"Für {model_name} wurden weniger als zwei Runs gefunden."
        )

    category_names = set(runs[0][2])
    for run_name, _, categories in runs[1:]:
        if set(categories) != category_names:
            raise ValueError(
                f"Die Kategorien in {run_name}/{model_name} unterscheiden "
                "sich von den anderen Runs."
            )

    total_count = sum(total_samples for _, total_samples, _ in runs)
    category_summary: dict[str, dict[str, int | float]] = {}
    for category in sorted(category_names):
        run_outcomes = [
            categories[category]
            for _, _, categories in runs
        ]
        category_count = sum(len(outcomes) for outcomes in run_outcomes)
        run_rates = [
            sum(outcomes) / len(outcomes) * 100
            for outcomes in run_outcomes
        ]
        target_mean, target_deviation, target_relative = calculate_statistics(
            run_rates
        )
        category_summary[category] = {
            "task_count": category_count,
            "task_share_percent": category_count / total_count * 100,
            "target_reached_count": sum(
                sum(outcomes) for outcomes in run_outcomes
            ),
            "target_mean_percent": target_mean,
            "target_deviation_percent": target_deviation,
            "target_relative_dispersion_percent": target_relative,
        }
    return total_count, category_summary


def print_model_report(
    model_name: str,
    categories: dict[str, dict[str, int | float]],
) -> None:
    print(f"\n{model_name}")
    print(
        f"{'Kategorie':<20} {'Anteil (%)':>12} {'Run Mean (%)':>17} "
        f"{'Run-SD (%)':>17} {'Relative Streuung (%)':>24}"
    )
    for category, values in categories.items():
        print(
            f"{category:<20} {values['task_share_percent']:>12.2f} "
            f"{values['target_mean_percent']:>17.2f} "
            f"{values['target_deviation_percent']:>17.2f} "
            f"{values['target_relative_dispersion_percent']:>24.2f}"
        )


def write_excel_chart_data(
    results_by_model: dict[str, list[tuple[str, int, dict[str, list[int]]]]],
) -> list[Path]:
    model_names = sorted(results_by_model)
    if len(model_names) != 3:
        raise ValueError(
            f"Für die Excel-Diagrammdaten werden genau drei Modelle erwartet, "
            f"gefunden: {len(model_names)}."
        )

    categories = sorted(
        next(iter(results_by_model.values()))[0][2],
        key=category_color_order,
    )
    task_counts = {category: 0 for category in categories}
    correct_target_counts = {category: 0 for category in categories}
    model_correct_target_counts = {
        model_name: {category: 0 for category in categories}
        for model_name in model_names
    }

    for model_name in model_names:
        for _, _, run_categories in results_by_model[model_name]:
            if set(run_categories) != set(categories):
                raise ValueError(
                    f"Die Kategorien in einem Run von '{model_name}' "
                    "stimmen nicht mit den anderen Ergebnissen überein."
                )
            for category in categories:
                task_counts[category] += len(run_categories[category])
                correct_count = sum(run_categories[category])
                correct_target_counts[category] += correct_count
                model_correct_target_counts[model_name][category] += correct_count

    total_tasks = sum(task_counts.values())
    total_correct_targets = sum(correct_target_counts.values())
    if total_tasks == 0:
        raise ValueError("Keine Aufgaben für die Excel-Diagrammdaten vorhanden.")
    if total_correct_targets == 0:
        raise ValueError(
            "Keine korrekten Targets für die Excel-Diagrammdaten vorhanden."
        )

    CHART_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written_files = []

    def write_rows(filename: str, rows: list[dict[str, str | int | float]]) -> None:
        output_path = CHART_DATA_DIR / filename
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(rows, json_file, ensure_ascii=False, indent=2)
            json_file.write("\n")
        written_files.append(output_path)

    write_rows(
        "category_distribution.json",
        [
            {
                "category": category,
                "share_percent": task_counts[category] / total_tasks * 100,
            }
            for category in categories
        ],
    )

    model_totals = {
        model_name: sum(model_correct_target_counts[model_name].values())
        for model_name in model_names
    }
    if any(total == 0 for total in model_totals.values()):
        raise ValueError(
            "Mindestens ein Modell hat keine korrekten Targets für "
            "die Excel-Diagrammdaten."
        )
    correct_target_rows = []
    for category in categories:
        model_shares = [
            model_correct_target_counts[model_name][category]
            / model_totals[model_name]
            * 100
            for model_name in model_names
        ]
        correct_target_rows.append(
            {
                "category": category,
                "correct_target_count": correct_target_counts[category],
                "share_percent": (
                    correct_target_counts[category] / total_correct_targets * 100
                ),
                "model_share_standard_deviation_percent": statistics.stdev(
                    model_shares
                ),
            }
        )
    write_rows("correct_target_share.json", correct_target_rows)

    for model_name in model_names:
        _, summary = summarize_model_runs(model_name, results_by_model[model_name])
        filename_model_name = (
            model_name.lower().replace(".", "_").replace("-", "_").replace(" ", "_")
        )
        write_rows(
            f"model_{filename_model_name}.json",
            [
                {
                    "model": model_name,
                    "category": category,
                    "target_mean_percent": summary[category]["target_mean_percent"],
                    "target_standard_deviation_percent": summary[category][
                        "target_deviation_percent"
                    ],
                }
                for category in categories
            ],
        )

    return written_files


def category_color_order(category: str) -> tuple[bool, bool, float, str]:
    red, green, blue = (
        int(CATEGORY_COLORS[category][index:index + 2], 16) / 255
        for index in (1, 3, 5)
    )
    hue, saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
    return category == "other", saturation == 0, hue, category


def plot_pie_charts(
    results_by_model: dict[str, list[tuple[str, int, dict[str, list[int]]]]],
    run_names: list[str],
) -> None:
    if not results_by_model or not run_names:
        raise ValueError("Keine Ergebnisse für die Pie-Charts vorhanden.")

    model_names = sorted(results_by_model)
    sample_categories = set(next(iter(results_by_model.values()))[0][2])
    missing_colors = sample_categories - CATEGORY_COLORS.keys()
    if missing_colors:
        raise ValueError(
            f"Für diese MMLU-Kategorien fehlen Pie-Chart-Farben: "
            f"{sorted(missing_colors)}."
        )
    if len(model_names) != 3:
        raise ValueError(
            f"Für das 4-Chart-Fenster werden genau drei Modelle erwartet, "
            f"gefunden: {len(model_names)}."
        )
    categories = sorted(sample_categories, key=category_color_order)
    figure = plt.figure(figsize=(19, 12))
    layout = figure.add_gridspec(
        2,
        6,
        left=0.04,
        right=0.82,
        bottom=0.12,
        top=0.88,
        wspace=0.12,
        hspace=0.32,
    )
    distribution_axis = figure.add_subplot(layout[0, 1:3])
    correct_targets_axis = figure.add_subplot(layout[0, 3:5])
    model_axes = [
        figure.add_subplot(layout[1, column * 2:(column + 1) * 2])
        for column in range(3)
    ]

    for model_index, model_name in enumerate(model_names):
        runs = results_by_model[model_name]
        if {run_name for run_name, _, _ in runs} != set(run_names):
            raise ValueError(
                f"Die Runs für Modell '{model_name}' stimmen nicht überein."
            )
        _, summary = summarize_model_runs(model_name, runs)
        category_values = [
            (
                category,
                summary[category]["target_mean_percent"],
                summary[category]["target_deviation_percent"],
            )
            for category in categories
        ]
        nonzero_values = [
            values for values in category_values if values[1] > 0
        ]
        if not nonzero_values:
            raise ValueError(f"Keine Target-Treffer für Modell '{model_name}'.")

        axis = model_axes[model_index]
        _, _, autotexts = axis.pie(
            [mean for _, mean, _ in nonzero_values],
            colors=[CATEGORY_COLORS[category] for category, _, _ in nonzero_values],
            startangle=90,
            counterclock=False,
            autopct="%1.1f%%",
            pctdistance=0.78,
            radius=0.82,
            textprops={"fontsize": 6},
            wedgeprops={"edgecolor": "white", "linewidth": 0.6},
        )
        for text, (_, mean, deviation) in zip(autotexts, nonzero_values):
            text.set_text(f"{mean:.1f}%\n±{deviation:.1f}%")
        axis.set_title(model_name)
        axis.axis("equal")

    task_counts = {category: 0 for category in categories}
    total_tasks = 0
    solved_task_counts = {category: 0 for category in categories}
    total_solved_tasks = 0
    model_solved_counts = {
        model_name: {category: 0 for category in categories}
        for model_name in model_names
    }
    model_total_solved = {model_name: 0 for model_name in model_names}
    for model_name in model_names:
        for _, total_samples, run_categories in results_by_model[model_name]:
            if set(run_categories) != set(categories):
                raise ValueError(
                    f"Die Kategorien in einem Run von '{model_name}' "
                    "stimmen nicht mit den anderen Ergebnissen überein."
                )
            total_tasks += total_samples
            for category in categories:
                task_counts[category] += len(run_categories[category])
                solved_count = sum(run_categories[category])
                solved_task_counts[category] += solved_count
                total_solved_tasks += solved_count
                model_solved_counts[model_name][category] += solved_count
                model_total_solved[model_name] += solved_count

    if total_tasks == 0:
        raise ValueError("Keine Aufgaben für die Verteilung vorhanden.")
    if total_solved_tasks == 0:
        raise ValueError("Keine korrekten Targets für die Verteilung vorhanden.")

    distribution_values = [
        (category, count / total_tasks * 100)
        for category, count in task_counts.items()
        if count > 0
    ]
    _, _, autotexts = distribution_axis.pie(
        [share for _, share in distribution_values],
        colors=[CATEGORY_COLORS[category] for category, _ in distribution_values],
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%",
        pctdistance=0.78,
        radius=0.9,
        textprops={"fontsize": 6},
        wedgeprops={"edgecolor": "white", "linewidth": 0.6},
    )
    for text, (_, share) in zip(autotexts, distribution_values):
        text.set_text(f"{share:.1f}%")
    distribution_axis.set_title("Aufgabenverteilung nach Kategorie")
    distribution_axis.axis("equal")

    correct_target_values = []
    for category in categories:
        count = solved_task_counts[category]
        if count > 0:
            correct_target_values.append(
                (category, count / total_solved_tasks * 100)
            )
    correct_target_deviations = {
        category: statistics.stdev(
            model_solved_counts[model_name][category]
            / model_total_solved[model_name]
            * 100
            for model_name in model_names
        )
        for category in categories
    }
    _, _, autotexts = correct_targets_axis.pie(
        [share for _, share in correct_target_values],
        colors=[
            CATEGORY_COLORS[category] for category, _ in correct_target_values
        ],
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%",
        pctdistance=0.78,
        radius=0.9,
        textprops={"fontsize": 6},
        wedgeprops={"edgecolor": "white", "linewidth": 0.6},
    )
    for text, (category, share) in zip(autotexts, correct_target_values):
        text.set_text(f"{share:.1f}%\n±{correct_target_deviations[category]:.1f}%")
    correct_targets_axis.set_title(
        "Anteil korrekter Targets nach Kategorie\nzwischen den Modellen"
    )
    correct_targets_axis.axis("equal")

    figure.suptitle(
        "MMLU-Pro-500 – Trefferquoten und Kategorieanteile",
        fontsize=14,
    )
    figure.text(
        0.43,
        0.045,
        "Modellcharts: Trefferquote je Kategorie (Mittelwert ± Standardabweichung in Prozent). "
        "Verteilung: Anteil an allen Aufgaben. Korrekte Targets: Anteil ± "
        "Standardabweichung zwischen den drei Modellen.",
        ha="center",
        fontsize=9,
    )
    figure.legend(
        [
            Patch(facecolor=CATEGORY_COLORS[category], label=category)
            for category in categories
        ],
        categories,
        title="Kategorie",
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        ncol=1,
        fontsize=8,
        frameon=False,
    )


def main(data_dir: Path = DATA_DIR) -> None:
    run_dirs = sorted(path for path in data_dir.glob("Run-*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"Keine Run-Ordner in '{data_dir}' gefunden.")

    results_by_model: dict[str, list[tuple[str, int, dict[str, list[int]]]]] = {}
    expected_models = None
    for run_dir in run_dirs:
        benchmark_dir = run_dir / BENCHMARK_DIR_NAME
        if not benchmark_dir.is_dir():
            raise FileNotFoundError(
                f"Benchmark-Ordner fehlt: '{benchmark_dir}'."
            )

        model_dirs = sorted(
            path for path in benchmark_dir.iterdir() if path.is_dir()
        )
        if not model_dirs:
            raise FileNotFoundError(
                f"Keine Modellordner in '{benchmark_dir}' gefunden."
            )

        run_models = {model_dir.name: model_dir for model_dir in model_dirs}
        if expected_models is None:
            expected_models = set(run_models)
        elif set(run_models) != expected_models:
            raise ValueError(
                f"Die Modelle in '{benchmark_dir}' unterscheiden sich "
                "zwischen den Runs."
            )

        for model_dir in model_dirs:
            sample_files = sorted(
                [
                    *model_dir.glob(f"{SAMPLE_FILE_PREFIX}*.json"),
                    *model_dir.glob(f"{SAMPLE_FILE_PREFIX}*.jsonl"),
                ]
            )
            if len(sample_files) != 1:
                raise ValueError(
                    f"Erwartete genau eine Sample-Datei in '{model_dir}', "
                    f"gefunden: {len(sample_files)}."
                )
            samples = read_samples(sample_files[0])
            categories = summarize_samples(samples, sample_files[0])
            results_by_model.setdefault(model_dir.name, []).append(
                (run_dir.name, len(samples), categories)
            )

    model_results = {}
    for model_name, runs in sorted(results_by_model.items()):
        total_tasks, categories = summarize_model_runs(model_name, runs)
        print_model_report(model_name, categories)
        model_results[model_name] = {
            "run_count": len(runs),
            "total_tasks": total_tasks,
            "categories": categories,
        }

    report = {
        "benchmark": BENCHMARK_DIR_NAME,
        "runs": [run_dir.name for run_dir in run_dirs],
        "models": model_results,
    }
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")
    print(f"\nJSON-Ergebnisse gespeichert: {OUTPUT_JSON_PATH}")
    chart_data_files = write_excel_chart_data(results_by_model)
    print("Excel-Diagrammdaten gespeichert:")
    for chart_data_file in chart_data_files:
        print(f"  {chart_data_file}")
    plot_pie_charts(
        results_by_model,
        [run_dir.name for run_dir in run_dirs],
    )
    plt.show()


if __name__ == "__main__":
    main()
