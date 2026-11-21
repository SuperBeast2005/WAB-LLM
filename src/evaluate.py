"""Benchmark-Ergebnisse aus CSV-, JSON- und llama-server-Logdateien auswerten."""

import csv
import json
import math
import re
import statistics
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


# Datenordner neben dem Quellcode, damit das Skript von jedem Arbeitsverzeichnis läuft.
DATA_DIR = Path(__file__).resolve().parents[1] / "bench-data"
SUMMARY_JSON_PATH = Path(__file__).resolve().parents[1] / "evaluation_summary.json"

# Anzeigename jeder CSV-Metrik mit dem Anfang ihres Dateinamens.
CSV_METRICS = {
    "VRAM-Verbrauch (GB)": "VRAM-Usage in GB-data",
    "GPU-Last (%)": "GPU-Last in %-data",
    "GPU-Temperatur (°C)": "GPU-Temperatur °C-data",
    "RAM-Verbrauch (GB)": "RAM-Usage (in GB)-data",
    "CPU-Last (%)": "CPU-Auslastung in %-data",
    "Lüftergeschwindigkeit (RPM)": "Lüfter-Geschwindigkeit in RPM-data",
    "Total Tokens": "Total Tokens-data",
}

# ANSI-Farbcodes entfernen; das Log-Zeitformat ist Minuten.Sekunden.ms.us.
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
# Nur Zeilen mit abgeschlossener Decode-Zeit auswerten, nicht Prompt-Token-Zeilen.
LOG_TPS_PATTERN = re.compile(
    r"^(?P<minutes>\d+)\.(?P<seconds>\d{2})\.(?P<milliseconds>\d{3})"
    r"\.(?P<microseconds>\d{3}).*?\|\s+eval time\s*=.*?,\s*"
    r"(?P<tps>\d+(?:\.\d+)?)\s+tokens per second"
)

# Benchmark-Score-Schlüssel in den jeweiligen Ergebnis-JSONs.
SCORE_METRICS = {
    "HumanEval": "pass@1",
    "MATH-500": "exact_match",
    "MMLU-Pro-500": "exact_match",
}

DURATION_METRIC = "Ausführungsdauer/E2E (Stunden:Minuten)"
DURATION_JSON_METRIC = "Ausführungsdauer/E2E (Stunden)"


def canonical_model_name(directory_name: str) -> str:
    """Vereinheitlicht Modellordnernamen und fasst alle 9B-Varianten zusammen."""
    normalized_name = directory_name.lower().replace("_", "-")

    if normalized_name.startswith("qwen3.5-9b"):
        return "Qwen3.5-9B Q4_K_M"
    if normalized_name.startswith("qwen3.5-2b-q4-k-m"):
        return "Qwen3.5-2B-Q4_K_M"
    if normalized_name.startswith("qwen3.5-2b"):
        return "Qwen3.5-2B"
    return directory_name


def read_csv_values(model_dir: Path, metric_prefix: str) -> list[float]:
    """Liest die numerischen Messwerte aus der zum Präfix passenden CSV-Spalte."""
    files = sorted(
        path for path in model_dir.glob("*.csv")
        if path.name.startswith(metric_prefix)
    )
    if len(files) != 1:
        raise ValueError(
            f"Erwartete genau eine CSV-Datei für '{metric_prefix}' in "
            f"'{model_dir}', gefunden: {len(files)}."
        )

    values = []
    with files[0].open(encoding="utf-8-sig", newline="") as csv_file:
        for line_number, row in enumerate(csv.reader(csv_file), start=1):
            if line_number == 1:
                continue
            if len(row) < 2 or not row[1].strip():
                continue
            try:
                value = float(row[1])
            except ValueError as error:
                raise ValueError(
                    f"Ungültiger Messwert in '{files[0]}' "
                    f"(Zeile {line_number}): {row[1]!r}."
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"Nicht-endlicher Messwert in '{files[0]}' "
                    f"(Zeile {line_number}): {row[1]!r}."
                )
            values.append(value)

    if not values:
        raise ValueError(f"Keine Messwerte in '{files[0]}' gefunden.")
    return values


def read_csv_duration_seconds(model_dir: Path) -> float:
    """Berechnet die Laufzeit aus dem frühesten und spätesten CSV-Zeitstempel."""
    timestamps = []
    csv_files = sorted(model_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"Keine CSV-Dateien für Zeitstempel in '{model_dir}' gefunden.")

    for csv_path in csv_files:
        with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
            for line_number, row in enumerate(csv.reader(csv_file), start=1):
                if line_number == 1 or not row or not row[0].strip():
                    continue
                try:
                    timestamps.append(datetime.fromisoformat(row[0].strip()))
                except ValueError as error:
                    raise ValueError(
                        f"Ungültiger Zeitstempel in '{csv_path}' "
                        f"(Zeile {line_number}): {row[0]!r}."
                    ) from error

    if not timestamps:
        raise ValueError(f"Keine Zeitstempel in CSV-Dateien in '{model_dir}' gefunden.")
    return (max(timestamps) - min(timestamps)).total_seconds()


def format_duration(seconds: float) -> str:
    """Formatiert Sekunden als Stunden und Minuten, auf volle Minuten gerundet."""
    total_minutes = int(seconds / 60 + 0.5)
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"{minutes} Min."
    return f"{hours} Std. {minutes:02d} Min."


def relative_dispersion_percent(mean: float, standard_deviation: float) -> float:
    """Berechnet die relative Standardabweichung in Prozent."""
    if mean == 0:
        if standard_deviation == 0:
            return 0.0
        raise ValueError(
            "Die relative Streuung ist bei einem Mittelwert von null "
            "und positiver Standardabweichung nicht definiert."
        )
    return standard_deviation / abs(mean) * 100


def read_tps_log_values(
    model_dir: Path, execution_duration: float
) -> list[float] | None:
    """Liest Decode-TPS aus dem Serverlog bis zum Ende der CSV-Ausführungsdauer.

    Fehlt das Log, wird gewarnt und None zurückgegeben, damit andere Runs
    weiterhin ausgewertet werden können.
    """
    log_files = sorted(model_dir.glob("*.txt"))
    if not log_files:
        warnings.warn(
            f"Keine llama-server-Logdatei in '{model_dir}'; "
            "dieser Run wird für TPS ausgelassen.",
            stacklevel=2,
        )
        return None
    if len(log_files) != 1:
        raise ValueError(
            f"Erwartete genau eine llama-server-Logdatei in '{model_dir}', "
            f"gefunden: {len(log_files)}."
        )

    values = []
    with log_files[0].open(encoding="utf-8-sig") as log_file:
        for line in log_file:
            line = ANSI_ESCAPE_PATTERN.sub("", line)
            match = LOG_TPS_PATTERN.search(line)
            if not match:
                continue

            elapsed_seconds = (
                int(match.group("minutes")) * 60
                + int(match.group("seconds"))
                + int(match.group("milliseconds")) / 1000
                + int(match.group("microseconds")) / 1_000_000
            )
            if elapsed_seconds > execution_duration:
                continue

            tps = float(match.group("tps"))
            if not math.isfinite(tps):
                raise ValueError(
                    f"Ungültiger TPS-Wert in '{log_files[0]}': {match.group('tps')}."
                )
            values.append(tps)

    if not values:
        raise ValueError(
            f"Keine TPS-Logwerte innerhalb der Ausführungsdauer von "
            f"{execution_duration:.2f} Sekunden in '{log_files[0]}' gefunden."
        )
    return values


def read_benchmark_score(model_dir: Path, benchmark: str) -> float:
    """Liest den passenden Pass@1- oder Exact-Match-Score als Prozentwert."""
    result_files = sorted(model_dir.glob("results_*.json"))
    if len(result_files) != 1:
        raise ValueError(
            f"Erwartete genau eine Ergebnis-JSON in '{model_dir}', "
            f"gefunden: {len(result_files)}."
        )

    with result_files[0].open(encoding="utf-8-sig") as json_file:
        payload = json.load(json_file)

    score_prefix = SCORE_METRICS[benchmark]
    score = None
    for task_result in payload.get("results", {}).values():
        for key, value in task_result.items():
            if key == score_prefix or key.startswith(score_prefix + ","):
                score = float(value)
                break
        if score is not None:
            break

    if score is None or not math.isfinite(score):
        raise ValueError(
            f"Score '{score_prefix}' fehlt oder ist ungültig in "
            f"'{result_files[0]}'."
        )
    return score * 100


def evaluate_bench_data(
    data_dir: Path = DATA_DIR,
) -> dict[str, dict[str, dict[str, float | tuple[float, ...]]]]:
    """Sammelt Metriken nach Benchmark und Modell und fasst die Runs zusammen.

    Für CSV-Messwerte wird zuerst pro Run ein Mittelwert gebildet. TPS stammt
    aus den innerhalb der Laufzeit liegenden Log-Einträgen. Fehlende TPS-Logs
    werden ausgelassen; Spitzenwerte für Temperatur und Lüfter bleiben erhalten.
    """
    run_dirs = sorted(path for path in data_dir.glob("Run-*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"Keine Run-Ordner in '{data_dir}' gefunden.")

    results: dict[str, dict[str, dict[str, list[float]]]] = {}
    temperature_maxima: dict[str, dict[str, list[float]]] = {}
    fan_speed_maxima: dict[str, dict[str, list[float]]] = {}
    for benchmark in SCORE_METRICS:
        benchmark_results: dict[str, dict[str, list[float]]] = {}
        benchmark_temperature_maxima: dict[str, list[float]] = {}
        benchmark_fan_speed_maxima: dict[str, list[float]] = {}
        expected_models = None

        for run_dir in run_dirs:
            benchmark_dir = run_dir / benchmark
            if not benchmark_dir.is_dir():
                raise FileNotFoundError(
                    f"Benchmark-Ordner fehlt: '{benchmark_dir}'."
                )

            model_dirs = sorted(path for path in benchmark_dir.iterdir() if path.is_dir())
            run_models = {
                canonical_model_name(model_dir.name): model_dir
                for model_dir in model_dirs
            }
            if len(run_models) != len(model_dirs):
                raise ValueError(
                    f"Mehrere Modellordner werden in '{benchmark_dir}' "
                    "demselben Modell zugeordnet."
                )

            if expected_models is None:
                expected_models = set(run_models)
            elif set(run_models) != expected_models:
                raise ValueError(
                    f"Die Modellordner in '{benchmark_dir}' stimmen nicht "
                    f"mit den anderen Runs überein: {sorted(run_models)}."
                )

            for model_name, model_dir in run_models.items():
                score_key = "Pass@1 (%)" if benchmark == "HumanEval" else "Accuracy (%)"
                if model_name not in benchmark_results:
                    benchmark_results[model_name] = {score_key: []}
                    benchmark_results[model_name]["TPS"] = []
                    benchmark_results[model_name][DURATION_METRIC] = []
                    benchmark_results[model_name].update(
                        {metric: [] for metric in CSV_METRICS}
                    )
                    benchmark_temperature_maxima[model_name] = []
                    benchmark_fan_speed_maxima[model_name] = []
                model_metrics = benchmark_results[model_name]
                score = read_benchmark_score(model_dir, benchmark)
                model_metrics[score_key].append(score)
                duration = read_csv_duration_seconds(model_dir)
                model_metrics[DURATION_METRIC].append(duration)
                tps_values = read_tps_log_values(model_dir, duration)
                if tps_values is not None:
                    model_metrics["TPS"].append(statistics.mean(tps_values))

                for metric_name, filename_prefix in CSV_METRICS.items():
                    values = read_csv_values(model_dir, filename_prefix)
                    model_metrics[metric_name].append(statistics.mean(values))
                    if metric_name == "GPU-Temperatur (°C)":
                        benchmark_temperature_maxima[model_name].append(max(values))
                    elif metric_name == "Lüftergeschwindigkeit (RPM)":
                        benchmark_fan_speed_maxima[model_name].append(max(values))

        results[benchmark] = benchmark_results
        temperature_maxima[benchmark] = benchmark_temperature_maxima
        fan_speed_maxima[benchmark] = benchmark_fan_speed_maxima

    # Die Streuung beschreibt die Unterschiede zwischen den Run-Ergebnissen.
    summary: dict[str, dict[str, dict[str, float | tuple[float, ...]]]] = {}
    for benchmark, models in results.items():
        summary[benchmark] = {}
        for model_name, metrics in models.items():
            summary[benchmark][model_name] = {}
            for metric_name, run_values in metrics.items():
                if metric_name == "TPS":
                    if len(run_values) < 2 or len(run_values) > len(run_dirs):
                        raise ValueError(
                            f"Für {benchmark}/{model_name}/TPS wurden "
                            f"{len(run_values)} von {len(run_dirs)} Runs ausgewertet; "
                            "mindestens zwei TPS-Logs sind für die "
                            "Standardabweichung erforderlich."
                        )
                elif len(run_values) != len(run_dirs):
                    raise ValueError(
                        f"Für {benchmark}/{model_name}/{metric_name} wurden "
                        f"{len(run_values)} von {len(run_dirs)} Runs ausgewertet."
                    )
                mean = statistics.mean(run_values)
                standard_deviation = statistics.stdev(run_values)
                relative_dispersion = relative_dispersion_percent(
                    mean, standard_deviation
                )
                if metric_name == "GPU-Temperatur (°C)":
                    summary[benchmark][model_name][metric_name] = (
                        mean,
                        standard_deviation,
                        relative_dispersion,
                        max(temperature_maxima[benchmark][model_name]),
                    )
                elif metric_name == "Lüftergeschwindigkeit (RPM)":
                    summary[benchmark][model_name][metric_name] = (
                        mean,
                        standard_deviation,
                        relative_dispersion,
                        max(fan_speed_maxima[benchmark][model_name]),
                    )
                else:
                    summary[benchmark][model_name][metric_name] = (
                        mean,
                        standard_deviation,
                        relative_dispersion,
                    )
    return summary


def print_summary(
    summary: dict[str, dict[str, dict[str, float | tuple[float, ...]]]],
) -> None:
    """Gibt das Ergebnis-Dictionary mit zwei Nachkommastellen strukturiert aus."""
    print("{")
    for benchmark_index, (benchmark, models) in enumerate(summary.items()):
        print(f"  {benchmark!r}: {{")
        for model_index, (model, metrics) in enumerate(models.items()):
            print(f"    {model!r}: {{")
            for metric_index, (metric, value) in enumerate(metrics.items()):
                comma = "," if metric_index < len(metrics) - 1 else ""
                if (
                    metric == DURATION_METRIC
                    and isinstance(value, tuple)
                    and len(value) >= 3
                ):
                    formatted_value = (
                        f"({format_duration(value[0])} ± "
                        f"{format_duration(value[1])}; "
                        f"{value[2]:.2f}%)"
                    )
                    if len(value) == 4:
                        formatted_value += f" (Max.: {value[3]:.2f})"
                elif (
                    metric in ("Pass@1 (%)", "Accuracy (%)")
                    and isinstance(value, tuple)
                    and len(value) >= 3
                ):
                    formatted_value = (
                        f"({value[0]:.2f}% ± {value[1]:.2f}%; "
                        f"{value[2]:.2f}%)"
                    )
                elif isinstance(value, tuple) and len(value) >= 3:
                    formatted_value = (
                        f"({value[0]:.2f} ± {value[1]:.2f}; "
                        f"{value[2]:.2f}%)"
                    )
                    if len(value) == 4:
                        formatted_value += f" (Max.: {value[3]:.2f})"
                elif isinstance(value, tuple):
                    formatted_value = (
                        f"({', '.join(f'{item:.2f}' for item in value)})"
                    )
                elif metric in ("Pass@1 (%)", "Accuracy (%)"):
                    if isinstance(value, tuple):
                        formatted_value = (
                            f"({', '.join(f'{item:.2f}%' for item in value)})"
                        )
                    else:
                        formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
                print(f"      {metric!r}: {formatted_value}{comma}")
            comma = "," if model_index < len(models) - 1 else ""
            print(f"    }}{comma}")
        comma = "," if benchmark_index < len(summary) - 1 else ""
        print(f"  }}{comma}")
    print("}")


def export_summary_to_json(
    summary: dict[str, dict[str, dict[str, float | tuple[float, ...]]]],
    output_path: Path = SUMMARY_JSON_PATH,
) -> None:
    """Exportiert die Metriken als UTF-8-JSON mit Laufzeiten in Dezimalstunden."""
    def serialize_metric(
        metric_name: str, value: float | tuple[float, ...]
    ) -> dict[str, float]:
        if not isinstance(value, tuple):
            return {"Mean": value}
        if len(value) not in (2, 3, 4):
            raise ValueError(f"Nicht unterstützte Metrikdarstellung: {value!r}.")

        mean = value[0]
        deviation = value[1]
        if metric_name == DURATION_METRIC:
            mean /= 3600
            deviation /= 3600

        metric: dict[str, float] = {
            "Mean": mean,
            "Deviation": deviation,
        }
        if len(value) >= 3:
            metric["Relative Dispersion (%)"] = value[2]
        if len(value) == 4:
            metric["Max"] = value[3]
        return metric

    json_summary = {
        benchmark: {
            model: {
                DURATION_JSON_METRIC if metric == DURATION_METRIC else metric:
                    serialize_metric(metric, value)
                for metric, value in metrics.items()
            }
            for model, metrics in models.items()
        }
        for benchmark, models in summary.items()
    }
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(json_summary, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")


def format_metric_value(
    value: float | tuple[float, ...], metric_name: str | None = None
) -> str:
    """Formatiert einen Messwert für eine Matplotlib-LaTeX-Tabelle."""
    if metric_name in ("Pass@1 (%)", "Accuracy (%)"):
        if not isinstance(value, tuple):
            return rf"${value:.2f}\%$"
        if len(value) == 3:
            mean, standard_deviation, relative_dispersion = value
            return (
                rf"${mean:.2f}\% \pm {standard_deviation:.2f}\%$"
                rf"$\,({relative_dispersion:.2f}\%)$"
            )
    if not isinstance(value, tuple):
        return rf"${value:.2f}$"
    if len(value) == 2:
        mean, standard_deviation = value
        if metric_name == DURATION_METRIC:
            return (
                rf"$\mathrm{{{format_duration(mean)}}}"
                rf"\pm \mathrm{{{format_duration(standard_deviation)}}}$"
            )
        return rf"${mean:.2f} \pm {standard_deviation:.2f}$"
    if len(value) == 3:
        mean, standard_deviation, relative_dispersion = value
        if metric_name == DURATION_METRIC:
            return (
                rf"$\mathrm{{{format_duration(mean)}}}"
                rf"\pm \mathrm{{{format_duration(standard_deviation)}}}"
                rf"\quad({relative_dispersion:.2f}\%)$"
            )
        return (
            rf"${mean:.2f} \pm {standard_deviation:.2f}$"
            rf"$\,({relative_dispersion:.2f}\%)$"
        )
    if len(value) == 4:
        mean, standard_deviation, relative_dispersion, maximum = value
        return (
            rf"${mean:.2f} \pm {standard_deviation:.2f}$"
            rf"$\,({relative_dispersion:.2f}\%)$"
            rf"$\,(\mathrm{{Max.}}\,{maximum:.2f})$"
        )
    raise ValueError(f"Nicht unterstützte Metrikdarstellung: {value!r}.")


def plot_summary_as_latex_table(
    summary: dict[str, dict[str, dict[str, float | tuple[float, ...]]]],
) -> list[plt.Figure]:
    """Zeigt die berechneten Metriken je Benchmark in einem eigenen Fenster."""
    if not summary:
        raise ValueError("Keine Zusammenfassung zum Anzeigen vorhanden.")

    best_is_high = {
        "Pass@1 (%)": True,
        "Accuracy (%)": True,
        "TPS": True,
        DURATION_METRIC: False,
        "VRAM-Verbrauch (GB)": False,
        "GPU-Last (%)": False,
        "GPU-Temperatur (°C)": False,
        "RAM-Verbrauch (GB)": False,
        "CPU-Last (%)": False,
        "Lüftergeschwindigkeit (RPM)": False,
        "Total Tokens": False,
    }
    figures = []
    for benchmark, models in summary.items():
        figure, axis = plt.subplots(figsize=(14, 3.2))
        underline_cells = []
        model_order = list(models)
        model_2b = "Qwen3.5-2B"
        model_2b_q4km = "Qwen3.5-2B-Q4_K_M"
        if model_2b in model_order and model_2b_q4km in model_order:
            index_2b = model_order.index(model_2b)
            index_2b_q4km = model_order.index(model_2b_q4km)
            model_order[index_2b], model_order[index_2b_q4km] = (
                model_order[index_2b_q4km],
                model_order[index_2b],
            )
        ordered_models = [(model_name, models[model_name]) for model_name in model_order]
        available_metrics = list(next(iter(models.values())))
        desired_metric_names = [
            "Pass@1 (%)",
            "Accuracy (%)",
            "TPS",
            DURATION_METRIC,
            "VRAM-Verbrauch (GB)",
            "GPU-Last (%)",
            "GPU-Temperatur (°C)",
            "RAM-Verbrauch (GB)",
            "CPU-Last (%)",
            "Lüftergeschwindigkeit (RPM)",
            "Total Tokens",
        ]
        metric_names = [
            metric_name
            for metric_name in desired_metric_names
            if metric_name in available_metrics
        ]
        metric_names.extend(
            metric_name
            for metric_name in available_metrics
            if metric_name not in metric_names
        )
        best_model_indexes = {}
        table_rows = []

        for metric_name in metric_names:
            if metric_name in best_is_high:
                values = [
                    metrics[metric_name][0]
                    if isinstance(metrics[metric_name], tuple)
                    else metrics[metric_name]
                    for _, metrics in ordered_models
                ]
                best_value = (
                    max(values) if best_is_high[metric_name] else min(values)
                )
                best_model_indexes[metric_name] = {
                    index for index, value in enumerate(values)
                    if value == best_value
                }

            row = [metric_name]
            row.extend(
                format_metric_value(metrics[metric_name], metric_name)
                for _, metrics in ordered_models
            )
            table_rows.append(row)

        axis.axis("off")
        table = axis.table(
            cellText=table_rows,
            colLabels=["Metrik", *model_order],
            loc="center",
            cellLoc="center",
            colWidths=[0.28, *([0.72 / len(model_order)] * len(model_order))],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)

        for (row_index, column_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#D9E2F3")
                cell.set_text_props(weight="bold")
            elif column_index == 0:
                cell.set_facecolor("#F2F2F2")
                cell.set_text_props(weight="bold")
            elif column_index - 1 in best_model_indexes.get(
                metric_names[row_index - 1], set()
            ):
                cell.set_text_props(weight="bold")
                underline_cells.append((axis, cell))

        figure.suptitle(rf"{{{benchmark}}}", fontsize=14, y=1.0)
        figure.tight_layout()
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for underline_axis, cell in underline_cells:
            text_bounds = cell.get_text().get_window_extent(renderer)
            line_y = text_bounds.y0 - 1
            line_points = underline_axis.transAxes.inverted().transform(
                [(text_bounds.x0, line_y), (text_bounds.x1, line_y)]
            )
            underline_axis.plot(
                line_points[:, 0],
                line_points[:, 1],
                color=cell.get_text().get_color(),
                linewidth=1,
                transform=underline_axis.transAxes,
                clip_on=False,
            )
        figures.append(figure)
    return figures


if __name__ == "__main__":
    summary = evaluate_bench_data()
    print_summary(summary)
    export_summary_to_json(summary)
    print(f"JSON-Ergebnis gespeichert: {SUMMARY_JSON_PATH}")
    plot_summary_as_latex_table(summary)
    plt.show()