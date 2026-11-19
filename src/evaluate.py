"""Benchmark-Ergebnisse aus CSV-, JSON- und llama-server-Logdateien auswerten."""

import csv
import json
import math
import re
import statistics
import warnings
from datetime import datetime
from pathlib import Path


# Datenordner neben dem Quellcode, damit das Skript von jedem Arbeitsverzeichnis läuft.
DATA_DIR = Path(__file__).resolve().parents[1] / "bench-data"

# Anzeigename jeder CSV-Metrik mit dem Anfang ihres Dateinamens.
CSV_METRICS = {
    "Total Tokens": "Total Tokens-data",
    "GPU-Last (%)": "GPU-Last in %-data",
    "CPU-Last (%)": "CPU-Auslastung in %-data",
    "RAM-Verbrauch (GB)": "RAM-Usage (in GB)-data",
    "VRAM-Verbrauch (GB)": "VRAM-Usage in GB-data",
    "GPU-Temperatur (°C)": "GPU-Temperatur °C-data",
    "Lüftergeschwindigkeit (RPM)": "Lüfter-Geschwindigkeit in RPM-data",
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


def canonical_model_name(directory_name: str) -> str:
    """Vereinheitlicht Modellordnernamen und fasst alle 9B-Varianten zusammen."""
    normalized_name = directory_name.lower().replace("_", "-")

    if normalized_name.startswith("qwen3.5-9b"):
        return "Qwen3.5-9B"
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
    """Liest den passenden Pass@1- oder Exact-Match-Score aus der Ergebnis-JSON."""
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
    return score


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
                score_key = "Pass@1" if benchmark == "HumanEval" else "Exact Match"
                if model_name not in benchmark_results:
                    benchmark_results[model_name] = {score_key: []}
                    benchmark_results[model_name]["TPS"] = []
                    benchmark_results[model_name]["Ausführungsdauer (Sekunden)"] = []
                    benchmark_results[model_name].update(
                        {metric: [] for metric in CSV_METRICS}
                    )
                    benchmark_temperature_maxima[model_name] = []
                    benchmark_fan_speed_maxima[model_name] = []
                model_metrics = benchmark_results[model_name]
                score = read_benchmark_score(model_dir, benchmark)
                model_metrics[score_key].append(score)
                duration = read_csv_duration_seconds(model_dir)
                model_metrics["Ausführungsdauer (Sekunden)"].append(duration)
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
                if metric_name == "Total Tokens":
                    summary[benchmark][model_name][metric_name] = mean
                elif metric_name == "GPU-Temperatur (°C)":
                    summary[benchmark][model_name][metric_name] = (
                        mean,
                        statistics.stdev(run_values),
                        max(temperature_maxima[benchmark][model_name]),
                    )
                elif metric_name == "Lüftergeschwindigkeit (RPM)":
                    summary[benchmark][model_name][metric_name] = (
                        mean,
                        statistics.stdev(run_values),
                        max(fan_speed_maxima[benchmark][model_name]),
                    )
                else:
                    summary[benchmark][model_name][metric_name] = (
                        mean,
                        statistics.stdev(run_values),
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
                if isinstance(value, tuple):
                    formatted_value = (
                        f"({', '.join(f'{item:.2f}' for item in value)})"
                    )
                else:
                    formatted_value = f"{value:.2f}"
                print(f"      {metric!r}: {formatted_value}{comma}")
            comma = "," if model_index < len(models) - 1 else ""
            print(f"    }}{comma}")
        comma = "," if benchmark_index < len(summary) - 1 else ""
        print(f"  }}{comma}")
    print("}")


if __name__ == "__main__":
    print_summary(evaluate_bench_data())