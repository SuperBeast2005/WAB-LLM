# WAB-LLM

Dieses Projekt vergleicht die Benchmark-Leistung und den Ressourcenverbrauch
verschiedener Qwen3.5-Modelle bei lokaler Inferenz mit `llama.cpp` und dessen
Vulkan-Backend.

## Untersuchte Modelle

- Qwen3.5 2B
- Qwen3.5 2B Q4_K_M
- Qwen3.5 9B Q4_K_M

Alle Modelle wurden im GGUF-Format mit einem Kontextfenster von 16.384 Tokens ausgeführt.
Die quantisierten Varianten wurden mit `llama-quantize` auf `Q4_K_M` quantisiert.

Die GGUF-Dateien in `gguf-models` haben folgende Größen:

| GGUF-Datei                  |  Größe | Größenreduktion |
| --------------------------- | -------: | --------------: |
| `qwen-3.5-2B-Q4_K_M.gguf` |  1,31 GB | 66,4 % |
| `qwen-3.5-2B.gguf`        |  3,90 GB | — |
| `qwen-3.5-9B-Q4_K_M.gguf` |  5,78 GB | 68,6 % |
| `qwen-3.5-9B.gguf`        | 18,41 GB | — |

Für den Betrieb des 9B-Modells mit Kontextlängen von 128k bis 256k Tokens wurde
`q4_0` wegen des begrenzten 8-GB-VRAMs verwendet. Für den Benchmark waren diese Größen des Kontextfenster nicht notwendig, jedoch wurde geschaut wie gut das 9B-Modell unter diesem KV-Cache performt.
Diese Langkontext-Konfiguration soll eine Alternative zu API-Modellen ermöglichen.
Die Modelle wurden mit derselben Vulkan-Inferenzumgebung getestet.

## Benchmarks

| Benchmark | Bewertungsmetrik        |
| --------- | ----------------------- |
| MATH-500  | Exact Match (Accuracy) |
| HumanEval | Pass@1                  |
| MMLU-Pro  | Exact Match (Accuracy)  |

## Erfasste Messwerte

Die Auswertung berücksichtigt neben den Benchmark-Scores folgende Messwerte:

- Total Tokens
- GPU-Auslastung (%)
- CPU-Auslastung (%)
- RAM-Verbrauch (GB)
- VRAM-Verbrauch (GB)
- GPU-Temperatur (°C)
- Lüftergeschwindigkeit (RPM)

Zusätzlich liest das Auswertungsskript Decode-Tokens pro Sekunde (TPS) aus den
`llama-server`-Logs und berechnet die Ausführungsdauer aus den CSV-Zeitstempeln.

## Hardware und Laufzeitumgebung

| Komponente      | Konfiguration                                      |
| --------------- | -------------------------------------------------- |
| CPU             | AMD Ryzen 5 3600 @ 4,2 GHz                         |
| GPU             | AMD Radeon RX 590 @ 1545 MHz, 8 GB VRAM @ 2100 MHz |
| Arbeitsspeicher | 16 GB DDR4 @ 3200 MHz                              |
| Betriebssystem  | Windows 10 Home 22H2                               |
| Inferenz        | `llama.cpp` mit Vulkan-Backend                   |

## Monitoring

System- und Inferenzmetriken wurden während der Läufe erfasst und gespeichert
mit:

- **windows_exporter** für Windows-Systemmetriken
- dem Metrik-Endpunkt des **llama.cpp-Servers** für Inferenzmetriken
- **Prometheus** zum Scrapen und Speichern der Metriken
- **Grafana** für Dashboards

Die exportierten Messwerte liegen als CSV-Dateien in den jeweiligen
Benchmark- und Modellordnern unter `bench-data`.

## Projektstruktur

```text
WAB-LLM/
├── bench-data/
│   ├── Run-0/
│   ├── Run-1/
│   └── Run-2/
├── MMLU-Pro/
│   ├── mmlu_pro_custom.yaml
│   └── mmlu_pro_sampled_504_seed_42.jsonl
├── src/
│   ├── evaluate.py
│   └── mmlu_extractor.py
├── gguf-models/
└── llama.cpp/
```

Jeder Lauf enthält separate Ordner für `MATH-500`, `HumanEval` und
`MMLU-Pro-500`; darin liegen die Messdaten und Benchmark-Ergebnisse je Modell.
`src/evaluate.py` sucht standardmäßig unter `bench-data` relativ zum
Projektverzeichnis.

## Auswertung

Voraussetzung für die Auswertung ist Python 3.10 oder neuer. Vom
Projektverzeichnis aus starten:

```powershell
python src\evaluate.py
```

Das Skript liest die Benchmark-Ergebnis-JSONs, CSV-Messwerte und
`llama-server`-Textlogs aus `bench-data` ein und gibt eine nach Benchmark und
Modell gruppierte Zusammenfassung aus. Bei CSV-Messwerten wird zunächst je
Lauf der Mittelwert berechnet; anschließend werden die Laufmittelwerte
zusammengefasst. Für die meisten Metriken wird dabei der Mittelwert und die
Stichproben-Standardabweichung über die Läufe ausgegeben. Temperatur und
Lüftergeschwindigkeit enthalten zusätzlich den höchsten erfassten Wert.

## MMLU-Pro-Sample erzeugen

`src/mmlu_extractor.py` lädt den `test`-Split von
`TIGER-Lab/MMLU-Pro`, zieht mit Seed `42` bis zu 36 Aufgaben je Kategorie und
speichert das Ergebnis als JSONL-Datei im aktuellen Arbeitsverzeichnis. Dazu
werden die Python-Pakete `datasets` und `pandas` benötigt.

## MMLU-Pro-Kategorieanalyse

```powershell
python src\analyze_mmlu_categories.py
```

Die Analyse schreibt fünf flache JSON-Dateien für Excel nach
`mmlu_chart_data`: eine Datei je Modell sowie Dateien für die
Aufgabenverteilung und den Anteil korrekter Targets. Die beiden
Kategorieverteilungen enthalten Anteile in Prozent; die Target-Datei enthält
zusätzlich die Standardabweichung zwischen den Modellen.
