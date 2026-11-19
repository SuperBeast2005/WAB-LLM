from datasets import load_dataset
import pandas as pd

# 1. Konfiguration für die Replikompetenz (Seed)
SEED = 42
SAMPLES_PER_CATEGORY = 36

print("Lade MMLU-Pro Datensatz von Hugging Face...")
# Wir nutzen standardmäßig den 'test'-Split von MMLU-Pro
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
df = pd.DataFrame(dataset["test"])

# 2. Kategorien ermitteln
categories = df["category"].unique()
print(f"Anzahl gefundener Kategorien: {len(categories)}")

sampled_dfs = []

# 3. Stratifiziertes Sampling mit fixem Seed pro Kategorie
for cat in categories:
  cat_df = df[df["category"] == cat]

  # Sicherstellen, dass genug Daten vorhanden sind (MMLU-Pro hat pro Kategorie hunderte/tausende)
  n_sample = min(SAMPLES_PER_CATEGORY, len(cat_df))

  # Stichprobe ziehen mit festem random_state für Reproduzierbarkeit
  sampled_cat = cat_df.sample(n=n_sample, random_state=SEED)
  sampled_dfs.append(sampled_cat)

# 4. Zusammenfügen der Stichproben
final_df = pd.concat(sampled_dfs).reset_index(drop=True)

print(f"Erfolgreich ausgewählt: {len(final_df)} Aufgaben insgesamt.")
print(
    "Aufteilung pro Kategorie:",
    final_df["category"].value_counts().to_dict(),
)

# 5. Optional: Als JSONL oder CSV speichern
output_filename = f"mmlu_pro_sampled_{len(final_df)}_seed_{SEED}.jsonl"
final_df.to_json(output_filename, orient="records", lines=True, force_ascii=False)
print(f"Gespeichert unter: {output_filename}")