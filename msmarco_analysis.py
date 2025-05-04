import os
import json
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from collections import Counter
from statistics import mean

# Set seaborn theme
sns.set(style="whitegrid")

# Initialize wandb (optional, can be commented out for testing)
# wandb.init(project="msmarco-analysis", name="msmarco-v1.1-stats")

# Create output directory
os.makedirs("plots", exist_ok=True)

def plot_and_log(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=50, kde=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path = f"plots/{filename}"
    plt.savefig(path)
    plt.close()
    # Log image to wandb (optional, can be commented out for testing)
    # wandb.log({title: wandb.Image(path)})

def analyze_msmarco():
    print("[🚀] Starting MSMARCO analysis...")

    try:
        print("[↓] Loading MSMARCO v1.1 from Hugging Face...")
        dataset = load_dataset("ms_marco", "v1.1")
    except Exception as e:
        print(f"[❌] Failed to load dataset: {e}")
        return

    stats = {}

    for split_name in dataset:
        print(f"[📊] Processing split: {split_name}")
        split = dataset[split_name]

        # Query lengths
        query_lengths = [len(example['query'].split()) for example in split]
        avg_query_len = mean(query_lengths)
        stats[f"{split_name}_avg_query_len"] = avg_query_len

        # Passage lengths (handle structure difference)
        if 'passage' in split.column_names:
            passage_lengths = [len(example['passage']['passage_text'].split()) for example in split]
        elif 'passage_text' in split.column_names:
            passage_lengths = [len(example['passage_text'].split()) for example in split]
        else:
            print(f"[!] Warning: No passage field found in split {split_name}")
            passage_lengths = []

        avg_passage_len = mean(passage_lengths) if passage_lengths else 0
        stats[f"{split_name}_avg_passage_len"] = avg_passage_len

        # Relevance judgments
        if 'relevance' in split.column_names:
            relevance_counts = Counter(example['relevance'] for example in split)
        else:
            relevance_counts = Counter()
        stats[f"{split_name}_relevance_distribution"] = dict(relevance_counts)

        # Plot distributions
        if query_lengths:
            plot_and_log(query_lengths,
                         f"{split_name.capitalize()} Query Lengths",
                         "Query Length (tokens)",
                         "Frequency",
                         f"{split_name}_query_lengths.png")

        if passage_lengths:
            plot_and_log(passage_lengths,
                         f"{split_name.capitalize()} Passage Lengths",
                         "Passage Length (tokens)",
                         "Frequency",
                         f"{split_name}_passage_lengths.png")

        if relevance_counts:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=list(relevance_counts.keys()), y=list(relevance_counts.values()))
            plt.title(f"{split_name.capitalize()} Relevance Distribution")
            plt.xlabel("Relevance Label")
            plt.ylabel("Frequency")
            rel_path = f"plots/{split_name}_relevance_distribution.png"
            plt.tight_layout()
            plt.savefig(rel_path)
            plt.close()
            # Log image to wandb (optional, can be commented out for testing)
            # wandb.log({f"{split_name}_relevance_distribution": wandb.Image(rel_path)})

    # Log all stats to wandb (optional, can be commented out for testing)
    # wandb.log(stats)

    print("[✔] Analysis complete.")
    print("[🏁] Script execution finished.")

if __name__ == "__main__":
    try:
        analyze_msmarco()
    except Exception as e:
        print(f"[❌] An error occurred during script execution: {e}")
