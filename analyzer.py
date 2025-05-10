import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from math import log
import wandb


class MSMARCOAnalyzer:
    def __init__(self, docs_df, queries_df, triples_df):
        self.docs_df = docs_df.rename(columns={"doc_id": "doc_id", "text": "doc_text"})
        self.queries_df = queries_df.rename(columns={"query_id": "query_id", "text": "query_text"})
        self.triples_df = triples_df.copy()

        # Tokenization
        self.docs_df["tokens"] = self.docs_df["doc_text"].str.split()
        self.queries_df["tokens"] = self.queries_df["query_text"].str.split()

        # Create quick lookup maps
        self.doc_map = dict(zip(self.docs_df["doc_id"], self.docs_df["tokens"]))
        self.query_map = dict(zip(self.queries_df["query_id"], self.queries_df["tokens"]))

    def analyze_triplet_counts(self):
        counts = self.triples_df["query_id"].value_counts()
        print(f"[📊] Triplet count statistics per query:\n"
              f"  Mean: {counts.mean():.2f}\n"
              f"  Median: {counts.median():.2f}\n"
              f"  Max: {counts.max()}\n"
              f"  Min: {counts.min()}")

    def plot_triplets_per_query(self, bins: int = 50):
        counts = self.triples_df["query_id"].value_counts()
        plt.figure(figsize=(8, 5))
        sns.histplot(counts, bins=bins, kde=False)
        plt.xlabel("Triplets per Query")
        plt.ylabel("Count")
        plt.title("Distribution of Triplets per Query")
        plt.tight_layout()
        plt.savefig("triplets_per_query.png")
        wandb.log({"Triplets per Query Histogram": wandb.Image("triplets_per_query.png")})
        plt.close()

    def plot_positive_vs_negative_overlap(self):
        grouped = self.triples_df.groupby("query_id")
        overlap_data = {
            "query_id": [],
            "n_pos": [],
            "n_neg": [],
        }

        for qid, group in grouped:
            pos = set(group["doc_id_a"])
            neg = set(group["doc_id_b"])
            overlap_data["query_id"].append(qid)
            overlap_data["n_pos"].append(len(pos))
            overlap_data["n_neg"].append(len(neg))

        df = pd.DataFrame(overlap_data)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x="n_pos", y="n_neg", alpha=0.5)
        plt.xlabel("Positive Docs")
        plt.ylabel("Negative Docs")
        plt.title("Positive vs Negative Docs per Query")
        plt.tight_layout()
        plt.savefig("positive_vs_negative_overlap.png")
        wandb.log({"Positive vs Negative Scatter": wandb.Image("positive_vs_negative_overlap.png")})
        plt.close()

    def token_stats(self):
        all_doc_tokens = [t for tokens in self.docs_df["tokens"] for t in tokens]
        all_query_tokens = [t for tokens in self.queries_df["tokens"] for t in tokens]

        doc_len = [len(t) for t in self.docs_df["tokens"]]
        query_len = [len(t) for t in self.queries_df["tokens"]]

        print(f"[📚] Document tokens: total={len(all_doc_tokens):,}, unique={len(set(all_doc_tokens)):,}, avg_len={np.mean(doc_len):.2f}")
        print(f"[🔍] Query tokens: total={len(all_query_tokens):,}, unique={len(set(all_query_tokens)):,}, avg_len={np.mean(query_len):.2f}")

    def top_token_freq(self, k=20):
        all_doc_tokens = [t for tokens in self.docs_df["tokens"] for t in tokens]
        all_query_tokens = [t for tokens in self.queries_df["tokens"] for t in tokens]

        doc_counter = Counter(all_doc_tokens)
        query_counter = Counter(all_query_tokens)

        top_docs = doc_counter.most_common(k)
        top_queries = query_counter.most_common(k)

        print(f"[📝] Top-{k} document tokens:\n", top_docs)
        print(f"[❓] Top-{k} query tokens:\n", top_queries)

        return top_docs, top_queries

    def idf_stats(self, tokens):
        N = len(self.docs_df)
        df_counts = Counter()

        for token_list in self.docs_df["tokens"]:
            for t in set(token_list):
                if t in tokens:
                    df_counts[t] += 1

        idf_scores = {t: log((N + 1) / (df_counts[t] + 1)) + 1 for t in tokens}
        print("[📐] Sample IDF scores:")
        for token, score in idf_scores.items():
            print(f"  {token}: {score:.3f}")
        return idf_scores

    def overlap_stats(self):
        overlaps = []
        grouped = self.triples_df.groupby("query_id")

        for qid, group in grouped:
            q_tokens = set(self.query_map.get(qid, []))
            for _, row in group.iterrows():
                pos_tokens = set(self.doc_map.get(row["doc_id_a"], []))
                neg_tokens = set(self.doc_map.get(row["doc_id_b"], []))

                pos_overlap = len(q_tokens & pos_tokens) / (len(q_tokens) + 1e-6)
                neg_overlap = len(q_tokens & neg_tokens) / (len(q_tokens) + 1e-6)

                overlaps.append({
                    "query_id": qid,
                    "pos_overlap": pos_overlap,
                    "neg_overlap": neg_overlap
                })

        df = pd.DataFrame(overlaps)
        print(f"[🔁] Avg positive overlap: {df['pos_overlap'].mean():.3f}")
        print(f"[🔁] Avg negative overlap: {df['neg_overlap'].mean():.3f}")
        return df

    def plot_and_log(self):
        self.plot_triplets_per_query()
        self.plot_positive_vs_negative_overlap()
