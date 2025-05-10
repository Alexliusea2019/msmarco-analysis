import wandb
from load_data import load_data
from analyzer import MSMARCOAnalyzer


def main():
    # Initialize Weights & Biases
    wandb.init(project="msmarco-analysis", name="full-stats-run")

    # Load data
    docs_df, queries_df, triples_df = load_data()

    # Instantiate analyzer
    analyzer = MSMARCOAnalyzer(docs_df, queries_df, triples_df)

    # 1. Triplet counts
    analyzer.analyze_triplet_counts()
    wandb.log({"triplet_count_per_query": triples_df["query_id"].value_counts().describe().to_dict()})

    # 2. Token stats
    analyzer.token_stats()

    # 3. Top-k token frequency
    top_doc_tokens, top_query_tokens = analyzer.top_token_freq()
    wandb.log({
        "top_doc_tokens": {k: v for k, v in top_doc_tokens},
        "top_query_tokens": {k: v for k, v in top_query_tokens}
    })

    # 4. IDF stats (for top query tokens)
    idf_scores = analyzer.idf_stats([t for t, _ in top_query_tokens])
    wandb.log({"idf_scores": idf_scores})

    # 5. Token overlap stats
    overlap_df = analyzer.overlap_stats()
    wandb.log({
        "avg_positive_overlap": overlap_df["pos_overlap"].mean(),
        "avg_negative_overlap": overlap_df["neg_overlap"].mean()
    })

    # 6. Plots
    analyzer.plot_triplets_per_query()
    analyzer.plot_positive_vs_negative_overlap()

    wandb.log({
        "triplets_per_query_plot": wandb.Image("triplets_per_query.png"),
        "positive_vs_negative_plot": wandb.Image("positive_vs_negative_overlap.png")
    })

    wandb.finish()
    print("[✅] Analysis complete and logged to Weights & Biases.")


if __name__ == "__main__":
    main()
