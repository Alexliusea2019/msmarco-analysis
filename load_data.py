import pandas as pd
from datasets import load_dataset

def load_data():
    print("[↓] Loading MSMARCO Passages (docs)...")
    passages = load_dataset("irds/msmarco-passage", "docs", split="train", trust_remote_code=True)
    passages_df = pd.DataFrame(passages)
    passages_df = passages_df.rename(columns={"docno": "doc_id", "text": "text"})[["doc_id", "text"]]
    passages_df["tokens"] = passages_df["text"].str.split()

    print("[↓] Loading MSMARCO Queries (dev queries)...")
    queries = load_dataset("irds/msmarco-passage_dev", "queries", split="train", trust_remote_code=True)
    queries_df = pd.DataFrame(queries)
    queries_df = queries_df.rename(columns={"query_id": "query_id", "text": "text"})[["query_id", "text"]]
    queries_df["tokens"] = queries_df["text"].str.split()

    print("[↓] Loading MSMARCO Triples (train_triples-small)...")
    triples = load_dataset("irds/msmarco-passage_train_triples-small", "docpairs", split="train", trust_remote_code=True)
    triples_df = pd.DataFrame(triples)
    triples_df = triples_df.rename(columns={
        "query_id": "query_id", 
        "doc_id_a": "doc_id_a", 
        "doc_id_b": "doc_id_b"
    })[["query_id", "doc_id_a", "doc_id_b"]]

    # ✅ Validate each DataFrame
    print("\n[✔] DataFrames Loaded:")
    print(f"Passages: {passages_df.shape}, Columns: {passages_df.columns.tolist()}")
    print(f"Queries: {queries_df.shape}, Columns: {queries_df.columns.tolist()}")
    print(f"Triples: {triples_df.shape}, Columns: {triples_df.columns.tolist()}")

    # 📏 Token length statistics
    avg_passage_len = passages_df["tokens"].apply(len).mean()
    avg_query_len = queries_df["tokens"].apply(len).mean()
    print(f"\n[📊] Average Passage Length (tokens): {avg_passage_len:.2f}")
    print(f"[📊] Average Query Length (tokens): {avg_query_len:.2f}")

    return passages_df, queries_df, triples_df

# Optional test
if __name__ == "__main__":
    load_data()
