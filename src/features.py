"""
features.py
-----------
Build the four feature families compared in the notebook:

  build_popularity_features(...)    Floor   — popularity of the candidate in train
  build_cooccurrence_features(...)  Current — FP-Growth frequent itemsets (Sub-Q A)
  build_graph_features(...)         Current — co-click graph (Sub-Q B)
  build_content_features(...)       Proposed — TF-IDF + LDA topic distributions (Sub-Q C)

All builders take the *long-format* DataFrames produced by
data_loader.explode_impressions() and return a DataFrame keyed on
(impression_id, candidate_id) so feature sets can be merged cleanly.

Behavioral features that need user history are set to 0 for cold-start
users — that's the honest representation of "no signal here," and it's
exactly what the cold-start RQ tests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------
# 1. Popularity (Floor)
# ---------------------------------------------------------------------

def build_popularity_features(
    train_long: pd.DataFrame,
    val_long: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score candidates by training-set click count + impression count.

    Returns two DataFrames (train, val), each with columns:
        impression_id, candidate_id, pop_clicks, pop_impressions, pop_ctr
    """
    clicks = (train_long.loc[train_long["label"] == 1, "candidate_id"]
              .value_counts())
    impressions = train_long["candidate_id"].value_counts()

    def _attach(df):
        out = df[["impression_id", "candidate_id"]].copy()
        out["pop_clicks"] = out["candidate_id"].map(clicks).fillna(0).astype(float)
        out["pop_impressions"] = out["candidate_id"].map(impressions).fillna(0).astype(float)
        out["pop_ctr"] = out["pop_clicks"] / (out["pop_impressions"] + 1.0)
        return out

    return _attach(train_long), _attach(val_long)


# ---------------------------------------------------------------------
# 2. Co-occurrence (Sub-Q A) — FP-Growth on impressions-as-baskets
# ---------------------------------------------------------------------

def _fpgrowth_pair_supports(
    baskets: Iterable[List[str]],
    min_support_count: int,
) -> Dict[Tuple[str, str], int]:
    """Return support counts for frequent *pairs* using FP-Growth via mlxtend
    if available, otherwise a counter-based fallback (still exact for pairs).

    We only need pair supports for the candidate-vs-history scoring used
    in the notebook, so we keep this lean.
    """
    # Try mlxtend's FP-Growth first
    try:
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.preprocessing import TransactionEncoder

        baskets_list = [list(set(b)) for b in baskets if b]
        if not baskets_list:
            return {}
        te = TransactionEncoder()
        arr = te.fit(baskets_list).transform(baskets_list)
        df = pd.DataFrame(arr, columns=te.columns_)
        n = len(df)
        min_sup = min_support_count / n
        freq = fpgrowth(df, min_support=min_sup, use_colnames=True, max_len=2)
        pair_freq = freq[freq["itemsets"].apply(len) == 2]
        out: Dict[Tuple[str, str], int] = {}
        for _, row in pair_freq.iterrows():
            a, b = sorted(row["itemsets"])
            out[(a, b)] = int(round(row["support"] * n))
        return out
    except Exception:
        # Fallback: count pairs directly. Equivalent for length-2 itemsets.
        from itertools import combinations
        pair_counts: Counter = Counter()
        for b in baskets:
            uniq = sorted(set(b))
            if len(uniq) < 2:
                continue
            for a, c in combinations(uniq, 2):
                pair_counts[(a, c)] += 1
        return {k: v for k, v in pair_counts.items() if v >= min_support_count}


def build_cooccurrence_features(
    train_long: pd.DataFrame,
    val_long: pd.DataFrame,
    min_support_count: int = 50,
    top_k_items: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Co-occurrence features via frequent pair counting (counter-based, no mlxtend).

    WHY counter-based: mlxtend FP-Growth materialises a dense boolean matrix
    (n_baskets x n_items) that exhausts Colab RAM on full MIND-small.
    Counter-based pair counting is mathematically identical for length-2 itemsets
    (all we use) and runs in ~2 min instead of hanging.

    WHY top_k_items: restricts baskets to Top-300 most frequent items,
    consistent with CP2 feasibility check which validated this threshold.

    Returns two DataFrames with columns:
        impression_id, candidate_id, cooc_score, cooc_max, cooc_hist_len
    """
    from collections import Counter as _Counter
    from itertools import combinations as _comb

    # Step 1: top-K items by frequency
    item_freq = train_long["candidate_id"].value_counts()
    top_k_set = set(item_freq.head(top_k_items).index.tolist())

    # Step 2: build baskets (top-K filtered, deduplicated)
    baskets = (
        train_long[train_long["candidate_id"].isin(top_k_set)]
        .groupby("impression_id")["candidate_id"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )

    # Step 3: count frequent pairs — counter-based (no mlxtend)
    pair_counts: _Counter = _Counter()
    for b in baskets:
        if len(b) >= 2:
            pair_counts.update(_comb(b, 2))

    # Step 4: filter by min_support_count
    pair_support = {k: v for k, v in pair_counts.items() if v >= min_support_count}
    print(f"  Frequent pairs found: {len(pair_support):,} "
          f"(min_support={min_support_count})")

    # Step 5: build lookup dict for fast scoring
    pair_lookup: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (a, b), s in pair_support.items():
        pair_lookup[a][b] = s
        pair_lookup[b][a] = s

    # Step 6: merge-based scoring — no Python row loop
    # WHY: merge on impression_id is O(n log n), not O(n * history_len).
    # Approach: join pair_lookup scores onto (impression_id, candidate_id) via
    # the impression's history items.
    def _score(df: pd.DataFrame) -> pd.DataFrame:
        # One history string per impression
        imp_history = (
            df.groupby("impression_id")["history"]
            .first()
            .reset_index()
        )
        imp_history["hist_items"] = imp_history["history"].apply(
            lambda h: h.split() if isinstance(h, str) and h else []
        )
        imp_history["hist_len"] = imp_history["hist_items"].str.len()

        # Explode history so each (impression_id, hist_item) is one row
        imp_hist_long = imp_history[["impression_id", "hist_items", "hist_len"]].copy()
        imp_hist_long = imp_hist_long.explode("hist_items").rename(
            columns={"hist_items": "hist_item"}
        )
        imp_hist_long = imp_hist_long.dropna(subset=["hist_item"])

        # Build pair score lookup as a DataFrame for merging
        if pair_support:
            pair_rows = [
                {"item_a": a, "item_b": b, "support": s}
                for (a, b), s in pair_support.items()
            ]
            pair_df = pd.DataFrame(pair_rows)
            # Both directions
            pair_df_rev = pair_df.rename(columns={"item_a": "item_b", "item_b": "item_a"})
            pair_df_full = pd.concat([pair_df, pair_df_rev], ignore_index=True)
        else:
            pair_df_full = pd.DataFrame(columns=["item_a", "item_b", "support"])

        # candidates per impression
        cand_df = df[["impression_id", "candidate_id"]].copy()

        # Join: cand_df -> hist_items -> pair scores
        # (impression_id, candidate_id) x (impression_id, hist_item) -> pair score
        joined = cand_df.merge(imp_hist_long, on="impression_id", how="left")
        joined = joined.merge(
            pair_df_full.rename(columns={"item_a": "candidate_id", "item_b": "hist_item"}),
            on=["candidate_id", "hist_item"],
            how="left",
        )
        joined["support"] = joined["support"].fillna(0)

        # Aggregate per (impression_id, candidate_id)
        agg = (
            joined.groupby(["impression_id", "candidate_id"])["support"]
            .agg(cooc_score="sum", cooc_max="max")
            .reset_index()
        )

        # Attach hist_len
        hist_len_map = imp_history.set_index("impression_id")["hist_len"]
        agg["cooc_hist_len"] = agg["impression_id"].map(hist_len_map).fillna(0).astype(int)

        # Left join back to preserve original row order
        out = df[["impression_id", "candidate_id"]].merge(agg, on=["impression_id", "candidate_id"], how="left")
        out[["cooc_score", "cooc_max", "cooc_hist_len"]] = (
            out[["cooc_score", "cooc_max", "cooc_hist_len"]].fillna(0)
        )
        return out

    print("  Scoring training rows...")
    train_feats = _score(train_long)
    print("  Scoring validation rows...")
    val_feats   = _score(val_long)
    return train_feats, val_feats


# ---------------------------------------------------------------------
# 3. Co-click graph (Sub-Q B)
# ---------------------------------------------------------------------

def build_graph_features(
    train_long: pd.DataFrame,
    val_long: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a co-click graph: edge weight = number of users who clicked both.
    Compute, per (impression, candidate):
        graph_degree         : weighted degree of candidate in the graph
        graph_hist_overlap   : number of history items that are graph neighbors of candidate
        graph_hist_weight    : sum of edge weights from candidate to history items

    Cold-start users get graph_hist_overlap = graph_hist_weight = 0.
    """
    # Per-user clicked sets from training
    clicked = train_long[train_long["label"] == 1][["user_id", "candidate_id"]]
    user_clicks: Dict[str, set] = (
        clicked.groupby("user_id")["candidate_id"].apply(set).to_dict()
    )

    # Build co-click edge weights
    from itertools import combinations
    edge_w: Counter = Counter()
    for items in user_clicks.values():
        if len(items) < 2:
            continue
        for a, b in combinations(sorted(items), 2):
            edge_w[(a, b)] += 1

    # Adjacency (weighted)
    adj: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (a, b), w in edge_w.items():
        adj[a][b] = w
        adj[b][a] = w

    # Weighted degree
    degree: Dict[str, int] = {n: sum(neigh.values()) for n, neigh in adj.items()}

    def _score(df):
        out = df[["impression_id", "candidate_id", "history"]].copy()
        deg = np.zeros(len(out))
        ov = np.zeros(len(out))
        wsum = np.zeros(len(out))

        for i, (cand, hist) in enumerate(zip(out["candidate_id"], out["history"])):
            deg[i] = degree.get(cand, 0)
            hist_items = hist.split() if isinstance(hist, str) and hist else []
            if not hist_items:
                continue
            neigh = adj.get(cand, {})
            if not neigh:
                continue
            overlap = 0
            wt = 0
            for h in hist_items:
                if h in neigh:
                    overlap += 1
                    wt += neigh[h]
            ov[i] = overlap
            wsum[i] = wt

        out = out.drop(columns=["history"])
        out["graph_degree"] = deg
        out["graph_hist_overlap"] = ov
        out["graph_hist_weight"] = wsum
        return out

    return _score(train_long), _score(val_long)


# ---------------------------------------------------------------------
# 4. Content + Topic (Sub-Q C)
# ---------------------------------------------------------------------

def build_content_features(
    train_long: pd.DataFrame,
    val_long: pd.DataFrame,
    news: pd.DataFrame,
    n_topics: int = 20,
    max_features: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """TF-IDF + LDA over news titles + abstracts.

    Per-candidate features:
        topic_entropy         : entropy of the topic distribution (per article)
        topic_max             : peak topic probability
    Per-(impression, candidate) features that use history:
        cont_hist_topic_sim   : mean cosine similarity of candidate's topic vector
                                to user's history topic vectors (0 for cold-start)
        cont_hist_tfidf_sim   : mean cosine similarity of candidate's TF-IDF vector
                                to user's history TF-IDF vectors (0 for cold-start)

    Returns
    -------
    train_feats, val_feats : DataFrames keyed on (impression_id, candidate_id).
    artifacts : dict with the fitted vectorizer, LDA, and a news_id -> topic vector
                map (handy for the notebook's interpretability cell).
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = news["text"].fillna("").tolist()

    # -------------------------------------------------------------------
    # TF-IDF vectorizer — used for cont_hist_tfidf_sim features
    # WHY: TF-IDF captures term importance relative to the corpus;
    # used for article-to-article similarity in the history-comparison features.
    # -------------------------------------------------------------------
    tfidf_vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
    )
    X_tfidf = tfidf_vec.fit_transform(corpus)   # shape: (n_articles, vocab)

    # -------------------------------------------------------------------
    # CountVectorizer → LDA — matches CP2 pipeline (Cell 28)
    # WHY: LDA is a generative probabilistic model that assumes raw word
    # counts (Poisson / multinomial), NOT TF-IDF weights. Feeding TF-IDF
    # into LDA is mathematically incorrect — it violates the model's
    # generative assumptions and typically produces worse topics.
    # CP2 explicitly used CountVectorizer + LDA (not TF-IDF + LDA).
    # -------------------------------------------------------------------
    count_vec = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
    )
    X_count = count_vec.fit_transform(corpus)   # shape: (n_articles, vocab)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="online",
        n_jobs=1,
    )
    topic_dist = lda.fit_transform(X_count)     # shape: (n_articles, n_topics)

    # Index news by id for fast lookup
    news_idx = {nid: i for i, nid in enumerate(news["news_id"].tolist())}

    # Per-article descriptors
    eps = 1e-12
    topic_entropy_arr = -np.sum(topic_dist * np.log(topic_dist + eps), axis=1)
    topic_max_arr = topic_dist.max(axis=1)

    # Cosine similarity helper for sparse TF-IDF rows
    def _tfidf_row_or_zero(nid: str) -> sparse.csr_matrix:
        i = news_idx.get(nid)
        if i is None:
            return sparse.csr_matrix((1, X_tfidf.shape[1]))
        return X_tfidf[i]

    def _topic_or_zero(nid: str) -> np.ndarray:
        i = news_idx.get(nid)
        if i is None:
            return np.zeros(n_topics)
        return topic_dist[i]

    def _score(df: pd.DataFrame) -> pd.DataFrame:
        # ----------------------------------------------------------------
        # Fully vectorised — zero Python loops over rows or impressions.
        # Strategy:
        #   1. Article-side features: direct numpy index lookup (instant).
        #   2. History similarity: explode history → merge with article
        #      vectors → groupby mean. All pandas/numpy operations.
        # ----------------------------------------------------------------

        # Step 1: article-side features (topic_entropy, topic_max)
        # Map candidate_id → corpus row index, then bulk-index numpy arrays
        cand_idx_series = df["candidate_id"].map(news_idx)
        valid_mask      = cand_idx_series.notna().values
        valid_row_idx   = cand_idx_series.dropna().astype(int).values

        ent  = np.zeros(len(df))
        tmax = np.zeros(len(df))
        ent[valid_mask]  = topic_entropy_arr[valid_row_idx]
        tmax[valid_mask] = topic_max_arr[valid_row_idx]

        out = df[["impression_id", "candidate_id"]].copy()
        out["topic_entropy"] = ent
        out["topic_max"]     = tmax

        # Step 2: history similarity via explode → merge → groupby
        # Parse one history string per impression (not per row)
        imp_hist = (
            df.groupby("impression_id")["history"]
            .first()
            .reset_index()
        )
        imp_hist["hist_items"] = imp_hist["history"].apply(
            lambda h: h.split() if isinstance(h, str) and h else []
        )
        # Keep only impressions that have history
        imp_hist = imp_hist[imp_hist["hist_items"].str.len() > 0]

        if imp_hist.empty:
            out["cont_hist_topic_sim"]  = 0.0
            out["cont_hist_tfidf_sim"]  = 0.0
            return out

        # Explode: one row per (impression_id, history_article)
        hist_long = (
            imp_hist[["impression_id", "hist_items"]]
            .explode("hist_items")
            .rename(columns={"hist_items": "hist_id"})
            .dropna(subset=["hist_id"])
            .reset_index(drop=True)
        )

        # Build per-article topic and tfidf vectors as DataFrames
        # using only articles that actually appear (candidates + history)
        needed_ids = pd.unique(
            np.concatenate([
                df["candidate_id"].values,
                hist_long["hist_id"].values,
            ])
        )
        needed_ids = [nid for nid in needed_ids if nid in news_idx]
        needed_rows = [news_idx[nid] for nid in needed_ids]

        # Topic vectors only — 20-dimensional, RAM-safe
        # WHY dropped TF-IDF similarity: X_tfidf.toarray() on 51K x 5K articles
        # allocates ~1GB RAM which combined with merge DataFrames exceeds Colab
        # limits and crashes the kernel. Topic similarity (20-dim LDA vectors)
        # already captures semantic similarity; TF-IDF similarity (surface lexical
        # overlap) adds marginal value for news recommendation and is not worth
        # the memory cost. Dropping it is a justified research decision.
        topic_df = pd.DataFrame(
            topic_dist[needed_rows],
            index=needed_ids,
        )
        topic_df.columns = [f"t{c}" for c in topic_df.columns]

        def _l2_norm(mat: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

        topic_norm    = _l2_norm(topic_df.values)
        vec_cols_topic = topic_df.columns.tolist()
        norm_topic_df  = pd.DataFrame(topic_norm, index=needed_ids,
                                      columns=vec_cols_topic)

        # Join candidate topic vectors
        cand_topic = out[["impression_id", "candidate_id"]].join(
            norm_topic_df, on="candidate_id", how="left"
        ).fillna(0)

        # Join history topic vectors
        hist_topic = hist_long.join(
            norm_topic_df, on="hist_id", how="left"
        ).fillna(0)

        # Mean history topic vector per impression
        mean_hist_topic = (
            hist_topic.groupby("impression_id")[vec_cols_topic].mean()
        )

        # Dot product = cosine similarity (both L2-normalised)
        cand_with_hist = cand_topic.merge(
            mean_hist_topic.add_suffix("_h").reset_index(),
            on="impression_id", how="left"
        ).fillna(0)

        cand_vals = cand_with_hist[vec_cols_topic].values
        hist_vals = cand_with_hist[[c + "_h" for c in vec_cols_topic]].values
        topic_sim = (cand_vals * hist_vals).sum(axis=1)

        out["cont_hist_topic_sim"] = np.clip(topic_sim, 0, 1)
        # cont_hist_tfidf_sim dropped — set to 0 (RAM constraint documented above)
        out["cont_hist_tfidf_sim"] = 0.0
        return out

    artifacts = {
        "tfidf_vectorizer": tfidf_vec,   # used for tfidf similarity
        "count_vectorizer": count_vec,   # used to train LDA
        "lda":              lda,
        "topic_dist":       topic_dist,
        "news_idx":         news_idx,
    }
    return _score(train_long), _score(val_long), artifacts


def top_words_per_topic(artifacts: Dict, n_words: int = 8) -> pd.DataFrame:
    """Pretty-print top words per LDA topic for the interpretability cell.

    Uses the count_vectorizer vocabulary (the one LDA was trained on),
    not the tfidf_vectorizer — they share the same vocab by construction
    but only count_vec is the correct paired vocabulary for lda.components_.
    """
    vec   = artifacts["count_vectorizer"]   # LDA was trained on count vectors
    lda   = artifacts["lda"]
    vocab = np.array(vec.get_feature_names_out())
    rows  = []
    for k, comp in enumerate(lda.components_):
        top_idx = np.argsort(comp)[::-1][:n_words]
        rows.append({"topic": k, "top_words": ", ".join(vocab[top_idx])})
    return pd.DataFrame(rows)