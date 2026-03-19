#!/usr/bin/env python3
"""Preprocess the Blog Authorship Corpus.

Reads raw XML files from data/raw/blogs/, cleans and filters posts,
balances classes, splits by blogger, and saves a parquet file.
"""

import os
import re
import html
import pathlib
import warnings
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# ── paths ────────────────────────────────────────────────────────────────
RAW_DIR = pathlib.Path("/root/characterprobing/data/raw/blogs")
OUT_PATH = pathlib.Path("/workspace/characterprobing/data/processed/blog_corpus.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── age bins ─────────────────────────────────────────────────────────────
AGE_BINS = {
    1: range(13, 18),   # 13-17
    2: range(23, 28),   # 23-27
    3: range(33, 48),   # 33-47
}


def age_to_bin(age: int):
    for bin_id, r in AGE_BINS.items():
        if age in r:
            return bin_id
    return None


# ── HTML / whitespace cleaning ───────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw)          # strip HTML tags
    text = html.unescape(text)            # decode entities
    text = _WS_RE.sub(" ", text).strip()  # normalise whitespace
    return text


# ── XML parsing (lenient) ────────────────────────────────────────────────
_POST_RE = re.compile(r"<post>(.*?)</post>", re.DOTALL | re.IGNORECASE)


def parse_blog_file(path: pathlib.Path) -> list[str]:
    """Return list of raw post strings from a blog XML file."""
    for enc in ("utf-8", "latin-1"):
        try:
            content = path.read_text(encoding=enc, errors="replace")
            break
        except Exception:
            continue
    else:
        return []

    posts = _POST_RE.findall(content)
    return posts


def parse_filename(fname: str):
    """Extract metadata from filename like {id}.{gender}.{age}.{industry}.{sign}.xml"""
    base = fname.rsplit(".xml", 1)[0]
    parts = base.split(".")
    if len(parts) < 5:
        return None
    blogger_id = parts[0]
    gender = parts[1]
    try:
        age = int(parts[2])
    except ValueError:
        return None
    industry = parts[3]
    star_sign = parts[4]
    return blogger_id, gender, age, industry, star_sign


# ── main ─────────────────────────────────────────────────────────────────
def main():
    # Load tokenizer (used only for counting)
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    xml_files = sorted(RAW_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files.")

    # ── Step 1-3: parse, clean, tokenise, filter posts ───────────────────
    # Accumulate per-blogger: metadata + list of (text, n_tokens)
    blogger_meta = {}          # blogger_id -> (gender, age, industry, star_sign)
    blogger_posts = defaultdict(list)  # blogger_id -> [(text, n_tokens), …]

    for fpath in tqdm(xml_files, desc="Parsing files"):
        meta = parse_filename(fpath.name)
        if meta is None:
            continue
        blogger_id, gender, age, industry, star_sign = meta

        # Age bin filter (step 5 – early, saves work)
        age_bin = age_to_bin(age)
        if age_bin is None:
            continue

        raw_posts = parse_blog_file(fpath)
        for raw in raw_posts:
            text = clean_text(raw)
            if not text:
                continue
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if 50 <= n_tokens <= 1024:
                blogger_posts[blogger_id].append((text, n_tokens))

        if blogger_id in blogger_posts:
            blogger_meta[blogger_id] = (gender, age, age_bin, star_sign)

    # ── Step 4: keep bloggers with >= 3 qualifying posts ─────────────────
    qualified = {
        bid for bid, posts in blogger_posts.items() if len(posts) >= 3
    }
    print(f"Bloggers with >=3 qualifying posts: {len(qualified)}")

    # Build a blogger-level DataFrame for balancing / splitting
    blogger_rows = []
    for bid in qualified:
        gender, age, age_bin, star_sign = blogger_meta[bid]
        blogger_rows.append({
            "blogger_id": bid,
            "gender": gender,
            "age": age,
            "age_bin": age_bin,
            "star_sign": star_sign,
        })
    bloggers_df = pd.DataFrame(blogger_rows)
    print(f"Bloggers after age-bin filter: {len(bloggers_df)}")

    # ── Step 6: class balancing (blogger level, primary = age_bin) ───────
    age_counts = bloggers_df["age_bin"].value_counts()
    min_age = age_counts.min()
    print(f"Age-bin counts before balancing: {age_counts.to_dict()}")
    print(f"Downsampling each age_bin to {min_age} bloggers.")

    balanced_ids = []
    for ab in bloggers_df["age_bin"].unique():
        subset = bloggers_df[bloggers_df["age_bin"] == ab]
        sampled = subset.sample(n=min_age, random_state=42)
        balanced_ids.append(sampled)
    bloggers_df = pd.concat(balanced_ids, ignore_index=True)
    print(f"Bloggers after age-bin balancing: {len(bloggers_df)}")

    # ── Step 7: stratified 70/15/15 split by blogger ─────────────────────
    train_df, temp_df = train_test_split(
        bloggers_df, test_size=0.30, random_state=42,
        stratify=bloggers_df["age_bin"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42,
        stratify=temp_df["age_bin"],
    )

    split_map = {}
    for bid in train_df["blogger_id"]:
        split_map[bid] = "train"
    for bid in val_df["blogger_id"]:
        split_map[bid] = "val"
    for bid in test_df["blogger_id"]:
        split_map[bid] = "test"

    # ── Step 8: build final post-level DataFrame and save ────────────────
    kept_ids = set(bloggers_df["blogger_id"])
    rows = []
    for bid in kept_ids:
        gender, age, age_bin, star_sign = blogger_meta[bid]
        split = split_map[bid]
        for text, n_tokens in blogger_posts[bid]:
            rows.append({
                "blogger_id": bid,
                "text": text,
                "age_bin": age_bin,
                "gender": gender,
                "star_sign": star_sign,
                "split": split,
                "n_tokens": n_tokens,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} posts to {OUT_PATH}")

    # ── Step 9: summary statistics ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for split_name in ("train", "val", "test"):
        sub = df[df["split"] == split_name]
        n_bloggers = sub["blogger_id"].nunique()
        n_posts = len(sub)
        print(f"\n── {split_name.upper()} ── bloggers: {n_bloggers}  posts: {n_posts}")

    print("\n── CLASS DISTRIBUTIONS (blogger level) ──")
    blogger_level = bloggers_df.copy()
    blogger_level["split"] = blogger_level["blogger_id"].map(split_map)

    for task in ("age_bin", "gender", "star_sign"):
        print(f"\n  {task}:")
        counts = blogger_level.groupby(["split", task]).size().unstack(fill_value=0)
        print(counts.to_string(index=True))

    print("\n── CLASS DISTRIBUTIONS (post level) ──")
    for task in ("age_bin", "gender", "star_sign"):
        print(f"\n  {task}:")
        counts = df.groupby(["split", task]).size().unstack(fill_value=0)
        print(counts.to_string(index=True))

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
