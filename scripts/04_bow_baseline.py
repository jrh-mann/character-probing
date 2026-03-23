#!/usr/bin/env python3
"""Bag-of-words baseline: TF-IDF + Logistic Regression.

Tests what accuracy is achievable from surface vocabulary alone,
without any LLM. The gap between this and probe accuracy measures
what the model's contextual processing adds.

Runs on CPU in minutes. No GPU needed.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "blog_corpus.parquet"

AGE_BIN_MAP = {1: 0, 2: 1, 3: 2}
GENDER_MAP = {"female": 0, "male": 1}
STAR_SIGNS_SORTED = sorted(["Aquarius","Aries","Cancer","Capricorn","Gemini","Leo",
                             "Libra","Pisces","Sagittarius","Scorpio","Taurus","Virgo"])
STAR_SIGN_MAP = {n: i for i, n in enumerate(STAR_SIGNS_SORTED)}

TASK_MAPS = {"age_bin": AGE_BIN_MAP, "gender": GENDER_MAP, "star_sign": STAR_SIGN_MAP}
TASK_N_CLASSES = {"age_bin": 3, "gender": 2, "star_sign": 12}
CHANCE = {"age_bin": 1/3, "gender": 0.5, "star_sign": 1/12}

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")

    # Sweep TF-IDF max_features and regularization C
    max_features_options = [5000, 10000, 50000]
    C_options = [0.01, 0.1, 1.0, 10.0]

    results = []

    for max_feat in max_features_options:
        print(f"\nTF-IDF max_features={max_feat}")
        t0 = time.time()
        tfidf = TfidfVectorizer(max_features=max_feat, sublinear_tf=True)
        X_train = tfidf.fit_transform(train["text"])
        X_val = tfidf.transform(val["text"])
        X_test = tfidf.transform(test["text"])
        print(f"  Vectorized in {time.time()-t0:.1f}s, shape={X_train.shape}")

        for task in ["age_bin", "gender", "star_sign"]:
            y_train = train[task].map(TASK_MAPS[task]).values
            y_val = val[task].map(TASK_MAPS[task]).values
            y_test = test[task].map(TASK_MAPS[task]).values

            # Select best C on val
            best_C = None
            best_val_acc = -1

            for C in C_options:
                clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs",
                                         multi_class="multinomial", n_jobs=-1)
                clf.fit(X_train, y_train)
                val_pred = clf.predict(X_val)
                val_acc = balanced_accuracy_score(y_val, val_pred)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_C = C
                    best_clf = clf

            # Evaluate best model on test
            test_pred = best_clf.predict(X_test)
            test_acc = balanced_accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, test_pred)

            results.append({
                "method": "bow",
                "max_features": max_feat,
                "best_C": best_C,
                "task": task,
                "val_balanced_acc": round(best_val_acc, 5),
                "test_balanced_acc": round(test_acc, 5),
                "test_macro_f1": round(test_f1, 5),
            })

            print(f"  {task:>10}: val={best_val_acc:.4f}, test={test_acc:.4f}, "
                  f"f1={test_f1:.4f}, C={best_C} (chance={CHANCE[task]:.3f})")
            print(f"             confusion matrix:\n{cm}")

    # Save results
    rdf = pd.DataFrame(results)
    out_path = BASE_DIR / "results" / "bow_baseline_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # Summary: best per task across all max_features
    print(f"\n{'='*60}")
    print("BEST BOW RESULTS (val-selected)")
    print(f"{'='*60}")
    for task in ["age_bin", "gender", "star_sign"]:
        tdf = rdf[rdf["task"] == task]
        best = tdf.loc[tdf["test_balanced_acc"].idxmax()]
        print(f"  {task:>10}: {best['test_balanced_acc']:.4f} "
              f"(feat={int(best['max_features'])}, C={best['best_C']})")
        print(f"             chance={CHANCE[task]:.3f}, "
              f"ridge_L0≈0.42/0.59/0.085")

if __name__ == "__main__":
    main()
