from __future__ import annotations
import re, joblib, pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src.metrics import classification_metrics, save_json
from src.utils import ROOT, ensure_dirs, load_config, set_seed

def clean_for_rf(text: str) -> str:
    text = BeautifulSoup(str(text), "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def train_random_forest() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs()
    p = ROOT/"data"/"processed"
    train_df, val_df, test_df = [pd.read_csv(p/f"{x}.csv") for x in ["train","val","test"]]
    rf_cfg = cfg["random_forest"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_for_rf, stop_words="english",
            max_features=int(rf_cfg["tfidf_max_features"]),
            ngram_range=(int(rf_cfg["ngram_min"]), int(rf_cfg["ngram_max"])),
            sublinear_tf=True,
        )),
        ("rf", RandomForestClassifier(random_state=cfg["seed"], n_jobs=-1))
    ])

    if rf_cfg.get("grid_search", True):
        grid = {
            "rf__n_estimators":[200,400],
            "rf__max_depth":[None,30],
            "rf__min_samples_split":[2,5],
            "rf__class_weight":[None,"balanced"],
        }
        search = GridSearchCV(
            pipe, grid, scoring="f1", cv=int(rf_cfg["cv_folds"]), n_jobs=-1, verbose=1
        )
        search.fit(train_df["text"], train_df["label"])
        model, best_params = search.best_estimator_, search.best_params_
    else:
        model = pipe.set_params(
            rf__n_estimators=300, rf__class_weight="balanced"
        )
        model.fit(train_df["text"], train_df["label"])
        best_params = {"grid_search": False}

    joblib.dump(model, ROOT/"models"/"random_forest.joblib")
    summary = {"best_params":best_params}

    for split_name, df in [("val",val_df),("test",test_df)]:
        probs = model.predict_proba(df["text"])[:,1]
        preds = (probs >= 0.5).astype(int)
        summary[split_name] = classification_metrics(df["label"], preds, probs)
        pd.DataFrame({
            "row_id":df["row_id"], "label":df["label"], "rf_prob":probs
        }).to_csv(ROOT/"results"/f"rf_{split_name}_probs.csv", index=False)

    save_json(summary, ROOT/"results"/"random_forest_metrics.json")
    print("RF concluído:", best_params)

if __name__ == "__main__":
    train_random_forest()
