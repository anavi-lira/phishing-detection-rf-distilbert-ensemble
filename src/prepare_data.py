from __future__ import annotations
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.utils import ROOT, ensure_dirs, load_config, set_seed

def normalize_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1","1.0","phishing","phishing email","malicious","fraud"}:
        return 1
    if text in {"0","0.0","safe","safe email","legitimate","legitimate email","ham","benign"}:
        return 0
    if "phish" in text or "fraud" in text or "malicious" in text:
        return 1
    if "safe" in text or "legit" in text or "benign" in text or text == "ham":
        return 0
    raise ValueError(f"Rótulo não reconhecido: {value!r}")

def prepare_data() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    ds_cfg = cfg["dataset"]
    set_seed(seed)
    ensure_dirs()

    print(f"Baixando dataset: {ds_cfg['name']} ...")
    df = load_dataset(ds_cfg["name"], split=ds_cfg["split"]).to_pandas()

    text_col, label_col = ds_cfg["text_column"], ds_cfg["label_column"]
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Colunas esperadas não encontradas. Disponíveis: {list(df.columns)}")

    df = df[[text_col,label_col]].rename(columns={text_col:"text", label_col:"raw_label"})
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["raw_label"].apply(normalize_label)
    df = df.drop(columns=["raw_label"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

    max_samples = int(ds_cfg["max_samples"])
    if max_samples and max_samples < len(df):
        df, _ = train_test_split(
            df, train_size=max_samples, random_state=seed, stratify=df["label"]
        )
        df = df.reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.insert(0, "row_id", range(len(df)))

    train_ratio = float(ds_cfg["train_ratio"])
    val_ratio = float(ds_cfg["val_ratio"])
    test_ratio = float(ds_cfg["test_ratio"])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio deve ser 1.0")

    train_df, temp_df = train_test_split(
        df, test_size=val_ratio+test_ratio, random_state=seed, stratify=df["label"]
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, random_state=seed, stratify=temp_df["label"]
    )

    out = ROOT/"data"/"processed"
    for name, part in [("train",train_df),("val",val_df),("test",test_df)]:
        part.to_csv(out/f"{name}.csv", index=False)
        print(name, len(part), part["label"].value_counts().sort_index().to_dict())

if __name__ == "__main__":
    prepare_data()
