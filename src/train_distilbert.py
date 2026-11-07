from __future__ import annotations
import numpy as np, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from src.metrics import classification_metrics, save_json
from src.utils import ROOT, ensure_dirs, load_config, set_seed

def to_hf_dataset(df, tokenizer, max_length):
    ds = Dataset.from_pandas(df[["row_id","text","label"]], preserve_index=False)
    ds = ds.map(lambda b: tokenizer(
        b["text"], truncation=True, padding="max_length", max_length=max_length
    ), batched=True)
    ds = ds.rename_column("label","labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels,preds),
        "precision": precision_score(labels,preds,zero_division=0),
        "recall": recall_score(labels,preds,zero_division=0),
        "f1": f1_score(labels,preds,zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels,preds),
    }

def train_distilbert() -> None:
    cfg=load_config(); set_seed(cfg["seed"]); ensure_dirs()
    db=cfg["distilbert"]; p=ROOT/"data"/"processed"
    train_df,val_df,test_df=[pd.read_csv(p/f"{x}.csv") for x in ["train","val","test"]]
    tokenizer=AutoTokenizer.from_pretrained(db["model_name"])
    model=AutoModelForSequenceClassification.from_pretrained(db["model_name"], num_labels=2)
    if hasattr(model.config,"dropout"): model.config.dropout=0.1
    if hasattr(model.config,"seq_classif_dropout"): model.config.seq_classif_dropout=0.1

    train_ds=to_hf_dataset(train_df,tokenizer,int(db["max_length"]))
    val_ds=to_hf_dataset(val_df,tokenizer,int(db["max_length"]))
    test_ds=to_hf_dataset(test_df,tokenizer,int(db["max_length"]))

    args=TrainingArguments(
        output_dir=str(ROOT/"models"/"distilbert_checkpoints"),
        learning_rate=float(db["learning_rate"]),
        per_device_train_batch_size=int(db["batch_size"]),
        per_device_eval_batch_size=int(db["batch_size"]),
        num_train_epochs=int(db["epochs"]),
        weight_decay=float(db["weight_decay"]),
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        greater_is_better=True, save_total_limit=2, report_to="none",
        seed=cfg["seed"], fp16=torch.cuda.is_available(),
    )
    trainer=Trainer(
        model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,
        tokenizer=tokenizer,compute_metrics=compute_metrics
    )
    trainer.train()
    model_dir=ROOT/"models"/"distilbert"
    trainer.save_model(str(model_dir)); tokenizer.save_pretrained(str(model_dir))

    summary={}
    for split_name,df,ds in [("val",val_df,val_ds),("test",test_df,test_ds)]:
        out=trainer.predict(ds)
        probs=torch.softmax(torch.tensor(out.predictions),dim=1).numpy()[:,1]
        preds=(probs>=0.5).astype(int)
        summary[split_name]=classification_metrics(df["label"],preds,probs)
        pd.DataFrame({
            "row_id":df["row_id"],"label":df["label"],"distilbert_prob":probs
        }).to_csv(ROOT/"results"/f"distilbert_{split_name}_probs.csv",index=False)
    save_json(summary,ROOT/"results"/"distilbert_metrics.json")
    print("DistilBERT concluído.")

if __name__=="__main__":
    train_distilbert()
