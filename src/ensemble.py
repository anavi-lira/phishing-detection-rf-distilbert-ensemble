from __future__ import annotations
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from src.metrics import classification_metrics, save_json
from src.utils import ROOT, ensure_dirs, load_config

def load_probs(split):
    rf=pd.read_csv(ROOT/"results"/f"rf_{split}_probs.csv")
    db=pd.read_csv(ROOT/"results"/f"distilbert_{split}_probs.csv")
    return rf.merge(db[["row_id","distilbert_prob"]],on="row_id",validate="one_to_one")

def choose_weight(df,step,threshold):
    best=None
    for alpha in np.arange(0,1+step/2,step):
        prob=alpha*df["rf_prob"].values+(1-alpha)*df["distilbert_prob"].values
        pred=(prob>=threshold).astype(int)
        score=f1_score(df["label"],pred,zero_division=0)
        if best is None or score>best["f1"]:
            best={"alpha_rf":float(round(alpha,4)),"alpha_distilbert":float(round(1-alpha,4)),"f1":float(score)}
    return best

def m(df,col,threshold):
    prob=df[col].values; pred=(prob>=threshold).astype(int)
    return classification_metrics(df["label"],pred,prob)

def main():
    cfg=load_config(); ensure_dirs()
    e=cfg["ensemble"]; threshold=float(e["threshold"])
    val,test=load_probs("val"),load_probs("test")
    best=choose_weight(val,float(e["weight_step"]),threshold)
    a=best["alpha_rf"]
    for df in [val,test]:
        df["ensemble_prob"]=a*df["rf_prob"]+(1-a)*df["distilbert_prob"]
    test["ensemble_50_50_prob"]=0.5*test["rf_prob"]+0.5*test["distilbert_prob"]

    metrics={
        "random_forest":m(test,"rf_prob",threshold),
        "distilbert":m(test,"distilbert_prob",threshold),
        "ensemble_50_50":m(test,"ensemble_50_50_prob",threshold),
        "ensemble_weighted":m(test,"ensemble_prob",threshold),
    }
    save_json({"chosen_on_validation":best,"threshold":threshold,"test":metrics},ROOT/"results"/"ensemble_metrics.json")
    test.to_csv(ROOT/"results"/"test_predictions.csv",index=False)

    rows=[]
    for name,vals in metrics.items():
        rows.append({"model":name,**{k:v for k,v in vals.items() if k!="confusion_matrix"}})
    comp=pd.DataFrame(rows)
    comp.to_csv(ROOT/"results"/"model_comparison.csv",index=False)

    cols=["accuracy","precision","recall","f1","balanced_accuracy","roc_auc"]
    ax=comp.set_index("model")[cols].plot(kind="bar",figsize=(12,6))
    ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("Comparação dos modelos")
    plt.xticks(rotation=15,ha="right"); plt.tight_layout()
    plt.savefig(ROOT/"results"/"model_comparison.png",dpi=180); plt.close()

    (ROOT/"models"/"ensemble_config.json").write_text(json.dumps({
        "alpha_rf":a,"alpha_distilbert":1-a,"threshold":threshold
    },indent=2),encoding="utf-8")
    print(comp.to_string(index=False))
    print("\nPeso escolhido:",best)

if __name__=="__main__":
    main()
