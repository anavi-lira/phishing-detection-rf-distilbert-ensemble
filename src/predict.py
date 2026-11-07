from __future__ import annotations
import json, joblib, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils import ROOT

class HybridPhishingDetector:
    def __init__(self):
        self.rf=joblib.load(ROOT/"models"/"random_forest.joblib")
        model_dir=ROOT/"models"/"distilbert"
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_dir))
        self.db=AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db.to(self.device).eval()
        self.cfg=json.loads((ROOT/"models"/"ensemble_config.json").read_text(encoding="utf-8"))

    def rf_probability(self,text):
        return float(self.rf.predict_proba([text])[:,1][0])

    @torch.no_grad()
    def db_probability(self,text):
        x=self.tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=256)
        x={k:v.to(self.device) for k,v in x.items()}
        return float(torch.softmax(self.db(**x).logits,dim=1)[0,1].item())

    def predict(self,text):
        prf=self.rf_probability(text); pdb=self.db_probability(text)
        p=self.cfg["alpha_rf"]*prf+self.cfg["alpha_distilbert"]*pdb
        label=int(p>=self.cfg["threshold"])
        return {
            "random_forest_probability":prf,
            "distilbert_probability":pdb,
            "ensemble_probability":p,
            "prediction":"PHISHING" if label else "LEGITIMO",
        }

if __name__=="__main__":
    d=HybridPhishingDetector()
    print("Cole o e-mail e finalize com uma linha contendo apenas FIM:")
    lines=[]
    while True:
        line=input()
        if line.strip()=="FIM": break
        lines.append(line)
    r=d.predict("\n".join(lines))
    print(f"Random Forest: {r['random_forest_probability']:.2%}")
    print(f"DistilBERT:    {r['distilbert_probability']:.2%}")
    print(f"Ensemble:      {r['ensemble_probability']:.2%}")
    print(f"Classificacao: {r['prediction']}")
