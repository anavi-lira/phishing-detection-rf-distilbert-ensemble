from src.prepare_data import prepare_data
from src.train_rf import train_random_forest
from src.train_distilbert import train_distilbert
from src.ensemble import main as build_ensemble

if __name__=="__main__":
    print("[1/4] Preparando dados"); prepare_data()
    print("[2/4] Treinando Random Forest"); train_random_forest()
    print("[3/4] Fine-tuning DistilBERT"); train_distilbert()
    print("[4/4] Ensemble"); build_ensemble()
    print("Concluído. Use: streamlit run app.py")
