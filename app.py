import streamlit as st
from src.predict import HybridPhishingDetector

st.set_page_config(page_title="Detector Híbrido de Phishing",page_icon="📧")
st.title("📧 Detector Híbrido de Phishing")
st.write("Random Forest + DistilBERT + Ensemble ponderado")

@st.cache_resource
def load_detector():
    return HybridPhishingDetector()

try:
    detector=load_detector()
except Exception as exc:
    st.error("Treine os modelos primeiro com: python run_all.py")
    st.exception(exc); st.stop()

text=st.text_area("Cole o conteúdo do e-mail",height=250)
if st.button("Analisar",type="primary",use_container_width=True):
    if not text.strip():
        st.warning("Digite o texto de um e-mail.")
    else:
        r=detector.predict(text)
        c1,c2,c3=st.columns(3)
        c1.metric("Random Forest",f"{r['random_forest_probability']:.1%}")
        c2.metric("DistilBERT",f"{r['distilbert_probability']:.1%}")
        c3.metric("Ensemble",f"{r['ensemble_probability']:.1%}")
        st.error("PHISHING") if r["prediction"]=="PHISHING" else st.success("LEGÍTIMO")
