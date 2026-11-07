# Detector Híbrido de Phishing — Random Forest + DistilBERT

Miniprojeto inspirado em:

**Tarapiah, S.; Abbas, L.; Mardawi, O.; Atalla, S.; Himeur, Y.; Mansoor, W.**
*Evaluating the Effectiveness of Large Language Models (LLMs) Versus Machine Learning (ML) in Identifying and Detecting Phishing Email Attempts.*
Algorithms, 18(10), 599, 2025.
DOI: https://doi.org/10.3390/a18100599

## Proposta

O artigo compara modelos clássicos de ML e Transformers para phishing e indica que uma abordagem híbrida/ensemble pode ser uma direção promissora.

Este repositório implementa:

1. **Random Forest + TF-IDF**
2. **DistilBERT com transfer learning**
3. **Ensemble 50/50**
4. **Ensemble ponderado proposto**

A modificação principal é o **ensemble entre um modelo tradicional (Random Forest) e um Transformer (DistilBERT)**.

A probabilidade final é:

`P(phishing) = α * P_RF + (1 - α) * P_DistilBERT`

O peso `α` é escolhido automaticamente no conjunto de validação para maximizar o F1-score. Assim, o teste final continua separado.

## Dataset

Por padrão é usada a base pública:

`zefang-liu/phishing-email-dataset`

Ela possui os campos `Email Text` e `Email Type` e é uma das fontes públicas citadas no artigo-base.

Para o miniprojeto, o código seleciona **5.000 e-mails por amostragem estratificada** e divide em:

- 60% treino
- 20% validação
- 20% teste

Todos os modelos usam exatamente os mesmos conjuntos.

## Random Forest

Pré-processamento:
- remoção de HTML;
- minúsculas;
- URLs substituídas por token;
- e-mails substituídos por token;
- caracteres especiais removidos;
- TF-IDF;
- unigramas + bigramas;
- stop words em inglês removidas.

Grid Search:
- `n_estimators`: 200, 400
- `max_depth`: None, 30
- `min_samples_split`: 2, 5
- `class_weight`: None, balanced
- 3-fold cross-validation
- seleção por F1

## DistilBERT

- base: `distilbert-base-uncased`
- transfer learning: sim
- max length: 256
- batch size: 16
- épocas: 3
- learning rate: 2e-5
- weight decay: 0.01
- dropout: 0.1
- melhor checkpoint selecionado pelo F1 na validação

## Métricas

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC
- Matriz de confusão

## Instalação

Python 3.11 recomendado.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Depois:

```bash
pip install -r requirements.txt
```

## Executar tudo

```bash
python run_all.py
```

A primeira execução baixa o dataset e o DistilBERT.

> Para o fine-tuning do DistilBERT, GPU é recomendada. Google Colab é uma boa opção caso o computador não tenha GPU adequada.

## Resultados

Os arquivos aparecem em `results/`:

- `random_forest_metrics.json`
- `distilbert_metrics.json`
- `ensemble_metrics.json`
- `model_comparison.csv`
- `model_comparison.png`
- `test_predictions.csv`

Os modelos ficam em `models/`.

## Demonstração

Depois do treinamento:

```bash
streamlit run app.py
```

A tela permite colar um e-mail e comparar:
- probabilidade do Random Forest;
- probabilidade do DistilBERT;
- resultado final do ensemble.

Também é possível usar o terminal:

```bash
python -m src.predict
```

## Relação com os critérios do miniprojeto

| Critério | Projeto |
|---|---|
| Base | Phishing Email Dataset |
| População de enxame | Não se aplica |
| Treino/validação/teste | 60/20/20 |
| Validação cruzada | 3-fold no RF |
| Algoritmos | Random Forest e DistilBERT |
| Hiperparâmetros | Grid Search no RF |
| Transfer learning | Sim |
| Regularização | Dropout + weight decay |
| Aumento de dados | Não |
| Desbalanceamento | Split estratificado + `class_weight` testado |
| Métricas | Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC |
| Modificação | Ensemble RF + DistilBERT |

## Observação

Este miniprojeto é uma reprodução reduzida e uma extensão do artigo, e não uma reprodução integral.
O artigo trabalha com diversas bases, modelos e URLs; aqui o foco é uma comparação reproduzível entre dois modelos e a proposta híbrida.

## Fonte do dataset

https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
