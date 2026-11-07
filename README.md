# Phishing Detection with Random Forest, DistilBERT and Weighted Ensemble

Projeto desenvolvido para avaliação de técnicas de Inteligência Artificial aplicadas à detecção de e-mails de phishing.

O trabalho foi inspirado no artigo:

**Tarapiah et al. (2025). _Evaluating the Effectiveness of Large Language Models (LLMs) Versus Machine Learning (ML) in Identifying and Detecting Phishing Email Attempts_. Algorithms, 18(10), 599.**

## Objetivo

Comparar o desempenho de duas abordagens diferentes para classificação de e-mails:

- Random Forest com TF-IDF;
- DistilBERT com transfer learning.

Como modificação em relação ao artigo original, também foi implementado um **ensemble híbrido**, combinando as probabilidades geradas pelos dois modelos.

A ideia foi verificar se a combinação entre um modelo tradicional de Machine Learning e um Transformer poderia apresentar desempenho superior ao uso individual dos modelos.

## Dataset

Foi utilizado o dataset público:

`zefang-liu/phishing-email-dataset`

disponível no Hugging Face.

Para reduzir o custo computacional do experimento, foi utilizado um subconjunto de **5.000 e-mails**.

A divisão utilizada foi:

- 60% para treinamento;
- 20% para validação;
- 20% para teste.

As classes consideradas foram:

- `0` — e-mail legítimo;
- `1` — phishing.

## Random Forest

Para o Random Forest foi utilizado:

- limpeza básica dos textos;
- TF-IDF;
- unigramas e bigramas;
- Grid Search;
- validação cruzada com 3 folds.

Os hiperparâmetros foram escolhidos utilizando o F1-score como métrica principal.

## DistilBERT

Foi utilizado o modelo:

`distilbert-base-uncased`

O modelo foi ajustado por meio de fine-tuning para a classificação binária entre e-mails legítimos e phishing.

Configuração utilizada:

- 3 épocas;
- batch size 16;
- learning rate `2e-5`;
- weight decay `0.01`;
- máximo de 256 tokens.

## Ensemble proposto

A principal modificação do projeto foi a combinação das previsões do Random Forest e do DistilBERT.

A probabilidade final foi calculada por:

```text
P_final = α * P_RF + (1 - α) * P_DistilBERT
```

O valor de `α` foi escolhido automaticamente no conjunto de validação, buscando o maior F1-score.

Também foi avaliado um ensemble simples com pesos 50/50.

## Resultados

Os resultados obtidos no conjunto de teste foram:

| Modelo | Accuracy | Precision | Recall | F1-score | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9660 | 0.9472 | 0.9625 | 0.9548 | 0.9653 | 0.9950 |
| DistilBERT | 0.9790 | 0.9809 | 0.9625 | 0.9716 | 0.9757 | 0.9979 |
| Ensemble 50/50 | 0.9810 | **0.9836** | 0.9651 | 0.9743 | 0.9778 | 0.9990 |
| **Ensemble ponderado** | **0.9860** | 0.9787 | **0.9839** | **0.9813** | **0.9856** | 0.9988 |

O **ensemble ponderado apresentou o melhor desempenho geral**, obtendo os maiores valores de:

- Accuracy;
- Recall;
- F1-score;
- Balanced Accuracy.

Isso indica que a combinação entre Random Forest e DistilBERT conseguiu aproveitar características complementares dos dois modelos.

## Estrutura do experimento

```text
E-mail
   |
   +----> TF-IDF ----> Random Forest ----+
   |                                     |
   |                                     +----> Ensemble ----> Resultado final
   |                                     |
   +----> Tokenizer ----> DistilBERT ----+
```

## Métricas utilizadas

Foram utilizadas as seguintes métricas:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- Balanced Accuracy;
- ROC-AUC;
- Matriz de confusão.

## Execução no Google Colab

O experimento foi desenvolvido e executado utilizando Google Colab com GPU T4.

Para executar o notebook:

1. Faça upload do arquivo `.ipynb` no Google Colab.
2. Ative a GPU em:
   `Ambiente de execução → Alterar tipo de ambiente de execução → GPU`.
3. Execute todas as células em sequência.
4. Ao final serão geradas as métricas, gráficos e matrizes de confusão.

## Arquivos gerados

Ao final da execução são produzidos arquivos como:

```text
model_comparison.csv
model_comparison.png
confusion_matrices.png
test_predictions.csv
ensemble_config.json
```

## Conclusão

O experimento mostrou que a combinação entre Random Forest e DistilBERT apresentou desempenho superior ao uso isolado dos modelos em várias métricas.

O ensemble ponderado alcançou:

- **98,6% de acurácia**;
- **98,39% de recall**;
- **98,13% de F1-score**;
- **98,56% de balanced accuracy**.

Os resultados indicam que a utilização conjunta de modelos tradicionais de Machine Learning e modelos baseados em Transformers pode ser uma alternativa interessante para detecção de phishing.

## Referência

Tarapiah, S.; Abbas, L.; Mardawi, O.; Atalla, S.; Himeur, Y.; Mansoor, W.  
**Evaluating the Effectiveness of Large Language Models (LLMs) Versus Machine Learning (ML) in Identifying and Detecting Phishing Email Attempts.**  
Algorithms, 18(10), 599, 2025.

DOI: `10.3390/a18100599`
