# GRASP-FS
Implementation of the GRASP-FS metaheuristic algorithm
---
## 🚀 Tecnologias e Bibliotecas

- Python 3.8+
- Numpy
- Pandas
- Scikit-learn
- XGBoost
- Logging personalizado (`logger.py`)

---

## 📊 Descrição do Algoritmo

O pipeline executa os seguintes passos:

1. **Leitura e pré-processamento dos dados**
   - Conversão de colunas não numéricas para valores categóricos.
   - Encoding de variáveis categóricas com `OrdinalEncoder`.
   - Encoding da variável target com `LabelEncoder`.

2. **Ganho de informação e criação do RCL (Restricted Candidate List)**
   - Fase inicial de **filtragem por Information Gain (mutual_info_classif)** selecionando os 30 melhores atributos para corte inicial.

3. **Wrapper**
- Aplicação do algoritmo com dois passos principais:
     - **Construção da solução:** seleciona um subconjunto do RCL dado um parâmetro de estimativa inicial de quantidade.
     - **Busca local:** explora vizinhos da solução atual trocando atributos do subconjunto com o RCL para maximizar o F1-Score.
   - O critério de parada é atingir um F1-Score mínimo (`f1_threshold`) ou o número máximo de iterações (`max_iter`).

3. **Classificação**
   - Modelo **XGBoostClassifier** é treinado e avaliado a cada iteração.

4. **Log dos resultados**
   - Os resultados são registrados via logger customizado (`logger.py`).

---

## 🔧 Instalação

### 1. Clone o repositório

```bash
git clone git@github.com:MarlonBrendonx/GRASP-FS.git
cd GRASP-FS
```
### 2. Crie um ambiente virtual (Opcional)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 3. Execução

```bash
python index.py
```
