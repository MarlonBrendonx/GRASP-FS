import pdb
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from logger import configlogger

logger = configlogger(__name__)


def grasp_fs(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    max_iter=100,
    alpha=0.3,
    f1_threshold=0.95,
):

    best_features = []
    best_f1 = -1

    rcl = InformationGain(X_train, y_train)

    for i in range(max_iter):
        logger.warning(f"[*] ITERATION: {i}")

        solution = build_solution(rcl, len(rcl), alpha)

        local_solution = local_search(X_train, y_train, X_test, rcl, y_test, solution)

        f1 = evaluate_solution(X_train, y_train, X_test, y_test, local_solution)

        if f1 > best_f1:
            best_f1 = f1
            best_features = local_solution

        if best_f1 >= f1_threshold:
            break

    best_feature_names = [feature_names[i] for i in best_features]
    return best_feature_names, best_f1


def InformationGain(X, y):
    selector = SelectKBest(score_func=mutual_info_classif, k=30)
    selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)

    return X.columns[selected_features]


def build_solution(rcl, num_features, alpha):
    num_selected = int(num_features * alpha)
    selected_features = np.random.choice(num_features, size=num_selected, replace=False)
    logger.info(f"[*] SUBSET: {rcl[selected_features]}\n")

    return selected_features


def local_search(X_train, y_train, X_test, rcl, y_test, solution):
    logger.info(f"[*] LOCAL SEARCH\n")

    best_solution = solution

    best_f1 = evaluate_solution(X_train, y_train, X_test, y_test, best_solution)
    logger.info(f"[*] BEST F1: {best_f1}\n")

    rclIndexs = [i for i, _ in enumerate(rcl)]
    rclAux = np.setdiff1d(rclIndexs, solution)

    for i in range(len(solution)):
        for j in range(len(rclAux)):

            new_solution = np.setdiff1d(solution, [solution[i]])
            new_solution = np.append(new_solution, rclAux[j])
            

            new_f1 = evaluate_solution(X_train, y_train, X_test, y_test, new_solution)
            logger.info(f"[*] NEW SOLUTION: {X_train.columns[new_solution]}\n")
            logger.info(f"[*] NEW F1: {new_f1}\n")

            if new_f1 > best_f1:
                best_solution = new_solution
                best_f1 = new_f1

    return best_solution


def evaluate_solution(X_train, y_train, X_test, y_test, solution):

    X_train_selected = X_train.iloc[:, solution]
    X_test_selected = X_test.iloc[:, solution]

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_selected.values, y_train)

    y_pred = model.predict(X_test_selected.values)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return f1


df = pd.read_csv(r"ERENO-2.0-100K.csv", header=0, low_memory=False)


for col in df.columns:
    if col == "class":
        continue
    df[col] = pd.to_numeric(df[col], errors="ignore")

print("DTypes finais:\n", df.dtypes.value_counts())

# ------------------ 2. Separação de X e y ------------------

y = df["class"].astype(str)  # string para o label encoder
X = df.drop(columns=["class"])

# ------------------ 3. Identificação de numérico vs categórico ------------------

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(
    f"{len(numeric_cols)} colunas numéricas e {len(categorical_cols)} colunas categóricas"
)

# ------------------ 4. Encoding ------------------


le = LabelEncoder()
y_enc = le.fit_transform(y)


ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = X[categorical_cols].fillna("##MISSING##")
X_cat_enc = pd.DataFrame(
    ord_enc.fit_transform(X_cat), columns=categorical_cols, index=X.index
)

X_num = X[numeric_cols]
X_proc = pd.concat([X_num, X_cat_enc], axis=1)
print("Dimensão de X após encoding:", X_proc.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_enc, test_size=0.2, random_state=42
)


best_feature_names, best_f1 = grasp_fs(
    X_train,
    y_train,
    X_test,
    y_test,
    df.drop("class", axis=1).columns.to_list(),
    max_iter=100,
    alpha=0.9,
    f1_threshold=0.95,
)


print("Melhores features selecionadas:", best_feature_names)
print("Melhor F1-Score:", best_f1)


logger.info(f"------------------RESULTS------------------------\n")
logger.info(f"[*] Best features {best_feature_names}\n")
logger.info(f"[*] Best F1-Score {best_f1}\n")
