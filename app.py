"""
╔══════════════════════════════════════════════════════════════════╗
║  EduRisco — Sistema Inteligente de Previsão de Desempenho Escolar ║
║  Prefeitura Municipal · Dataset: Student Alcohol Consumption      ║
║  Autores: Pedro, Thiago e equipe                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings('ignore')

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EduRisco · Previsão Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CSS — IDENTIDADE VISUAL
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.edurisco-header {
    background: linear-gradient(135deg, #0A2540 0%, #0D3B66 55%, #1565A8 100%);
    border-radius: 16px; padding: 2.2rem 2.8rem; margin-bottom: 1.8rem;
    color: white; position: relative; overflow: hidden;
}
.edurisco-header::after {
    content: '🎓'; position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%); font-size: 7rem; opacity: 0.07;
}
.edurisco-header h1 { font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -0.4px; }
.edurisco-header .sub  { margin: 0.35rem 0 0; opacity: 0.75; font-size: 0.9rem; font-weight: 300; }
.edurisco-header .tags { margin-top: 0.8rem; }
.edurisco-header .tag  {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25); border-radius: 20px;
    padding: 0.15rem 0.75rem; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.5px; margin-right: 0.4rem; color: #B8D4F8;
}

.m-card {
    background: #fff; border-radius: 14px; padding: 1.4rem 1.2rem;
    border: 1px solid #E2EAF4; border-top: 4px solid #1A6BA0;
    box-shadow: 0 2px 10px rgba(13,59,102,0.07); text-align: center;
}
.m-card.vermelho { border-top-color: #E74C3C; }
.m-card .val {
    font-size: 2rem; font-weight: 800; color: #0D3B66;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.m-card .val.verm { color: #E74C3C; }
.m-card .lbl {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #64748B; margin-top: 0.4rem;
}
.m-card .sub-lbl { font-size: 0.72rem; color: #94A3B8; margin-top: 0.2rem; }

.res-alto {
    background: linear-gradient(135deg, #FFF5F5, #FED7D7);
    border: 2px solid #E74C3C; border-radius: 16px; padding: 2rem 2.5rem;
}
.res-baixo {
    background: linear-gradient(135deg, #F0FFF4, #C6F6D5);
    border: 2px solid #27AE60; border-radius: 16px; padding: 2rem 2.5rem;
}
.res-alto  h2 { color: #C53030; font-size: 1.5rem; margin: 0 0 0.4rem; }
.res-baixo h2 { color: #1A5C36; font-size: 1.5rem; margin: 0 0 0.4rem; }
.res-alto  p  { color: #742A2A; margin: 0; }
.res-baixo p  { color: #1C4532; margin: 0; }

.barra-wrap { background: #E2E8F0; border-radius: 999px; height: 16px; overflow: hidden; margin: 0.8rem 0 0.3rem; }
.barra-fill { height: 100%; border-radius: 999px; }

.fator {
    background: #F8FAFC; border-left: 4px solid #CBD5E1;
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
}
.fator.alto  { border-left-color: #E74C3C; background: #FFF8F8; }
.fator.medio { border-left-color: #F39C12; background: #FFFBF0; }
.fator.baixo { border-left-color: #2ECC71; background: #F0FFF4; }
.fator .ft   { font-weight: 700; font-size: 0.9rem; }
.fator .fd   { font-size: 0.82rem; color: #475569; margin-top: 0.15rem; }

.sec-titulo {
    font-size: 1rem; font-weight: 700; color: #0D3B66;
    border-bottom: 2px solid #E2EAF4; padding-bottom: 0.5rem;
    margin: 1.6rem 0 1rem; letter-spacing: -0.2px;
}

.etico {
    background: #FFFBEB; border: 1px solid #FCD34D;
    border-left: 4px solid #F59E0B; border-radius: 8px;
    padding: 1rem 1.2rem; font-size: 0.82rem; color: #78350F; line-height: 1.6;
}

[data-testid="stSidebar"] { background: #0A2540 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #93C5FD !important; }

.stButton > button {
    background: linear-gradient(135deg, #0D3B66, #1A6BA0) !important;
    color: white !important; font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important; width: 100% !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
NOTA_CORTE = 10
DATA_DIR   = "data"

MERGE_COLS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]

COLS_CAT = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic"
]

LABEL_MAP = {
    "Mjob":     {"teacher":"Professora","health":"Saúde","services":"Serviços","at_home":"Em casa","other":"Outro"},
    "Fjob":     {"teacher":"Professor","health":"Saúde","services":"Serviços","at_home":"Em casa","other":"Outro"},
    "reason":   {"home":"Proximidade","reputation":"Reputação","course":"Curso","other":"Outro"},
    "guardian": {"mother":"Mãe","father":"Pai","other":"Outro"},
}

PALETTE = {
    "fundo":"#F4F7FC", "azul":"#1A6BA0", "verde":"#27AE60",
    "vermelho":"#E74C3C", "amarelo":"#F39C12", "cinza":"#94A3B8"
}


# ══════════════════════════════════════════════════════════════════
# CARREGAMENTO E MERGE
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📂 Carregando datasets...")
def carregar_dados(modo: str = "mat"):
    mat_path = os.path.join(DATA_DIR, "student-mat.csv")
    por_path = os.path.join(DATA_DIR, "student-por.csv")

    df_mat = pd.read_csv(mat_path, sep=",")
    df_por = pd.read_csv(por_path, sep=",")

    if modo == "mat":
        df = df_mat.copy()
    elif modo == "por":
        df = df_por.copy()
    else:
        # Replica exatamente o student-merge.R
        df = pd.merge(df_mat, df_por, on=MERGE_COLS, suffixes=("_mat", "_por"))
        # Usar colunas de matemática como base
        rename_mat = {c: c.replace("_mat", "") for c in df.columns if c.endswith("_mat")}
        df = df.rename(columns=rename_mat)
        # Dropar duplicatas de português (mantém G3_por para comparação opcional)
        cols_drop = [c for c in df.columns if c.endswith("_por") and c != "G3_por"]
        df = df.drop(columns=cols_drop)

    df["risco"] = (df["G3"] < NOTA_CORTE).astype(int)
    return df, df_mat, df_por


# ══════════════════════════════════════════════════════════════════
# PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════

def preprocessar(df: pd.DataFrame):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    for col in df.columns:
        if df[col].dtype == object:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    cols_remover = ["G1", "G2", "G3", "risco"]
    if "G3_por" in df.columns:
        cols_remover.append("G3_por")
    cols_remover = [c for c in cols_remover if c in df.columns]

    X = df.drop(columns=cols_remover)
    y = df["risco"].values

    encoders = {}
    for col in COLS_CAT:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    feature_cols = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders, feature_cols, df

#Par 1 =============================
#Part 1 =====================================================================
# ══════════════════════════════════════════════════════════════════
# MODELO MLP
# ══════════════════════════════════════════════════════════════════

def construir_mlp(input_dim: int) -> Sequential:
    modelo = Sequential(name="EduRisco_MLP")
    # Camada de entrada
    modelo.add(Dense(128, activation="relu", input_dim=input_dim))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    # 1ª camada oculta
    modelo.add(Dense(64, activation="relu"))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.3))
    # 2ª camada oculta
    modelo.add(Dense(32, activation="relu"))
    modelo.add(Dropout(0.2))
    # Saída — classificação binária
    modelo.add(Dense(1, activation="sigmoid"))

    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


@st.cache_resource(show_spinner="🤖 Treinando Rede Neural MLP...")
def treinar(_X, _y, _modo):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    modelo = construir_mlp(input_dim=X_train.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=20,
                       restore_best_weights=True, verbose=0)

    hist = modelo.fit(
        X_train, y_train,
        epochs=200, batch_size=32,
        validation_split=0.15,
        callbacks=[es], verbose=0
    )

    y_prob = modelo.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metricas = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "auc"      : roc_auc_score(y_test, y_prob),
        "cm"       : confusion_matrix(y_test, y_pred),
        "report"   : classification_report(y_test, y_pred,
                         target_names=["Baixo Risco", "Alto Risco"]),
        "hist"     : hist.history,
        "epocas"   : len(hist.history["loss"]),
    }
    return modelo, metricas

#part 2 =================================
# ══════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════

def _fig_base(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax


def fig_distribuicao_notas(df):
    vc = df["G3"].value_counts().sort_index()
    cores = [PALETTE["vermelho"] if g < NOTA_CORTE else PALETTE["verde"] for g in vc.index]
    fig, ax = _fig_base(9, 4)
    ax.bar(vc.index, vc.values, color=cores, edgecolor="white", linewidth=0.7, width=0.85)
    ax.axvline(NOTA_CORTE - 0.5, color="#1A202C", linestyle="--", linewidth=1.8)
    ax.set_xlabel("Nota Final (G3)", fontsize=11, color="#0D3B66", fontweight="600")
    ax.set_ylabel("Nº de Alunos",    fontsize=11, color="#0D3B66", fontweight="600")
    ax.set_title("Distribuição das Notas Finais", fontsize=13, fontweight="800", color="#0D3B66")
    patch_r = mpatches.Patch(color=PALETTE["vermelho"], label=f"Alto Risco (G3 < {NOTA_CORTE})")
    patch_g = mpatches.Patch(color=PALETTE["verde"],   label=f"Baixo Risco (G3 ≥ {NOTA_CORTE})")
    ax.legend(handles=[patch_r, patch_g], fontsize=9)
    plt.tight_layout()
    return fig


def fig_pizza_risco(df):
    vc = df["risco"].value_counts().sort_index()
    labels = [f"Baixo Risco\n(G3 ≥ {NOTA_CORTE})", f"Alto Risco\n(G3 < {NOTA_CORTE})"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.patch.set_facecolor(PALETTE["fundo"])
    wedges, texts, autotexts = ax.pie(
        vc, labels=labels,
        colors=[PALETTE["verde"], PALETTE["vermelho"]],
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2.5),
        textprops=dict(fontsize=9)
    )
    for at in autotexts:
        at.set_fontweight("800"); at.set_fontsize(11)
    ax.set_title("Proporção de Alunos em Risco", fontsize=12, fontweight="800", color="#0D3B66")
    plt.tight_layout()
    return fig
#part 3 ============================================

def fig_boxplot_fatores(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor(PALETTE["fundo"])
    vars_plot = [
        ("failures", "Reprovações Anteriores"),
        ("studytime","Tempo de Estudo (1–4)"),
        ("absences", "Nº de Faltas"),
    ]
    for i, (col, titulo) in enumerate(vars_plot):
        axes[i].set_facecolor(PALETTE["fundo"])
        axes[i].spines[["top","right"]].set_visible(False)
        grupos = [df[df[col] == v]["G3"].dropna().values for v in sorted(df[col].unique())]
        labels = sorted(df[col].unique())
        axes[i].boxplot(grupos, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor="#DBEAFE", color="#1A6BA0"),
                        medianprops=dict(color=PALETTE["vermelho"], linewidth=2),
                        whiskerprops=dict(color="#64748B"),
                        capprops=dict(color="#64748B"),
                        flierprops=dict(marker="o", color=PALETTE["cinza"], markersize=3))
        axes[i].axhline(NOTA_CORTE, color=PALETTE["vermelho"], linestyle="--", linewidth=1.3, alpha=0.7)
        axes[i].set_title(titulo, fontsize=10, fontweight="700", color="#0D3B66")
        axes[i].set_ylabel("Nota G3", fontsize=9)
    plt.suptitle("Impacto dos Principais Fatores na Nota Final (G3)",
                 fontsize=12, fontweight="800", color="#0D3B66", y=1.02)
    plt.tight_layout()
    return fig

#Part 4 ========================================

def fig_correlacoes(df):
    cols_num = ["age","Medu","Fedu","traveltime","studytime","failures",
                "famrel","freetime","goout","Dalc","Walc","health","absences","G3","risco"]
    cols_num = [c for c in cols_num if c in df.columns]
    corr = df[cols_num].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(PALETTE["fundo"])
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                square=True, linewidths=0.6, cbar_kws={"shrink": 0.65}, ax=ax)
    ax.set_title("Matriz de Correlação — Fatores de Risco vs Desempenho",
                 fontsize=12, fontweight="800", color="#0D3B66", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return fig


def fig_alcool_desempenho(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(PALETTE["fundo"])
    for ax, (col, titulo) in zip(axes, [("Dalc","Dias Úteis"), ("Walc","Fim de Semana")]):
        ax.set_facecolor(PALETTE["fundo"])
        ax.spines[["top","right"]].set_visible(False)
        medias = df.groupby(col)["G3"].mean()
        cores = [PALETTE["vermelho"] if m < NOTA_CORTE else PALETTE["azul"] for m in medias.values]
        ax.bar(medias.index, medias.values, color=cores, edgecolor="white", linewidth=0.7)
        ax.axhline(NOTA_CORTE, color=PALETTE["vermelho"], linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel(f"Consumo de Álcool ({titulo}) · 1=Baixo, 5=Alto", fontsize=9)
        ax.set_ylabel("Média Nota G3", fontsize=9)
        ax.set_title(f"Álcool ({titulo}) vs Desempenho", fontsize=10, fontweight="700", color="#0D3B66")
        ax.set_xticks([1,2,3,4,5])
    plt.suptitle("Impacto do Consumo de Álcool na Nota Final",
                 fontsize=12, fontweight="800", color="#0D3B66")
    plt.tight_layout()
    return fig


def fig_comparativo_datasets(df_mat, df_por):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(PALETTE["fundo"])
    for ax, (df_, label, cor) in zip(axes, [
        (df_mat, "Matemática", PALETTE["azul"]),
        (df_por, "Português",  PALETTE["verde"])
    ]):
        ax.set_facecolor(PALETTE["fundo"])
        ax.spines[["top","right"]].set_visible(False)
        vc = df_["G3"].value_counts().sort_index()
        cores = [PALETTE["vermelho"] if g < NOTA_CORTE else cor for g in vc.index]
        ax.bar(vc.index, vc.values, color=cores, edgecolor="white", linewidth=0.7)
        ax.axvline(NOTA_CORTE - 0.5, color="#1A202C", linestyle="--", linewidth=1.5)
        risco_pct = (df_["G3"] < NOTA_CORTE).mean()
        ax.set_title(f"{label} · {len(df_)} alunos · {risco_pct:.1%} em risco",
                     fontsize=10, fontweight="700", color="#0D3B66")
        ax.set_xlabel("Nota G3", fontsize=9); ax.set_ylabel("Alunos", fontsize=9)
    plt.suptitle("Comparativo: Distribuição de Notas por Disciplina",
                 fontsize=12, fontweight="800", color="#0D3B66")
    plt.tight_layout()
    return fig


def fig_curvas_treino(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.patch.set_facecolor(PALETTE["fundo"])
    configs = [
        ("accuracy","val_accuracy","Acurácia por Época", PALETTE["azul"],    PALETTE["verde"]),
        ("loss",    "val_loss",    "Loss por Época",     PALETTE["vermelho"],PALETTE["amarelo"]),
    ]
    for ax, (tr, val, titulo, c1, c2) in zip(axes, configs):
        ax.set_facecolor(PALETTE["fundo"])
        ax.spines[["top","right"]].set_visible(False)
        ax.plot(hist[tr],  color=c1, linewidth=2, label="Treino")
        ax.plot(hist[val], color=c2, linewidth=2, linestyle="--", label="Validação")
        ax.set_title(titulo, fontsize=11, fontweight="700", color="#0D3B66")
        ax.set_xlabel("Épocas"); ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def fig_matriz_confusao(cm):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.patch.set_facecolor(PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Baixo Risco","Alto Risco"],
                yticklabels=["Baixo Risco","Alto Risco"],
                linewidths=2, linecolor="white", ax=ax,
                annot_kws={"size": 15, "fontweight": "bold"})
    ax.set_xlabel("Previsto", fontsize=11, fontweight="600", color="#0D3B66")
    ax.set_ylabel("Real",     fontsize=11, fontweight="600", color="#0D3B66")
    ax.set_title("Matriz de Confusão", fontsize=12, fontweight="800", color="#0D3B66")
    plt.tight_layout()
    return fig

#Part 5 ========================================
# ══════════════════════════════════════════════════════════════════
# INTERPRETABILIDADE
# ══════════════════════════════════════════════════════════════════

def fatores_de_risco(inp: dict) -> list:
    fatores = []

    if inp.get("failures", 0) >= 2:
        fatores.append({"nivel":"alto","titulo":"🔴 Múltiplas reprovações anteriores",
            "desc":f"{int(inp['failures'])} reprovação(ões) — fator de maior impacto negativo no modelo"})
    elif inp.get("failures", 0) == 1:
        fatores.append({"nivel":"medio","titulo":"🟠 Reprovação anterior registrada",
            "desc":"1 reprovação — aumenta significativamente a probabilidade de risco"})

    if inp.get("absences", 0) > 20:
        fatores.append({"nivel":"alto","titulo":"🔴 Taxa de faltas crítica",
            "desc":f"{int(inp['absences'])} faltas — fortemente correlacionado com baixo G3"})
    elif inp.get("absences", 0) > 10:
        fatores.append({"nivel":"medio","titulo":"🟠 Faltas acima da média",
            "desc":f"{int(inp['absences'])} faltas — atenção à frequência escolar"})

    if inp.get("Dalc", 1) >= 4:
        fatores.append({"nivel":"alto","titulo":"🔴 Alto consumo de álcool (dias úteis)",
            "desc":f"Nível {inp['Dalc']}/5 — impacto direto na cognição e frequência"})
    elif inp.get("Dalc", 1) >= 3:
        fatores.append({"nivel":"medio","titulo":"🟠 Consumo moderado de álcool (dias úteis)",
            "desc":f"Nível {inp['Dalc']}/5 — monitorar"})

    if inp.get("studytime", 2) <= 1:
        fatores.append({"nivel":"medio","titulo":"🟠 Tempo de estudo insuficiente",
            "desc":"Menos de 2h/semana — abaixo do mínimo recomendado"})

    if inp.get("Walc", 1) >= 4:
        fatores.append({"nivel":"medio","titulo":"🟠 Alto consumo de álcool no fim de semana",
            "desc":f"Nível {inp['Walc']}/5 — pode comprometer o rendimento"})

    if inp.get("goout", 3) >= 4:
        fatores.append({"nivel":"medio","titulo":"🟠 Alta frequência de saídas sociais",
            "desc":f"Nível {inp['goout']}/5 — pode reduzir tempo de estudo"})

    if inp.get("famrel", 3) <= 2:
        fatores.append({"nivel":"medio","titulo":"🟠 Relação familiar comprometida",
            "desc":f"Nível {inp['famrel']}/5 — suporte familiar é fator protetor importante"})

    if inp.get("higher", 1) == "yes":
        fatores.append({"nivel":"baixo","titulo":"🟢 Aspiração ao ensino superior",
            "desc":"Aluno motivado a continuar estudando — fator protetor"})

    if inp.get("studytime", 2) >= 3:
        fatores.append({"nivel":"baixo","titulo":"🟢 Bom hábito de estudo",
            "desc":f"Nível {inp['studytime']}/4 — dedicação consistente"})

    if inp.get("Medu", 0) >= 3 or inp.get("Fedu", 0) >= 3:
        fatores.append({"nivel":"baixo","titulo":"🟢 Pais com ensino médio ou superior",
            "desc":"Escolaridade dos pais correlaciona positivamente com G3"})

    if not fatores:
        fatores.append({"nivel":"baixo","titulo":"🟢 Nenhum fator crítico identificado",
            "desc":"Perfil sem indicadores de risco relevantes detectados"})

    ordem = {"alto": 0, "medio": 1, "baixo": 2}
    fatores.sort(key=lambda x: ordem[x["nivel"]])
    return fatores


# ══════════════════════════════════════════════════════════════════
# PREPARA INPUT PARA O MODELO
# ══════════════════════════════════════════════════════════════════

def preparar_input(inp: dict, feature_cols: list, scaler: StandardScaler) -> np.ndarray:
    encode_map = {
        "school":     {"GP":0,"MS":1},
        "sex":        {"F":0,"M":1},
        "address":    {"R":0,"U":1},
        "famsize":    {"GT3":0,"LE3":1},
        "Pstatus":    {"A":0,"T":1},
        "Mjob":       {"at_home":0,"health":1,"other":2,"services":3,"teacher":4},
        "Fjob":       {"at_home":0,"health":1,"other":2,"services":3,"teacher":4},
        "reason":     {"course":0,"home":1,"other":2,"reputation":3},
        "guardian":   {"father":0,"mother":1,"other":2},
        "schoolsup":  {"no":0,"yes":1},
        "famsup":     {"no":0,"yes":1},
        "paid":       {"no":0,"yes":1},
        "activities": {"no":0,"yes":1},
        "nursery":    {"no":0,"yes":1},
        "higher":     {"no":0,"yes":1},
        "internet":   {"no":0,"yes":1},
        "romantic":   {"no":0,"yes":1},
    }
    row = {}
    for col in feature_cols:
        if col in inp:
            val = inp[col]
            if col in encode_map and isinstance(val, str):
                val = encode_map[col].get(val, 0)
            row[col] = val
        else:
            row[col] = 0

    df_in = pd.DataFrame([row], columns=feature_cols)
    return scaler.transform(df_in)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — FORMULÁRIO
# ══════════════════════════════════════════════════════════════════

def sidebar_formulario() -> dict:
    st.sidebar.markdown("## 👤 Perfil do Aluno")
    st.sidebar.caption("Preencha os dados para gerar a previsão de risco")
    st.sidebar.markdown("---")

    inp = {}

    st.sidebar.markdown("### 🏫 Dados Escolares")
    inp["school"] = "GP" if st.sidebar.selectbox(
        "Escola", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"]
    ).startswith("Gabriel") else "MS"

    inp["reason"] = st.sidebar.selectbox(
        "Motivo de escolha da escola",
        ["reputation","course","home","other"],
        format_func=lambda x: LABEL_MAP["reason"][x]
    )
    inp["studytime"] = st.sidebar.select_slider(
        "Horas de estudo semanais", [1, 2, 3, 4],
        format_func=lambda x: {1:"< 2h",2:"2–5h",3:"5–10h",4:"> 10h"}[x], value=2
    )
    inp["failures"]  = st.sidebar.slider("Reprovações anteriores", 0, 4, 0)
    inp["absences"]  = st.sidebar.slider("Número de faltas no ano", 0, 93, 4)
    inp["schoolsup"] = "yes" if st.sidebar.selectbox("Suporte pedagógico extra", ["Não","Sim"]) == "Sim" else "no"
    inp["paid"]      = "yes" if st.sidebar.selectbox("Aulas particulares pagas",  ["Não","Sim"]) == "Sim" else "no"
    inp["activities"]= "yes" if st.sidebar.selectbox("Atividades extracurriculares",["Não","Sim"]) == "Sim" else "no"
    inp["higher"]    = "yes" if st.sidebar.selectbox("Deseja cursar ensino superior",["Sim","Não"]) == "Sim" else "no"

    st.sidebar.markdown("### 👨‍👩‍👧 Contexto Familiar")
    inp["sex"]     = "F" if st.sidebar.selectbox("Sexo",["Feminino","Masculino"]) == "Feminino" else "M"
    inp["age"]     = st.sidebar.slider("Idade", 15, 22, 17)
    inp["address"] = "U" if st.sidebar.selectbox("Residência",["Urbana","Rural"]) == "Urbana" else "R"
    inp["famsize"] = "GT3" if st.sidebar.selectbox(
        "Tamanho da família",["> 3 membros (GT3)","≤ 3 membros (LE3)"]
    ).startswith(">") else "LE3"
    inp["Pstatus"] = "T" if st.sidebar.selectbox("Situação dos pais",["Juntos","Separados"]) == "Juntos" else "A"
    inp["Medu"] = st.sidebar.select_slider(
        "Escolaridade da mãe", [0,1,2,3,4],
        format_func=lambda x: {0:"Nenhuma",1:"Fund. I",2:"Fund. II",3:"Médio",4:"Superior"}[x], value=2
    )
    inp["Fedu"] = st.sidebar.select_slider(
        "Escolaridade do pai", [0,1,2,3,4],
        format_func=lambda x: {0:"Nenhuma",1:"Fund. I",2:"Fund. II",3:"Médio",4:"Superior"}[x], value=2
    )
    inp["Mjob"] = st.sidebar.selectbox(
        "Profissão da mãe", ["teacher","health","services","at_home","other"],
        format_func=lambda x: LABEL_MAP["Mjob"][x], index=4
    )
    inp["Fjob"] = st.sidebar.selectbox(
        "Profissão do pai", ["teacher","health","services","at_home","other"],
        format_func=lambda x: LABEL_MAP["Fjob"][x], index=4
    )
    inp["guardian"] = st.sidebar.selectbox(
        "Responsável principal",["mother","father","other"],
        format_func=lambda x: LABEL_MAP["guardian"][x]
    )
    inp["famrel"]   = st.sidebar.slider("Qualidade das relações familiares (1–5)", 1, 5, 4)
    inp["famsup"]   = "yes" if st.sidebar.selectbox("Suporte educacional da família",["Sim","Não"]) == "Sim" else "no"
    inp["nursery"]  = "yes" if st.sidebar.selectbox("Frequentou pré-escola",["Sim","Não"]) == "Sim" else "no"
    inp["internet"] = "yes" if st.sidebar.selectbox("Acesso à internet em casa",["Sim","Não"]) == "Sim" else "no"
    inp["traveltime"] = st.sidebar.select_slider(
        "Tempo de deslocamento", [1,2,3,4],
        format_func=lambda x: {1:"< 15 min",2:"15–30 min",3:"30–60 min",4:"> 60 min"}[x], value=1
    )

    st.sidebar.markdown("### 🎭 Vida Social e Saúde")
    inp["romantic"] = "yes" if st.sidebar.selectbox("Em relacionamento amoroso",["Não","Sim"]) == "Sim" else "no"
    inp["freetime"] = st.sidebar.slider("Tempo livre após escola (1–5)", 1, 5, 3)
    inp["goout"]    = st.sidebar.slider("Saídas com amigos (1=raramente, 5=muito)", 1, 5, 3)
    inp["Dalc"]     = st.sidebar.slider("Consumo de álcool — dias úteis (1–5)", 1, 5, 1)
    inp["Walc"]     = st.sidebar.slider("Consumo de álcool — fim de semana (1–5)", 1, 5, 2)
    inp["health"]   = st.sidebar.slider("Condição de saúde atual (1=ruim, 5=ótima)", 1, 5, 4)

    return inp

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    # ── CABEÇALHO ──────────────────────────────────────────────
    st.markdown("""
    <div class="edurisco-header">
        <h1>EduRisco — Previsão Inteligente de Desempenho Escolar</h1>
        <p class="sub">Sistema de apoio à gestão municipal · Rede Neural MLP · Python + TensorFlow + Streamlit</p>
        <div class="tags">
            <span class="tag">REDE NEURAL MLP</span>
            <span class="tag">STUDENT ALCOHOL CONSUMPTION · UCI/KAGGLE</span>
            <span class="tag">MAT · POR · MERGE (student-merge.R)</span>
            <span class="tag">LGPD COMPLIANT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SELEÇÃO DO DATASET ─────────────────────────────────────
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        modo = st.radio(
            "📁 Dataset para treinamento",
            ["mat", "por", "merged"],
            format_func=lambda x: {
                "mat":    "📐 Matemática — student-mat.csv (395 alunos)",
                "por":    "📖 Português — student-por.csv (649 alunos)",
                "merged": "🔗 Mesclado — student-merge.R (382 alunos)"
            }[x]
        )
    with col_sel2:
        st.markdown("""
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
                    padding:1rem 1.2rem;font-size:0.82rem;color:#1E40AF;line-height:1.8;margin-top:0.5rem">
        <strong>ℹ️ Sobre o student-merge.R</strong><br>
        O script R original executa <code>merge(d1, d2, by=c("school","sex","age","address","famsize",
        "Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))</code> —
        identificando os <strong>382 alunos presentes em ambas as disciplinas</strong>.
        Esta aplicação replica essa lógica exatamente com <code>pd.merge(..., on=MERGE_COLS)</code>,
        usando as notas de <strong>Matemática como variável-alvo (G3)</strong>.
        </div>
        """, unsafe_allow_html=True)

    # ── CARREGAMENTO E TREINAMENTO ─────────────────────────────
    df, df_mat, df_por = carregar_dados(modo)
    X_scaled, y, scaler, encoders, feature_cols, df_proc = preprocessar(df)
    modelo, metricas = treinar(X_scaled, y, modo)

    # ── FORMULÁRIO SIDEBAR ─────────────────────────────────────
    inputs_usuario = sidebar_formulario()
    prever = st.sidebar.button("🔍 Prever Risco do Aluno", type="primary")

    # ── CARDS MÉTRICAS ─────────────────────────────────────────
    st.markdown('<div class="sec-titulo">📊 Desempenho do Modelo de Inteligência Artificial</div>',
                unsafe_allow_html=True)

    total     = len(df_proc)
    em_risco  = int(df_proc["risco"].sum())
    pct_risco = em_risco / total

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="m-card"><div class="val">{metricas["accuracy"]:.1%}</div><div class="lbl">Acurácia</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="m-card"><div class="val">{metricas["f1"]:.1%}</div><div class="lbl">F1-Score</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="m-card"><div class="val">{metricas["auc"]:.1%}</div><div class="lbl">AUC-ROC</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="m-card"><div class="val">{metricas["epocas"]}</div><div class="lbl">Épocas Treinadas</div><div class="sub-lbl">EarlyStopping</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="m-card vermelho"><div class="val verm">{pct_risco:.1%}</div><div class="lbl">Alunos em Risco</div><div class="sub-lbl">{em_risco} de {total}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ANÁLISE EXPLORATÓRIA ───────────────────────────────────
    st.markdown('<div class="sec-titulo">📈 Análise Exploratória dos Dados</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Notas & Risco",
        "🔗 Comparativo Mat vs Por",
        "📦 Fatores de Impacto",
        "🍺 Álcool vs Desempenho",
        "🌡️ Correlações",
        "🧠 Treinamento do Modelo"
    ])

    with tab1:
        ca, cb = st.columns([2, 1])
        with ca:
            st.pyplot(fig_distribuicao_notas(df_proc)); plt.close()
        with cb:
            st.pyplot(fig_pizza_risco(df_proc)); plt.close()
        st.markdown(f"""
        <div style="font-size:0.82rem;color:#475569;background:#F8FAFC;
                    padding:0.8rem 1rem;border-radius:8px;border:1px solid #E2E8F0">
        📌 <strong>Critério:</strong> G3 &lt; {NOTA_CORTE} → Alto Risco de Evasão/Baixo Desempenho.
        Dataset atual: <strong>{total} alunos</strong> · <strong>{em_risco} em risco ({pct_risco:.1%})</strong> ·
        Média G3: <strong>{df_proc['G3'].mean():.2f}</strong> · Desvio padrão: <strong>{df_proc['G3'].std():.2f}</strong>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.pyplot(fig_comparativo_datasets(df_mat, df_por)); plt.close()
        mat_risco = (df_mat["G3"] < NOTA_CORTE).mean()
        por_risco = (df_por["G3"] < NOTA_CORTE).mean()
        st.markdown(f"""
        <div style="font-size:0.82rem;color:#475569;background:#F8FAFC;
                    padding:0.8rem 1rem;border-radius:8px;border:1px solid #E2E8F0">
        📌 <strong>student-merge.R</strong> identifica 382 alunos matriculados nas duas disciplinas via join por
        13 atributos demográficos. · Risco em Matemática: <strong>{mat_risco:.1%}</strong> ·
        Risco em Português: <strong>{por_risco:.1%}</strong>
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.pyplot(fig_boxplot_fatores(df_proc)); plt.close()
        st.caption("Reprovações anteriores e alto número de faltas são os fatores de maior impacto negativo sobre G3.")

    with tab4:
        st.pyplot(fig_alcool_desempenho(df_proc)); plt.close()
        st.markdown("""
        <div class="etico">
        ⚠️ <strong>Interpretação responsável:</strong> O consumo de álcool é tratado como
        <em>indicador de vulnerabilidade social</em> — não julgamento moral.
        Deve gerar acolhimento e suporte ao aluno, nunca punição ou estigma.
        </div>""", unsafe_allow_html=True)

    with tab5:
        st.pyplot(fig_correlacoes(df_proc)); plt.close()
        st.caption("Principais correlações com G3: failures (−), absences (−), Medu/Fedu (+), studytime (+), higher (+).")

    with tab6:
        ca, cb = st.columns([3, 2])
        with ca:
            st.pyplot(fig_curvas_treino(metricas["hist"])); plt.close()
        with cb:
            st.pyplot(fig_matriz_confusao(metricas["cm"])); plt.close()
        st.markdown("**Relatório Completo de Classificação (Conjunto de Teste — 20%):**")
        st.code(metricas["report"], language=None)
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#475569;background:#F8FAFC;padding:0.8rem 1rem;
                    border-radius:8px;border:1px solid #E2E8F0;line-height:1.8">
        🧠 <strong>Arquitetura MLP:</strong>
        Input({len(feature_cols)}) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) →
        Dense(64, ReLU) → BatchNorm → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) →
        Dense(1, Sigmoid)<br>
        ⚙️ Otimizador: <code>Adam(lr=0.001)</code> · Loss: <code>Binary Crossentropy</code> ·
        <code>EarlyStopping(patience=20)</code> · Treinamento: <strong>{metricas['epocas']} épocas</strong>
        </div>""", unsafe_allow_html=True)

    # ── PREVISÃO INDIVIDUAL ────────────────────────────────────
    st.markdown('<div class="sec-titulo">🎯 Previsão Individual de Risco Escolar</div>',
                unsafe_allow_html=True)

    if prever:
        with st.spinner("⚙️ Processando perfil do aluno..."):
            try:
                X_in    = preparar_input(inputs_usuario, feature_cols, scaler)
                prob    = float(modelo.predict(X_in, verbose=0)[0][0])
                alto    = prob >= 0.5
                fatores = fatores_de_risco(inputs_usuario)

                col_res, col_fat = st.columns([1, 1])

                with col_res:
                    if alto:
                        st.markdown(f"""
                        <div class="res-alto">
                            <h2>⚠️ ALTO RISCO IDENTIFICADO</h2>
                            <p>Probabilidade de risco: <strong>{prob:.1%}</strong></p>
                        </div>""", unsafe_allow_html=True)
                        cor_barra = PALETTE["vermelho"]
                    else:
                        st.markdown(f"""
                        <div class="res-baixo">
                            <h2>✅ BAIXO RISCO — PERFIL ESTÁVEL</h2>
                            <p>Probabilidade de risco: <strong>{prob:.1%}</strong></p>
                        </div>""", unsafe_allow_html=True)
                        cor_barra = PALETTE["verde"]

                    st.markdown(f"""
                    <div style="margin-top:1rem">
                        <div style="font-size:0.8rem;font-weight:700;color:#475569;margin-bottom:4px">
                            Índice de Probabilidade de Risco
                        </div>
                        <div class="barra-wrap">
                            <div class="barra-fill" style="width:{prob*100:.1f}%;background:{cor_barra}"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94A3B8">
                            <span>0% — Sem risco</span><span>50% — Limiar</span><span>100% — Risco máximo</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if alto:
                        st.warning("**📋 Ação recomendada:** Encaminhar para acompanhamento pedagógico individualizado. Acionar rede de proteção social se necessário. Contato urgente com o responsável recomendado.")
                    else:
                        st.success("**📋 Ação recomendada:** Manter acompanhamento regular. Nenhum indicador crítico detectado. Reavaliar ao fim do bimestre.")

                with col_fat:
                    st.markdown("#### 🔍 Fatores que Influenciaram a Previsão")
                    st.caption("Análise interpretável baseada nos padrões aprendidos e correlações do dataset:")
                    for f in fatores:
                        st.markdown(f"""
                        <div class="fator {f['nivel']}">
                            <div class="ft">{f['titulo']}</div>
                            <div class="fd">{f['desc']}</div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erro ao processar previsão: {e}")

    else:
        st.info("👈 **Preencha o perfil do aluno** na barra lateral e clique em **Prever Risco do Aluno**.")

    # ── RODAPÉ ÉTICO ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="etico">
    <strong>⚖️ Diretrizes de Uso Ético (LGPD · Lei 13.709/2018):</strong><br>
    Esta ferramenta apoia profissionais da educação na identificação precoce de estudantes que
    precisam de suporte. A previsão de "Alto Risco" <strong>não é rótulo permanente</strong> e
    jamais deve gerar discriminação, exclusão ou estigma. Dados sobre álcool indicam vulnerabilidade
    social — e devem gerar <strong>acolhimento, não punição</strong>. Acesso restrito a profissionais
    autorizados. Revisão humana obrigatória antes de qualquer intervenção.
    </div>
    <p style="text-align:center;color:#94A3B8;font-size:0.75rem;margin-top:1.5rem">
        EduRisco v2.0 · Matemática + Português + Merge (student-merge.R) ·
        MLP (TensorFlow/Keras) · Student Alcohol Consumption (UCI/Kaggle) · Pedro, Thiago e equipe
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()