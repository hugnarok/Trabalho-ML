"""Separa as literaturas A e C (definidas em par.py) com TF-IDF e classificadores lineares.

A classe positiva é C. Título e resumo viram um vetor TF-IDF; a
logística (L2, a regularização Ridge da aula) estima P(C | texto).
O teste não entra no treino. Artigos que já citam os dois temas ficam de fora.

O primeiro ajuste vê o texto inteiro, inclusive as palavras da consulta.
O segundo apaga essas palavras, para verificar se as literaturas ainda
se separam pelo vocabulário de cada área — e não só pelo termo buscado.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline

from par import ANOS, NOME_A, NOME_C, ROTULO_C, SLUG, palavras_das_consultas

RAIZ = Path(__file__).resolve().parent
CORPUS = RAIZ / "data" / f"{SLUG}_{ANOS[0]}_{ANOS[-1]}.csv"
FIGURA = RAIZ / "figuras" / f"termos_{SLUG}.png"
METRICAS = RAIZ / "data" / f"metricas_{SLUG}.csv"
VALIDACAO_CRUZADA = RAIZ / "data" / f"validacao_cruzada_{SLUG}.csv"
BOOTSTRAP = RAIZ / "data" / f"bootstrap_{SLUG}.csv"
MCNEMAR = RAIZ / "data" / f"mcnemar_{SLUG}.csv"
MATRIZES = RAIZ / "data" / f"matrizes_{SLUG}.csv"

SEMENTE = 42
DOBRAS = 5
LIMIAR = 0.5
N_BOOTSTRAP = 1000
CLASSE_POSITIVA = ROTULO_C
NOME_POSITIVO = NOME_C
NOME_NEGATIVO = NOME_A

# Termos que a própria busca do PubMed usou. Sem retirá-los, o modelo
# só reaprende a consulta.
CONSULTA = palavras_das_consultas()


def texto_do_artigo(linha, sem_consulta=False):
    bruto = f"{linha.titulo or ''} {linha.resumo or ''}"
    if sem_consulta:
        bruto = CONSULTA.sub(" ", bruto)
    return bruto


def metricas(y_real, probabilidade, limiar=LIMIAR):
    previsto = (probabilidade >= limiar).astype(int)
    precisao, recall, f1, _ = precision_recall_fscore_support(
        y_real, previsto, average="binary", zero_division=0
    )
    if np.unique(probabilidade).size == 1:
        roc = 0.5
    else:
        roc = roc_auc_score(y_real, probabilidade)
    return {
        "acuracia": (previsto == y_real).mean(),
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": average_precision_score(y_real, probabilidade),
        "matriz": confusion_matrix(y_real, previsto, labels=[1, 0]),
    }


def novo_vetorizador():
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
    )


def vetorizar(textos_treino, textos_teste):
    vetorizador = novo_vetorizador()
    return vetorizador, vetorizador.fit_transform(textos_treino), vetorizador.transform(textos_teste)


def validacao_cruzada(textos_treino, y_treino):
    """Cinco dobras estratificadas só sobre o treino. O TF-IDF é refeito em cada dobra."""
    modelos = {
        "Regressão logística": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=SEMENTE
        ),
        "Naive Bayes": MultinomialNB(),
        "SVM linear": LinearSVC(C=1.0, class_weight="balanced", random_state=SEMENTE),
    }
    dobras = StratifiedKFold(n_splits=DOBRAS, shuffle=True, random_state=SEMENTE)
    metricas_cv = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    linhas = []
    for nome, modelo in modelos.items():
        resultado = cross_validate(
            make_pipeline(novo_vetorizador(), modelo),
            textos_treino,
            y_treino,
            cv=dobras,
            scoring=metricas_cv,
        )
        linha = {"modelo": nome}
        for metrica in metricas_cv:
            valores = resultado[f"test_{metrica}"]
            linha[f"{metrica}_media"] = valores.mean()
            linha[f"{metrica}_desvio"] = valores.std()
        linhas.append(linha)
        print(
            f"  {nome}: F1={linha['f1_media']:.3f} ± {linha['f1_desvio']:.3f}  "
            f"PR-AUC={linha['average_precision_media']:.3f} ± {linha['average_precision_desvio']:.3f}"
        )
    return pd.DataFrame(linhas)


def treinar(textos_treino, y_treino, textos_teste):
    vetorizador, x_treino, x_teste = vetorizar(textos_treino, textos_teste)
    modelo = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=SEMENTE,
    )
    modelo.fit(x_treino, y_treino)
    return vetorizador, modelo, modelo.predict_proba(x_treino)[:, 1], modelo.predict_proba(x_teste)[:, 1]


def comparar(textos_treino, y_treino, textos_teste, y_teste):
    """Naive Bayes e SVM linear no mesmo TF-IDF e na mesma divisão da logística."""
    _, x_treino, x_teste = vetorizar(textos_treino, textos_teste)
    resultados = {}

    bayes = MultinomialNB().fit(x_treino, y_treino)
    score_bayes_treino = bayes.predict_proba(x_treino)[:, 1]
    score_bayes_teste = bayes.predict_proba(x_teste)[:, 1]
    resultados["Naive Bayes"] = (
        metricas(y_treino, score_bayes_treino),
        metricas(y_teste, score_bayes_teste),
        score_bayes_teste,
        LIMIAR,
    )

    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEMENTE).fit(x_treino, y_treino)
    # A SVM devolve distância à margem. Limiar 0 equivale a 0,5 na escala da probabilidade.
    margem_treino = svm.decision_function(x_treino)
    margem_teste = svm.decision_function(x_teste)
    resultados["SVM linear"] = (
        metricas(y_treino, margem_treino, limiar=0.0),
        metricas(y_teste, margem_teste, limiar=0.0),
        margem_teste,
        0.0,
    )
    return resultados


def bootstrap_metricas(y_real, score, limiar=LIMIAR, n=N_BOOTSTRAP, semente=SEMENTE):
    """IC 95% por reamostragem com reposição do conjunto de teste."""
    rng = np.random.default_rng(semente)
    chaves = ("precisao", "recall", "f1", "roc_auc", "pr_auc", "acuracia")
    amostras = {chave: [] for chave in chaves}
    n_teste = len(y_real)
    for _ in range(n):
        idx = rng.integers(0, n_teste, size=n_teste)
        m = metricas(y_real[idx], score[idx], limiar=limiar)
        for chave in chaves:
            amostras[chave].append(m[chave])
    return {
        chave: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for chave, vals in amostras.items()
    }


def teste_mcnemar(y_real, prev_a, prev_b):
    """McNemar exato (binomial bilateral) entre dois classificadores no mesmo teste."""
    a_certo = prev_a == y_real
    b_certo = prev_b == y_real
    b01 = int((~a_certo & b_certo).sum())  # A errou, B acertou
    b10 = int((a_certo & ~b_certo).sum())  # A acertou, B errou
    discordantes = b01 + b10
    if discordantes == 0:
        return {"b01": b01, "b10": b10, "p": 1.0}
    p = float(binomtest(b10, n=discordantes, p=0.5, alternative="two-sided").pvalue)
    return {"b01": b01, "b10": b10, "p": p}


def termos_extremos(vetorizador, modelo, n=12):
    nomes = vetorizador.get_feature_names_out()
    coeficientes = modelo.coef_[0]
    ordem = coeficientes.argsort()
    negativos = [(nomes[i], coeficientes[i]) for i in ordem[:n]]
    positivos = [(nomes[i], coeficientes[i]) for i in ordem[-n:][::-1]]
    return negativos, positivos


def desenhar(termos_negativos, termos_positivos):
    fig, eixos = plt.subplots(1, 2, figsize=(10, 5.2))
    paineis = (
        (eixos[0], termos_negativos[::-1], NOME_NEGATIVO, "#1f4e79"),
        (eixos[1], termos_positivos[::-1], NOME_POSITIVO, "#0f6e56"),
    )
    for eixo, termos, titulo, cor in paineis:
        rotulos = [nome for nome, _ in termos]
        valores = [abs(peso) for _, peso in termos]
        eixo.barh(rotulos, valores, color=cor)
        eixo.set_title(titulo)
        eixo.set_xlabel("peso absoluto na regressão logística")
    fig.suptitle(f"Termos que separam {NOME_POSITIVO} e {NOME_NEGATIVO}, sem as palavras da consulta")
    fig.tight_layout()
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA, dpi=150)
    plt.close(fig)


def main():
    corpus = pd.read_csv(CORPUS)
    corpus = corpus[corpus["literatura"] != "ambas"].reset_index(drop=True)
    corpus["titulo"] = corpus["titulo"].fillna("")
    corpus["resumo"] = corpus["resumo"].fillna("")
    y = (corpus["literatura"] == CLASSE_POSITIVA).astype(int).to_numpy()

    indice_treino, indice_teste = train_test_split(
        np.arange(len(corpus)),
        test_size=0.2,
        stratify=y,
        random_state=SEMENTE,
    )
    y_treino, y_teste = y[indice_treino], y[indice_teste]

    linhas = []
    linhas_bootstrap = []
    linhas_matriz = []
    scores_sem_consulta = {}
    for nome, sem_consulta in (("com as palavras da consulta", False), ("sem as palavras da consulta", True)):
        textos = corpus.apply(lambda linha: texto_do_artigo(linha, sem_consulta), axis=1)
        vetorizador, modelo, prob_treino, prob_teste = treinar(
            textos.iloc[indice_treino],
            y_treino,
            textos.iloc[indice_teste],
        )
        teste = metricas(y_teste, prob_teste)
        treino = metricas(y_treino, prob_treino)
        linhas.append(
            {
                "modelo": f"Regressão logística, {nome}",
                "f1_treino": treino["f1"],
                "f1_teste": teste["f1"],
                "precisao": teste["precisao"],
                "recall": teste["recall"],
                "roc_auc": teste["roc_auc"],
                "pr_auc": teste["pr_auc"],
                "acuracia": teste["acuracia"],
            }
        )
        print(f"\n{nome}")
        print(
            f"  teste  F1={teste['f1']:.3f}  precisão={teste['precisao']:.3f}  "
            f"recall={teste['recall']:.3f}  ROC-AUC={teste['roc_auc']:.3f}  "
            f"PR-AUC={teste['pr_auc']:.3f}  acurácia={teste['acuracia']:.3f}"
        )
        print(f"  treino F1={treino['f1']:.3f}")
        vp, fn, fp, vn = teste["matriz"].ravel()
        print(f"  matriz (positivo = {NOME_POSITIVO}): VP={vp} FN={fn} FP={fp} VN={vn}")
        print(f"  vocabulário TF-IDF: {len(vetorizador.vocabulary_)} termos")
        if sem_consulta:
            scores_sem_consulta["Regressão logística"] = (prob_teste, LIMIAR)
            linhas_matriz.append(
                {
                    "modelo": "Regressão logística",
                    "vp": int(vp),
                    "fn": int(fn),
                    "fp": int(fp),
                    "vn": int(vn),
                }
            )
            print(f"  bootstrap IC 95% ({N_BOOTSTRAP} reamostragens):")
            ic = bootstrap_metricas(y_teste, prob_teste)
            for metrica, (lo, hi) in ic.items():
                print(f"    {metrica}: [{lo:.3f}, {hi:.3f}]")
                linhas_bootstrap.append(
                    {
                        "modelo": "Regressão logística",
                        "metrica": metrica,
                        "ic_inf": lo,
                        "ic_sup": hi,
                    }
                )
            print(f"  validação cruzada ({DOBRAS} dobras, só treino):")
            validacao_cruzada(textos.iloc[indice_treino], y_treino).to_csv(
                VALIDACAO_CRUZADA, index=False
            )
            termos_negativos, termos_positivos = termos_extremos(vetorizador, modelo)
            desenhar(termos_negativos, termos_positivos)
            print(f"  termos {NOME_NEGATIVO}:", ", ".join(nome for nome, _ in termos_negativos))
            print(f"  termos {NOME_POSITIVO}:", ", ".join(nome for nome, _ in termos_positivos))
            comparacao = comparar(
                textos.iloc[indice_treino], y_treino, textos.iloc[indice_teste], y_teste
            )
            for nome_modelo, (treino_c, teste_c, score_c, limiar_c) in comparacao.items():
                scores_sem_consulta[nome_modelo] = (score_c, limiar_c)
                linhas.append(
                    {
                        "modelo": f"{nome_modelo}, {nome}",
                        "f1_treino": treino_c["f1"],
                        "f1_teste": teste_c["f1"],
                        "precisao": teste_c["precisao"],
                        "recall": teste_c["recall"],
                        "roc_auc": teste_c["roc_auc"],
                        "pr_auc": teste_c["pr_auc"],
                        "acuracia": teste_c["acuracia"],
                    }
                )
                vp_c, fn_c, fp_c, vn_c = teste_c["matriz"].ravel()
                linhas_matriz.append(
                    {
                        "modelo": nome_modelo,
                        "vp": int(vp_c),
                        "fn": int(fn_c),
                        "fp": int(fp_c),
                        "vn": int(vn_c),
                    }
                )
                print(
                    f"  {nome_modelo}: F1={teste_c['f1']:.3f}  precisão={teste_c['precisao']:.3f}  "
                    f"recall={teste_c['recall']:.3f}  ROC-AUC={teste_c['roc_auc']:.3f}  "
                    f"PR-AUC={teste_c['pr_auc']:.3f}"
                )
                print(f"    matriz: VP={vp_c} FN={fn_c} FP={fp_c} VN={vn_c}")
                ic_c = bootstrap_metricas(y_teste, score_c, limiar=limiar_c)
                for metrica, (lo, hi) in ic_c.items():
                    linhas_bootstrap.append(
                        {
                            "modelo": nome_modelo,
                            "metrica": metrica,
                            "ic_inf": lo,
                            "ic_sup": hi,
                        }
                    )
                print(
                    f"    bootstrap F1=[{ic_c['f1'][0]:.3f}, {ic_c['f1'][1]:.3f}]  "
                    f"PR-AUC=[{ic_c['pr_auc'][0]:.3f}, {ic_c['pr_auc'][1]:.3f}]"
                )

    prevalencia = y_teste.mean()
    linhas.insert(
        0,
        {
            "modelo": f"sempre {NOME_NEGATIVO}",
            "f1_treino": 0.0,
            "f1_teste": 0.0,
            "precisao": 0.0,
            "recall": 0.0,
            "roc_auc": 0.5,
            "pr_auc": prevalencia,
            "acuracia": 1 - prevalencia,
        },
    )

    print("\nMcNemar (texto sem palavras da consulta):")
    previsoes = {
        nome: (score >= limiar).astype(int)
        for nome, (score, limiar) in scores_sem_consulta.items()
    }
    pares = [
        ("Regressão logística", "Naive Bayes"),
        ("Regressão logística", "SVM linear"),
        ("Naive Bayes", "SVM linear"),
    ]
    linhas_mcnemar = []
    for a, b in pares:
        r = teste_mcnemar(y_teste, previsoes[a], previsoes[b])
        linhas_mcnemar.append({"modelo_a": a, "modelo_b": b, **r})
        print(f"  {a} vs {b}: b10={r['b10']} b01={r['b01']} p={r['p']:.4f}")

    tabela = pd.DataFrame(linhas)
    METRICAS.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(METRICAS, index=False)
    pd.DataFrame(linhas_bootstrap).to_csv(BOOTSTRAP, index=False)
    pd.DataFrame(linhas_mcnemar).to_csv(MCNEMAR, index=False)
    pd.DataFrame(linhas_matriz).to_csv(MATRIZES, index=False)
    print(f"\nMétricas em {METRICAS}")
    print(f"Bootstrap em {BOOTSTRAP}")
    print(f"McNemar em {MCNEMAR}")
    print(f"Matrizes em {MATRIZES}")
    print(f"Figura em {FIGURA}")
    print(f"Teste: {len(y_teste)} artigos, {int(y_teste.sum())} de {NOME_POSITIVO} ({prevalencia:.1%})")


if __name__ == "__main__":
    main()
