"""Procura conceitos MeSH que possam ligar as literaturas A e C (definidas em par.py).

Um descritor entra como ponte se aparece nas duas literaturas, com pelo
menos três artigos de cada lado. Sai se for genérico: pessoa, animal,
desenho de estudo, se nomeia um dos temas, ou se cobre 25% ou mais de uma das literaturas.

O suporte é a média harmônica das duas contagens. Ela sobe quando o
conceito tem artigos nos dois lados e cai quando a evidência está só
em um deles.
"""

from collections import Counter
from statistics import median
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from par import (
    ANOS,
    MESH_DAS_CONSULTAS,
    NOME_A,
    NOME_C,
    ROTULO_A,
    ROTULO_C,
    SLUG,
    VIAS_CONHECIDAS,
)

RAIZ = Path(__file__).resolve().parent
CORPUS = RAIZ / "data" / f"{SLUG}_{ANOS[0]}_{ANOS[-1]}.csv"
FIGURA = RAIZ / "figuras" / f"pontes_mesh_{SLUG}.png"
TABELA = RAIZ / "data" / f"pontes_mesh_{SLUG}.csv"
VALIDACAO = RAIZ / "data" / f"validacao_pontes_{SLUG}.csv"
LADO_A = ROTULO_A
LADO_C = ROTULO_C

MINIMO_DE_CADA_LADO = 3
FRACAO_GENERICA = 0.25
TOP_K = 10
N_BOOTSTRAP = 1000
N_PERMUTACAO = 1000
SEMENTE = 42

# Primeira versão do filtro anti-absurdo: MeSH que não descreve mecanismo.
STOP = {
    "Humans",
    "Male",
    "Female",
    "Adult",
    "Middle Aged",
    "Aged",
    "Aged, 80 and over",
    "Young Adult",
    "Adolescent",
    "Child",
    "Child, Preschool",
    "Infant",
    "Infant, Newborn",
    "Pregnancy",
    "Sex Factors",
    "Age Factors",
    "Animals",
    "Rats",
    "Mice",
    "Rabbits",
    "Dogs",
    "Guinea Pigs",
    "Cats",
    "Cattle",
    "Swine",
    "Sheep",
    "Haplorhini",
    "Time Factors",
    "Double-Blind Method",
    "Clinical Trials as Topic",
    "Prospective Studies",
    "Retrospective Studies",
    "Random Allocation",
    "Follow-Up Studies",
    "Pilot Projects",
    "Research Design",
    "Placebos",
    "Reference Values",
    "Cells, Cultured",
}

MECANISMO_CLASSICO = VIAS_CONHECIDAS


def contar(serie):
    contagem = Counter()
    for mesh in serie.fillna(""):
        termos = {termo.strip() for termo in mesh.split(";") if termo.strip()}
        contagem.update(termos)
    return contagem


def suporte(artigos_a, artigos_c):
    return 2 * artigos_a * artigos_c / (artigos_a + artigos_c)


def enriquecimento(termo, ambas, n_ambas, exclusivas, n_exclusivas):
    """Quantas vezes o termo é mais comum nos artigos que já citam os dois temas."""
    return (ambas[termo] / n_ambas) / (exclusivas[termo] / n_exclusivas)


def _mesh_por_artigo(serie):
    return [{termo.strip() for termo in mesh.split(";") if termo.strip()} for mesh in serie.fillna("")]


def _enriquecimento_lista(termos, ambas_listas, exclusivas_listas):
    n_ambas = len(ambas_listas)
    n_exclusivas = len(exclusivas_listas)
    if n_ambas == 0 or n_exclusivas == 0:
        return [0.0] * len(termos)
    valores = []
    for termo in termos:
        n_ac = sum(termo in m for m in ambas_listas)
        n_exc = sum(termo in m for m in exclusivas_listas)
        if n_exc == 0:
            valores.append(float("inf") if n_ac > 0 else 0.0)
        else:
            valores.append((n_ac / n_ambas) / (n_exc / n_exclusivas))
    return valores


def bootstrap_mediana(termos, ambas_listas, exclusivas_listas, n=N_BOOTSTRAP, semente=SEMENTE):
    """IC 95% da mediana de E(B) reamostrando os artigos com MeSH."""
    rng = np.random.default_rng(semente)
    n_ambas = len(ambas_listas)
    n_exc = len(exclusivas_listas)
    amostras = []
    for _ in range(n):
        idx_a = rng.integers(0, n_ambas, size=n_ambas)
        idx_e = rng.integers(0, n_exc, size=n_exc)
        vals = _enriquecimento_lista(
            termos,
            [ambas_listas[i] for i in idx_a],
            [exclusivas_listas[i] for i in idx_e],
        )
        finitos = [v for v in vals if np.isfinite(v)]
        amostras.append(median(finitos) if finitos else 0.0)
    return float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))


def permutacao_mediana(termos_topo, candidatos, ambas_listas, exclusivas_listas, n=N_PERMUTACAO, semente=SEMENTE):
    """p-valor: fração de sorteios de k candidatos com mediana E >= observada."""
    observada = median(_enriquecimento_lista(termos_topo, ambas_listas, exclusivas_listas))
    rng = np.random.default_rng(semente)
    k = len(termos_topo)
    pool = list(candidatos)
    acima = 0
    for _ in range(n):
        amostra = rng.choice(pool, size=k, replace=False)
        med = median(_enriquecimento_lista(amostra, ambas_listas, exclusivas_listas))
        if med >= observada:
            acima += 1
    return observada, (acima + 1) / (n + 1)


def validar(pontes, tabela, corpus, k=TOP_K):
    """Compara o topo do ranking com o topo por frequência bruta, sem filtro.

    Os artigos "ambas" não entram no ranking. Um conceito que liga mesmo as
    duas literaturas deve ser mais comum neles do que nas literaturas exclusivas.
    """
    com_mesh = corpus.mesh.notna()
    ambas_mask = (corpus.literatura == "ambas") & com_mesh
    exclusivas_mask = corpus.literatura.isin([LADO_A, LADO_C]) & com_mesh
    ambas = contar(corpus.loc[ambas_mask, "mesh"])
    n_ambas = int(ambas_mask.sum())
    exclusivas = contar(corpus.loc[exclusivas_mask, "mesh"])
    n_exclusivas = int(exclusivas_mask.sum())
    ambas_listas = _mesh_por_artigo(corpus.loc[ambas_mask, "mesh"])
    exclusivas_listas = _mesh_por_artigo(corpus.loc[exclusivas_mask, "mesh"])

    frequencia = tabela.assign(
        total=tabela.artigos_a + tabela.artigos_c
    ).sort_values("total", ascending=False)
    resultado = {}
    linhas_saida = []
    for nome, ranking in (("pontes filtradas", pontes), ("frequência bruta", frequencia)):
        topo = list(ranking.head(k).termo)
        valores = [enriquecimento(t, ambas, n_ambas, exclusivas, n_exclusivas) for t in topo]
        presentes = sum(ambas[t] > 0 for t in topo)
        med = median(valores)
        ic_lo, ic_hi = bootstrap_mediana(topo, ambas_listas, exclusivas_listas)
        resultado[nome] = (med, presentes, ic_lo, ic_hi)
        print(
            f"  {nome:17}  mediana do enriquecimento top-{k}={med:.2f}  "
            f"IC95%=[{ic_lo:.2f}, {ic_hi:.2f}]  presentes em 'ambas'={presentes}/{k}"
        )
        linhas_saida.append(
            {
                "ranking": nome,
                "mediana": med,
                "ic_inf": ic_lo,
                "ic_sup": ic_hi,
                "presentes": presentes,
                "k": k,
                "p_permutacao": np.nan,
            }
        )

    candidatos = list(pontes.termo)
    med_obs, p_perm = permutacao_mediana(
        list(pontes.head(k).termo), candidatos, ambas_listas, exclusivas_listas
    )
    print(
        f"  permutação (sorteio de {k} entre {len(candidatos)} pontes): "
        f"mediana observada={med_obs:.2f}  p={p_perm:.4f}"
    )
    linhas_saida[0]["p_permutacao"] = p_perm
    print(f"  ({n_ambas} artigos 'ambas' com MeSH; {n_exclusivas} exclusivos com MeSH)")
    pd.DataFrame(linhas_saida).to_csv(VALIDACAO, index=False)
    print(f"  validação salva em {VALIDACAO}")
    return resultado


def classificar(termo, artigos_a, artigos_c, n_a, n_c):
    if termo in STOP:
        return "stop"
    if termo in MESH_DAS_CONSULTAS:
        return "nomeia a consulta"
    if max(artigos_a / n_a, artigos_c / n_c) >= FRACAO_GENERICA:
        return "frequente demais"
    if min(artigos_a, artigos_c) < MINIMO_DE_CADA_LADO:
        return "evidencia fraca"
    return "ponte"


def main():
    corpus = pd.read_csv(CORPUS)
    print(f"artigos que já citam os dois temas: {(corpus.literatura == 'ambas').sum()}")
    n_a = int((corpus.literatura == LADO_A).sum())
    n_c = int((corpus.literatura == LADO_C).sum())
    lado_a = contar(corpus.loc[corpus.literatura == LADO_A, "mesh"])
    lado_c = contar(corpus.loc[corpus.literatura == LADO_C, "mesh"])

    linhas = []
    for termo in set(lado_a) & set(lado_c):
        artigos_a = lado_a[termo]
        artigos_c = lado_c[termo]
        linhas.append(
            {
                "termo": termo,
                "artigos_a": artigos_a,
                "artigos_c": artigos_c,
                "suporte": suporte(artigos_a, artigos_c),
                "situacao": classificar(termo, artigos_a, artigos_c, n_a, n_c),
                "mecanismo_classico": termo in MECANISMO_CLASSICO,
            }
        )

    tabela = pd.DataFrame(linhas).sort_values(
        ["situacao", "suporte"], ascending=[True, False]
    )
    TABELA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(TABELA, index=False)

    pontes = tabela[tabela.situacao == "ponte"].sort_values("suporte", ascending=False)
    print(f"MeSH em comum: {len(tabela)}")
    print(tabela.situacao.value_counts().to_string())
    print("\nPontes com maior suporte:")
    for linha in pontes.head(15).itertuples(index=False):
        marca = "  [via já discutida]" if linha.mecanismo_classico else ""
        print(
            f"  {linha.suporte:5.1f}  {NOME_A}={linha.artigos_a:4d}  "
            f"{NOME_C}={linha.artigos_c:4d}  {linha.termo}{marca}"
        )

    classicas = tabela[tabela.mecanismo_classico].sort_values("suporte", ascending=False)
    print("\nVias já discutidas, estejam ou não no topo:")
    for linha in classicas.itertuples(index=False):
        print(
            f"  {linha.situacao:16}  suporte={linha.suporte:5.1f}  "
            f"{NOME_A}={linha.artigos_a:4d}  {NOME_C}={linha.artigos_c:4d}  {linha.termo}"
        )

    print("\nValidação com os artigos que já citam os dois temas:")
    validar(pontes, tabela, corpus)

    desenhar(pontes.head(12))
    print(f"\nTabela em {TABELA}")
    print(f"Figura em {FIGURA}")


def desenhar(pontes):
    pontes = pontes.iloc[::-1]
    cores = ["#0f6e56" if classico else "#1f4e79" for classico in pontes.mecanismo_classico]
    rotulos = [
        f"{termo}  ({artigos_a}/{artigos_c})"
        for termo, artigos_a, artigos_c in zip(
            pontes.termo, pontes.artigos_a, pontes.artigos_c
        )
    ]
    fig, eixo = plt.subplots(figsize=(9.4, 5.6))
    eixo.barh(rotulos, pontes.suporte, color=cores)
    eixo.set_xlabel("suporte (média harmônica dos artigos nos dois lados)")
    eixo.set_title(f"Conceitos MeSH nas duas literaturas\n(entre parênteses: artigos de {NOME_A} / {NOME_C})")
    eixo.legend(
        handles=[
            Patch(facecolor="#0f6e56", label="via já discutida na literatura"),
            Patch(facecolor="#1f4e79", label="outra ponte"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
