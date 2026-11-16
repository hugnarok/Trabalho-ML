"""Recorte leve de duas literaturas, A e C, via API do PubMed.

O par vem de par.py. Não baixa a literatura inteira: para cada ano da janela,
pede só os artigos mais recentes de "só A" e de "só C". Os que citam os dois
entram até um limite, também os mais recentes da janela.
"""

import time
from pathlib import Path

import pandas as pd

from coleta_pubmed import PAUSA_SEGUNDOS, baixar_pmids, pmids_mais_recentes
from par import ANOS, LIMITE_AMBAS, POR_ANO, SLUG, consultas

SAIDA = Path(__file__).resolve().parent / "data" / f"{SLUG}_{ANOS[0]}_{ANOS[-1]}.csv"
JANELA = f"{ANOS[0]}:{ANOS[-1]}[pdat]"


def pmids_da_literatura(nome, base):
    if nome == "ambas":
        total, pmids = pmids_mais_recentes(f"({base}) AND {JANELA}", LIMITE_AMBAS)
        print(f"{nome}: {total} no PubMed, baixando {len(pmids)}")
        return pmids

    pmids = []
    total_janela = 0
    for ano in ANOS:
        total, pagina = pmids_mais_recentes(f"({base}) AND {ano}:{ano}[pdat]", POR_ANO)
        total_janela += total
        pmids.extend(pagina)
        print(f"  {ano}: {total} no PubMed, {len(pagina)} baixados")
        time.sleep(PAUSA_SEGUNDOS)
    unicos = list(dict.fromkeys(pmids))
    print(f"{nome}: {total_janela} no PubMed na janela, baixando {len(unicos)}")
    return unicos


def coletar():
    tabelas = []
    for nome, base in consultas().items():
        tabela = pd.DataFrame(baixar_pmids(pmids_da_literatura(nome, base)))
        tabela["literatura"] = nome
        tabelas.append(tabela)
        time.sleep(PAUSA_SEGUNDOS)

    corpus = pd.concat(tabelas, ignore_index=True).drop_duplicates("pmid")
    colunas = ["pmid", "literatura", "ano", "titulo", "resumo", "mesh", "doi"]
    corpus = corpus[colunas].sort_values(["literatura", "ano", "pmid"])
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(SAIDA, index=False)
    return corpus


if __name__ == "__main__":
    dados = coletar()
    print(f"\nSalvo em {SAIDA}")
    print(dados["literatura"].value_counts().to_string())
    print(f"sem resumo: {(dados['resumo'].fillna('') == '').sum()}")
    print(f"sem MeSH: {(dados['mesh'].fillna('') == '').sum()}")
    print(f"anos: {dados['ano'].min()}–{dados['ano'].max()}")
