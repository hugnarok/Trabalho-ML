"""Cliente do PubMed (NCBI E-utilities): busca os PMIDs mais recentes e baixa os registros."""

import os
import time
import json
import xml.etree.ElementTree as ET

import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL = "trabalho_ml_lbd"
EMAIL = os.environ.get("NCBI_EMAIL", "")
PAUSA_SEGUNDOS = 0.4
LOTE = 200


def _parametros(extra):
    base = {"tool": TOOL}
    if EMAIL:
        base["email"] = EMAIL
    base.update(extra)
    return base


def _texto(elemento):
    if elemento is None:
        return ""
    return "".join(elemento.itertext()).strip()


def _ano(article):
    pubdate = article.find("./Journal/JournalIssue/PubDate")
    if pubdate is None:
        return ""
    year = (pubdate.findtext("Year") or "").strip()
    if year:
        return year
    medline = pubdate.findtext("MedlineDate") or ""
    for token in medline.replace("-", " ").split():
        if len(token) >= 4 and token[:4].isdigit():
            return token[:4]
    return ""


def _pedido(url, params=None, data=None, timeout=90):
    """Repete se o PubMed devolver erro temporário."""
    ultimo = None
    for tentativa in range(4):
        if data is None:
            resposta = requests.get(url, params=params, timeout=timeout)
        else:
            resposta = requests.post(url, data=data, timeout=timeout)
        if resposta.status_code < 500:
            resposta.raise_for_status()
            return resposta
        ultimo = resposta
        time.sleep(2 * (tentativa + 1))
    ultimo.raise_for_status()
    return ultimo


def pmids_mais_recentes(termo, limite):
    """Uma chamada à API: a contagem total e só os PMIDs mais recentes."""
    resposta = _pedido(
        ESEARCH,
        params=_parametros(
            {
                "db": "pubmed",
                "term": termo,
                "retmode": "json",
                "retmax": limite,
                "sort": "pub_date",
            }
        ),
        timeout=60,
    )
    resultado = json.loads(resposta.text, strict=False)["esearchresult"]
    if "errorlist" in resultado:
        raise RuntimeError(resultado["errorlist"])
    return int(resultado["count"]), resultado.get("idlist") or []


def _artigos_do_xml(conteudo):
    raiz = ET.fromstring(conteudo)
    linhas = []
    for artigo_pubmed in raiz.findall(".//PubmedArticle"):
        citacao = artigo_pubmed.find("./MedlineCitation")
        article = citacao.find("./Article")
        if article is None:
            continue
        partes_resumo = []
        for bloco in article.findall("./Abstract/AbstractText"):
            trecho = _texto(bloco)
            rotulo = bloco.attrib.get("Label")
            partes_resumo.append(f"{rotulo}: {trecho}" if rotulo else trecho)
        doi = ""
        for identificador in article.findall("./ELocationID"):
            if identificador.attrib.get("EIdType") == "doi":
                doi = _texto(identificador)
        if not doi:
            for identificador in artigo_pubmed.findall("./PubmedData/ArticleIdList/ArticleId"):
                if identificador.attrib.get("IdType") == "doi":
                    doi = _texto(identificador)
                    break
        mesh = [
            _texto(descritor)
            for descritor in citacao.findall("./MeshHeadingList/MeshHeading/DescriptorName")
            if _texto(descritor)
        ]
        linhas.append(
            {
                "pmid": _texto(citacao.find("./PMID")),
                "titulo": _texto(article.find("./ArticleTitle")),
                "resumo": "\n".join(partes_resumo),
                "ano": _ano(article),
                "mesh": "; ".join(mesh),
                "doi": doi,
            }
        )
    return linhas


def baixar_pmids(pmids):
    linhas = []
    total = len(pmids)
    for inicio in range(0, total, LOTE):
        lote = pmids[inicio : inicio + LOTE]
        resposta = _pedido(
            EFETCH,
            data=_parametros(
                {
                    "db": "pubmed",
                    "id": ",".join(lote),
                    "retmode": "xml",
                }
            ),
            timeout=120,
        )
        resposta.raise_for_status()
        linhas.extend(_artigos_do_xml(resposta.content))
        print(f"  {min(inicio + LOTE, total)}/{total}")
        if inicio + LOTE < total:
            time.sleep(PAUSA_SEGUNDOS)
    return linhas
