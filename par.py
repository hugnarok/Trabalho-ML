"""Par de temas A e C. Escolha o par com a variável de ambiente PAR_ID.

Exemplos:
    PAR_ID=cogumelo_alzheimer python coleta.py
    PAR_ID=parkinson_alzheimer python separar_literaturas.py
    PAR_ID=couve_anemia python pontes_mesh.py

O padrão é o estudo de caso principal (cogumelos e Alzheimer).
"""

import os
import re

# ---------------------------------------------------------------------------
# Catálogo de pares. Trocar de tema = só escolher outro PAR_ID.
# ---------------------------------------------------------------------------

PARES = {
    "cogumelo_alzheimer": {
        "NOME_A": "Alzheimer",
        "NOME_C": "Cogumelo",
        "CONSULTA_A": '"Alzheimer"[Title/Abstract]',
        "CONSULTA_C": (
            '("mushroom"[Title/Abstract] OR "mushrooms"[Title/Abstract] '
            "OR Hericium[Title/Abstract] OR Ganoderma[Title/Abstract] "
            'OR Pleurotus[Title/Abstract] OR "Agaricus"[Title/Abstract] '
            "OR ergothioneine[Title/Abstract])"
        ),
        # "Mushroom body" = estrutura do cérebro de insetos, não um cogumelo.
        "EXCLUIR": '"mushroom body"[Title/Abstract] OR "mushroom bodies"[Title/Abstract]',
        "PALAVRAS_EXTRAS": [
            r"erinaceus",
            r"lion'?s?\s*mane",
            r"ad",
            r"lucidum",
            r"ganoderma",
        ],
        "MESH_DAS_CONSULTAS": {
            "Alzheimer Disease",
            "Mushrooms",
            "Agaricales",
            "Ergothioneine",
        },
        "VIAS_CONHECIDAS": {
            "Nerve Growth Factor",
            "Brain-Derived Neurotrophic Factor",
            "Amyloid beta-Peptides",
            "tau Proteins",
            "Oxidative Stress",
            "Neuroprotective Agents",
            "Microglia",
            "Antioxidants",
            "Ergothioneine",
            "beta-Glucans",
            "Polysaccharides",
        },
    },
    # Controle de literaturas próximas: duas doenças neurodegenerativas.
    "parkinson_alzheimer": {
        "NOME_A": "Alzheimer",
        "NOME_C": "Parkinson",
        "CONSULTA_A": '"Alzheimer"[Title/Abstract]',
        "CONSULTA_C": '"Parkinson"[Title/Abstract]',
        "EXCLUIR": "",
        "PALAVRAS_EXTRAS": [r"parkinsonian", r"parkinsonism", r"ad", r"pd"],
        "MESH_DAS_CONSULTAS": {
            "Alzheimer Disease",
            "Parkinson Disease",
            "Parkinsonian Disorders",
        },
        "VIAS_CONHECIDAS": {
            "Oxidative Stress",
            "alpha-Synuclein",
            "Dopamine",
            "Substantia Nigra",
            "Lewy Bodies",
            "Amyloid beta-Peptides",
            "tau Proteins",
            "Neuroprotective Agents",
            "Microglia",
        },
    },
    # Segundo par distante alimento–doença (exemplo citado no artigo).
    "couve_anemia": {
        "NOME_A": "Anemia",
        "NOME_C": "Couve",
        "CONSULTA_A": '"anemia"[Title/Abstract] OR "anaemia"[Title/Abstract]',
        "CONSULTA_C": (
            '("kale"[Title/Abstract] OR "Brassica oleracea"[Title/Abstract] '
            'OR "leafy green"[Title/Abstract] OR "leafy greens"[Title/Abstract] '
            'OR "Brassica vegetables"[Title/Abstract])'
        ),
        "EXCLUIR": "",
        "PALAVRAS_EXTRAS": [r"brassica", r"cruciferous"],
        "MESH_DAS_CONSULTAS": {
            "Anemia",
            "Brassica",
            "Brassica oleracea",
            "Vegetables",
        },
        "VIAS_CONHECIDAS": {
            "Iron",
            "Folic Acid",
            "Vitamin K",
            "Vitamin C",
            "Dietary Fiber",
            "Antioxidants",
            "Hemoglobins",
            "Ferritins",
        },
    },
}

PAR_ID = os.environ.get("PAR_ID", "cogumelo_alzheimer")
if PAR_ID not in PARES:
    raise SystemExit(f"PAR_ID desconhecido: {PAR_ID!r}. Opções: {', '.join(PARES)}")

_cfg = PARES[PAR_ID]

NOME_A = _cfg["NOME_A"]
NOME_C = _cfg["NOME_C"]
CONSULTA_A = _cfg["CONSULTA_A"]
CONSULTA_C = _cfg["CONSULTA_C"]
EXCLUIR = _cfg["EXCLUIR"]
PALAVRAS_EXTRAS = _cfg["PALAVRAS_EXTRAS"]
MESH_DAS_CONSULTAS = _cfg["MESH_DAS_CONSULTAS"]
VIAS_CONHECIDAS = _cfg["VIAS_CONHECIDAS"]

ANOS = range(2022, 2027)
POR_ANO = 100
LIMITE_AMBAS = 100

ROTULO_A = NOME_A.lower()
ROTULO_C = NOME_C.lower()
SLUG = f"{ROTULO_C}_{ROTULO_A}"


def consultas():
    """Só A, só C e os dois, já sem os sentidos excluídos."""
    excluir = f" NOT ({EXCLUIR})" if EXCLUIR else ""
    return {
        ROTULO_A: f"({CONSULTA_A}) NOT ({CONSULTA_C}){excluir}",
        ROTULO_C: f"({CONSULTA_C}) NOT ({CONSULTA_A}){excluir}",
        "ambas": f"({CONSULTA_A}) AND ({CONSULTA_C}){excluir}",
    }


def palavras_das_consultas():
    """Regex com os termos das duas consultas (e plural simples), mais os sinônimos extras."""
    termos = []
    for consulta in (CONSULTA_A, CONSULTA_C):
        for entre_aspas, solto in re.findall(r'"([^"]+)"\[|([\w-]+)\[', consulta):
            termo = (entre_aspas or solto).lower()
            padrao = r"\s+".join(map(re.escape, termo.split()))
            if not termo.endswith("s"):
                padrao += "s?"
            if padrao not in termos:
                termos.append(padrao)
    return re.compile(r"\b(?:" + "|".join(termos + PALAVRAS_EXTRAS) + r")\b", flags=re.IGNORECASE)
