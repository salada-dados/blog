# Perguntas e respostas sobre o RPM de setembro de 2026

`python answer.py "pergunta"` responde em uma frase curta, usando só modelos locais do Ollama.

## Como funciona

1. **Indexação** (`build_index.py`, roda uma vez e grava em `index/`):
   - Lê o PDF com PyMuPDF e remonta as linhas da página pela posição das palavras.
   - **Tabelas** (24): acha a legenda "Tabela X – …" e o fim ("Fonte", "Nota" ou o próximo título), separa
     o cabeçalho dos dados e dá a cada célula o rótulo da coluna, juntando as linhas do cabeçalho.
     Anos que abrangem várias colunas são distribuídos pela sequência de trimestres e meses (I, II… / Jan, Fev…).
     Cada linha da tabela vira um trecho próprio, por exemplo:
     `Tabela 1.2.1 – PIB | Agropecuária | … Tri ante tri -1 (a.s.) 2026 II: 2,8`.
   - **Gráficos** (130): título, legendas e rótulos viram texto de busca. A área do gráfico é recortada em PNG
     (`index/charts/`).
   - **Texto corrido**: blocos de cerca de 170 palavras, sem o texto de dentro das tabelas e gráficos.
   - Embeddings com `bge-m3` (pelo Ollama) em `index/emb.npy`.
2. **Resposta** (`answer.py`):
   - Busca híbrida: BM25 sem acentos mais similaridade de cosseno do bge-m3, combinados por RRF, com uma
     pequena expansão de termos ("doze meses" → "12m", "média móvel de três meses" → "MM3M").
     O contexto sempre leva 4 linhas de tabela e 6 trechos de texto.
   - **Filtro de período**: se a pergunta cita um ano, um mês ou um trimestre ("novembro de 2026",
     "quarto trimestre de 2027"), cada linha de tabela vai ao LLM só com as células dessa coluna. Isso
     resolveu os erros em que o modelo pegava a coluna vizinha.
   - Pergunta sobre gráfico (pico, "em que ano", "subiu mais", "média móvel"…): manda o recorte do gráfico mais
     relevante ao `qwen2.5vl:3b`.
   - Nos outros casos: `qwen2.5:7b` com os trechos e a instrução de copiar o número exato.

## Instalação

```
python -m venv .venv
.venv\Scripts\pip install pymupdf numpy requests rank_bm25
ollama pull bge-m3
.venv\Scripts\python build_index.py
.venv\Scripts\python answer.py "Qual foi o IPCA observado em julho de 2026?"
```

## Resultados (`python eval.py`, CPU, cerca de 50 s por pergunta)

| Rodada | tabela | gráfico | texto | total |
|---|---|---|---|---|
| 1: busca híbrida + linhas de tabela rotuladas + visão nos gráficos | 5/7 | 3/7 | 7/8 | 15/22 |
| 2: + filtro de período, prompt "para quanto subiu = novo nível", expansão de termos | 7/7 | 4/7 | 8/8 | **19/22** |

Erros que sobraram, todos de gráfico:
- c02 (pico do TPU) e c03 (mínimo da DLSP): o gráfico certo é recuperado, mas o `qwen2.5vl:3b` lê o ano errado
  no eixo.
- c07 (exportações em MM3M): a busca prefere "Exportações brasileiras para EUA" ao gráfico da Balança Comercial.

Próximos passos possíveis: mandar ao modelo de visão os 2 ou 3 gráficos mais bem colocados, ou extrair as
séries dos gráficos vetoriais pelos desenhos do PDF (`page.get_drawings()`), em vez de depender da leitura
visual.
