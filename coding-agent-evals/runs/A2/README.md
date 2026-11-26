# Perguntas e respostas sobre o RPM de setembro de 2026

Sistema RAG 100% local sobre `rpm202609p.pdf`.

## Uso
```
.venv\Scripts\python answer.py "Qual a projeção de crescimento do PIB para 2026?"
# -> A projeção de crescimento do PIB para 2026 é de 1,8%.
```
Requer o Ollama rodando em `http://localhost:11434` com `qwen2.5:7b` e `bge-m3`.

## Como funciona
1. **`ingest.py`** (já executado; gera `index_chunks.json` e `index_vecs.npy`):
   - extrai o texto de cada página com PyMuPDF (remove cabeçalho repetido);
   - divide em trechos de ~1400 caracteres com sobreposição de 300, anotados com página e título de seção/boxe;
   - extrai também as tabelas (`page.find_tables()`) como trechos próprios e as **lineariza**: cada valor vira `rótulo → período: valor` (ex.: `IPCA Administrados → 2026 IV: 5,3`), com anos propagados sobre os trimestres e células mescladas separadas. Sem isso o modelo confundia colunas;
   - gera embeddings com `bge-m3` (Ollama).
   Se o índice não existir, `answer.py` o recria automaticamente (~5 min em CPU).
2. **`answer.py`**:
   - recuperação híbrida: similaridade de cosseno (bge-m3) + BM25 (rank_bm25), combinadas por Reciprocal Rank Fusion; top‑8 trechos;
   - geração com `qwen2.5:7b` (temperatura 0, contexto 8k) e um prompt que exige uma frase curta em português, com o número/nome exato do texto;
   - imprime apenas a primeira linha da resposta. `RPM_DEBUG=1` mostra os trechos recuperados; `RPM_LLM=qwen2.5:3b` troca o modelo (mais rápido, menos preciso).

## Avaliação
12 perguntas de teste com resposta conferida no PDF (PIB 2026 e 2º tri, projeção no horizonte relevante, Selic 13,75%, probabilidade de 90%, IPCA de agosto, IPCA de dezembro, IPCA administrados no 4º tri de 2026, déficit em transações correntes, crédito em 2027, agropecuária, IDP): 12/12 corretas após a linearização das tabelas e o ajuste do prompt (antes: 10/12).

## Observações
- Nesta máquina o Ollama roda só em CPU; cada resposta leva de ~15 s a ~4 min com o 7b (mais na primeira chamada, quando o modelo ainda é carregado) (`keep_alive` de 60 min mantém os modelos na memória).
- Valores que só existem dentro de gráficos (imagens) não são recuperáveis; tabelas e texto são.
- Dependências: `pymupdf numpy requests rank_bm25` (instaladas em `.venv`).
