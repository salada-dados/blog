# QA sobre o RPM de setembro de 2026

`python answer.py "pergunta"` → resposta em uma frase curta. Usa somente Ollama local
(`qwen2.5:7b` para gerar, `bge-m3` para embeddings). Rode com o Python do `.venv`.

## Como funciona

1. **Extração (`extract.py`)** — lê o PDF com PyMuPDF e gera `data/chunks.json` com três tipos de trecho:
   - **texto**: blocos de parágrafos (~900 caracteres) por página;
   - **tabelas**: parser próprio por coordenadas de palavras (o `find_tables` do PyMuPDF embaralhava as
     células). Cada linha da tabela vira um trecho do tipo
     `Tabela X – título | rótulo da linha | cabeçalho da coluna: valor; ...`. Cabeçalhos agrupados
     (ex.: "2025" sobre I–IV) são atribuídos por uma partição contígua das colunas;
   - **gráficos**: os gráficos do RPM são vetoriais, então as séries são reconstruídas dos caminhos do
     PDF: eixo y calibrado pelos rótulos numéricos (incluindo eixo direito, "dir"), eixo x de tempo
     pelos rótulos de ano/mês, nomes das séries pela cor da legenda. Para cada série o texto traz
     máximo, mínimo, início, fim, resumo por ano e "ano do pico/menor valor"; para barras, o valor
     por categoria.
2. **Índice (`build_index.py`)** — embeddings `bge-m3` (via Ollama) em `data/emb.npy`.
3. **Resposta (`answer.py`)** — busca híbrida (BM25 + densa, fusão RRF; sinônimos como
   MM3M → "média móvel de três meses"; trechos de gráfico recebem peso extra quando a pergunta
   tem cara de pergunta de gráfico: "em que ano", "pico", "subiu mais"...). Contexto de ~3.800
   caracteres; linhas de tabela são mostradas uma célula por linha ("rótulo, coluna: valor"),
   o que ajuda o 7b a escolher a coluna certa. `qwen2.5:7b`, temperatura 0, uma frase.

Roda só na CPU: cada pergunta leva ~50–60 s (quase tudo é o processamento do prompt pelo 7b).

`check_retrieval.py` mede, sem LLM, se a resposta esperada aparece no contexto recuperado.

## Reconstruir

```
python -m venv .venv && .venv\Scripts\pip install pymupdf numpy requests
.venv\Scripts\python extract.py
.venv\Scripts\python build_index.py
.venv\Scripts\python eval.py
```

## Resultados

Avaliação com `.venv\Scripts\python eval.py`:

| Rodada | Mudança | Tabelas | Gráficos | Texto | Total |
|---|---|---|---|---|---|
| 1 | versão inicial | 5/7 | 4/7 | 8/8 | **17/22** |
| 2 | correções no parser (cabeçalhos, "d/q", unidade dos gráficos), sinônimos, instrução extra no prompt | 3/7 | 4/7 | 8/8 | 15/22 |
| 3 | prompt original + peso extra para gráficos + tabela uma célula por linha (versão atual) | 3/7 | 6/7 | 8/8 | **17/22** |

Na versão atual, o contexto recuperado contém a resposta esperada nas 22 perguntas
(`check_retrieval.py`). Os 5 erros restantes vêm do `qwen2.5:7b`, que lê a linha ou coluna vizinha
na tabela certa (ex.: responde Dez em vez de Nov, ou 2027 I em vez de 2027 IV) ou escolhe um gráfico
parecido (c07). Próximos passos possíveis: filtrar as células da tabela pelo período citado na
pergunta antes de chamar o LLM, ou extrair o valor deterministicamente quando a pergunta cita
linha e período. Não deu para testar dentro do tempo: cada avaliação leva ~18–23 min na CPU.

Nada foi ajustado para respostas específicas do `dev.json`: as mudanças são gerais (parser,
busca, formato do contexto).
