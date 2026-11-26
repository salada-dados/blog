# Perguntas e respostas sobre o RPM de set/2026 (BCB)

Uso (o Ollama precisa estar rodando em `localhost:11434`):

```
.venv\Scripts\python answer.py "Qual a projeção de inflação para o 1º trimestre de 2028?"
```

## Como funciona
1. **`build_index.py`** (já executado; o resultado fica em `index/`): extrai o texto do `rpm202609p.pdf` com PyMuPDF.
   As linhas são remontadas a partir das coordenadas das palavras, para que cada linha de tabela vire uma linha
   de texto (ex.: `Agropecuária 0,0 -1,1 16,3 ...`), e o cabeçalho repetido das páginas é removido.
   O texto de cada página é dividido em trechos de ~1.100 caracteres com sobreposição (345 trechos), e cada trecho
   recebe um embedding do `intfloat/multilingual-e5-base` (sentence-transformers, CPU).
2. **`answer.py`**: faz uma busca híbrida, com **BM25** (texto sem acentos) e **similaridade de embeddings**,
   juntando as duas listas por *reciprocal rank fusion*. Os 5 melhores trechos, com a página de cada um, vão
   para o **`qwen2.5:7b`** (Ollama, temperatura 0). O prompt pede uma única frase curta com o número
   ou o nome pedido, e um exemplo no prompt mostra o formato esperado.

Variáveis opcionais: `RPM_LLM` (ex.: `qwen2.5:3b`, que é mais rápido e menos preciso), `RPM_TOPK` (padrão 5),
`RPM_DEBUG=1` (mostra no stderr as páginas recuperadas).

Para refazer o índice: `.venv\Scripts\python build_index.py`.

## Testes feitos
| Pergunta | Resposta | Certa? |
|---|---|---|
| Projeção de inflação no 1º tri de 2028 (cenário de referência) | 3,2% | sim |
| Crescimento do PIB no 2º tri de 2026 | 0,5% | sim |
| Taxa de desocupação no trimestre encerrado em julho | 5,4% | sim |
| Projeção de crescimento do PIB para 2027 | 1,4% | sim |
| Número da reunião do Copom usada no RPM | "A reunião do Copom considerada neste RPM foi a 281ª reunião." | sim |
| Crescimento da agropecuária no 1º tri de 2026 | "A agropecuária cresceu 2,8% no primeiro trimestre de 2026." | **não** (o certo é 2,3%) |

As quatro primeiras respostas saíram de versões anteriores do sistema: só o número, e as de 2027 e a da reunião com o `qwen2.5:3b`. As duas últimas linhas já são do `answer.py` atual. As três primeiras não foram rodadas de novo depois que o formato em frase foi adicionado.

## Limitações
- A máquina não tem GPU. Cada resposta leva cerca de **1,5 a 2 minutos**, e quase todo esse tempo é o 7b lendo o contexto.
- Em tabelas largas (colunas anuais seguidas de trimestrais), o modelo às vezes pega a coluna errada.
  Foi isso que aconteceu no erro da agropecuária acima.
- Valores que só aparecem em gráficos (como imagem) não são lidos. O `qwen2.5vl:3b` poderia descrever essas páginas, mas isso não foi implementado.
