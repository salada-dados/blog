"""Constrói o índice do RPM: trechos de texto, linhas de tabela rotuladas e gráficos (com recorte em PNG).

Uso: .venv/Scripts/python build_index.py
Gera index/chunks.json, index/emb.npy e index/charts/*.png
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
import requests

HERE = Path(__file__).parent
PDF = HERE / "rpm202609p.pdf"
OUT = HERE / "index"
OLLAMA = "http://localhost:11434"
EMB_MODEL = "bge-m3"

NUM = re.compile(r"^\[?[-–+]?\d[\d.]*(,\d+)?%?\]?[¹²³⁴⁵]?$|^[-–]$|^\.\.\.$|^n\.d\.$")
CAPTION = re.compile(r"^(Tabela|Gráfico|Figura|Quadro)\s+[\dA-Z][\d.]*\s*[–-]")
YEAR = re.compile(r"^(19|20)\d\d$")
QUARTERS = {"I": 1, "II": 2, "III": 3, "IV": 4, "1º": 1, "2º": 2, "3º": 3, "4º": 4}
MONTHS = {m: i for i, m in enumerate("jan fev mar abr mai jun jul ago set out nov dez".split(), 1)}


def page_lines(page, ytol=3.0):
    words = page.get_text("words")
    words = [w for w in words if w[4].strip()]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1]["y"] - yc) <= ytol:
            rows[-1]["w"].append(w)
        else:
            rows.append({"y": yc, "w": [w]})
    for r in rows:
        r["w"].sort(key=lambda w: w[0])
    return rows


def phrases(words, gap=5.0):
    """Junta palavras vizinhas em frases: [(x0, x1, texto)]."""
    out = []
    for w in words:
        if out and w[0] - out[-1][1] <= gap:
            out[-1] = (out[-1][0], w[2], out[-1][2] + " " + w[4])
        else:
            out.append((w[0], w[2], w[4]))
    return out


def is_num(t):
    return bool(NUM.match(t.strip()))


def seq_key(t):
    t = t.strip().lower().rstrip(".")
    if t.upper() in QUARTERS:
        return QUARTERS[t.upper()]
    if t[:3] in MONTHS and len(t) <= 9:
        return MONTHS[t[:3]]
    return None


# ---------------------------------------------------------------- regiões
def find_captions(lines):
    caps = []
    for li, r in enumerate(lines):
        for ph in phrases(r["w"], gap=8):
            if CAPTION.match(ph[2]):
                caps.append({"li": li, "y": r["y"], "x0": ph[0], "text": ph[2],
                             "kind": ph[2].split()[0]})
    return caps


def caption_text(lines, cap, x0, x1):
    """Título completo (pode continuar na linha seguinte)."""
    txt = " ".join(w[4] for w in lines[cap["li"]]["w"] if x0 <= w[0] < x1 and w[0] >= cap["x0"] - 2)
    nxt = lines[cap["li"] + 1] if cap["li"] + 1 < len(lines) else None
    if nxt and nxt["y"] - cap["y"] < 13:
        ws = [w for w in nxt["w"] if x0 <= w[0] < x1]
        if ws and abs(ws[0][0] - cap["x0"]) < 4 and not any(is_num(w[4]) for w in ws):
            txt += " " + " ".join(w[4] for w in ws)
    return txt


def regions(page, lines):
    W = page.rect.width
    caps = find_captions(lines)
    regs = []
    for c in caps:
        # faixa horizontal: meia página se houver outra legenda ao lado ou se começar à direita
        side = [o for o in caps if o is not c and abs(o["y"] - c["y"]) < 15]
        if c["x0"] > W * 0.47:
            x0, x1 = W * 0.47, W
        elif side and side[0]["x0"] > c["x0"]:
            x0, x1 = 0, side[0]["x0"] - 3
        elif side:
            x0, x1 = side[0]["x0"] + 1, W
        else:
            x0, x1 = 0, W
        # fim: linha "Fonte"/"Fontes" nessa faixa
        y_end, li_end, y_src = None, None, None
        last = c["li"]
        for lj in range(c["li"] + 1, len(lines)):
            ws = [w for w in lines[lj]["w"] if x0 <= w[0] < x1]
            if not ws:
                continue
            if any(CAPTION.match(p[2]) for p in phrases(ws, 8)) and lines[lj]["y"] - c["y"] > 20:
                y_end, li_end = lines[last]["y"], last
                break
            if lines[lj]["y"] - lines[last]["y"] > 32 and lj - c["li"] > 3:
                y_end, li_end = lines[last]["y"], last
                break
            if re.match(r"^(Fontes?|Elaboração|Nota|Obs)\b", ws[0][4]):
                y_end, li_end = lines[lj]["y"], lj
                y_src = lines[lj]["y"]
                # inclui notas logo abaixo (1/ ..., Nota:)
                for lk in range(lj + 1, min(lj + 6, len(lines))):
                    ws2 = [w for w in lines[lk]["w"] if x0 <= w[0] < x1]
                    if ws2 and re.match(r"^(\d+/|Nota|Obs|Fonte|\*)", ws2[0][4]) and lines[lk]["y"] - lines[lk - 1]["y"] < 14:
                        y_end, li_end = lines[lk]["y"], lk
                    else:
                        break
                break
            last = lj
        if y_end is None:
            continue
        regs.append({**c, "x0r": x0, "x1r": x1, "y_end": y_end, "li_end": li_end,
                     "title": caption_text(lines, c, x0, x1), "y_src": y_src or y_end})
    return regs


# ---------------------------------------------------------------- tabelas
def table_rows(lines, reg):
    x0r, x1r = reg["x0r"], reg["x1r"]
    body = []
    for lj in range(reg["li"] + 1, reg["li_end"]):
        ws = [w for w in lines[lj]["w"] if x0r <= w[0] < x1r]
        if ws:
            body.append(ws)
    # linha de dados: tem rótulo de texto seguido de números, ou só números com coluna à direita
    def split(ws):
        """Rótulo = tokens iniciais; células = sequência final de números."""
        k = len(ws)
        while k > 0 and is_num(ws[k - 1][4]):
            k -= 1
        lab, nums = ws[:k], ws[k:]
        if not lab and nums and nums[0][0] < reg["x0"] + 60:
            lab, nums = nums[:1], nums[1:]
        return lab, nums

    first_data = None
    for i, ws in enumerate(body):
        lab, nums = split(ws)
        if lab and any(re.search(r"\d", w[4]) for w in nums) and not YEAR.match(nums[0][4]):
            first_data = i
            break
    if first_data is None:
        return None
    header, data = body[:first_data], body[first_data:]
    # colunas: agrupa os centros dos números
    centers = sorted((w[0] + w[2]) / 2 for ws in data for w in split(ws)[1])
    cols = []
    for c in centers:
        if cols and c - cols[-1][-1] < 9:
            cols[-1].append(c)
        else:
            cols.append([c])
    cols = [(min(c), max(c), sum(c) / len(c)) for c in cols if len(c) >= 1]
    colx = [c[2] for c in cols]
    left_lim = min(c[0] for c in cols) - 12
    if not cols:
        return None

    def col_of(x):
        return min(range(len(colx)), key=lambda k: abs(colx[k] - x))

    # cabeçalho
    hlines = []
    for ws in header:
        phs = [p for p in phrases(ws, 6) if p[1] > left_lim]
        if phs:
            hlines.append(phs)
    labels = [[] for _ in colx]
    unit = ""
    # linha de períodos (I, II.. ou meses) para segmentar anos
    seq_line = None
    for phs in hlines:
        if len(phs) >= 2 and all(seq_key(p[2]) for p in phs):
            seq_line = phs
    seq_of_col = [None] * len(colx)
    if seq_line:
        for p in seq_line:
            seq_of_col[col_of((p[0] + p[1]) / 2)] = seq_key(p[2])
    for phs in hlines:
        if len(phs) == 1 and phs[0][2].strip() in ("%", "R$ bilhões", "US$ bilhões") or re.match(r"^(%|R\$|US\$)", phs[0][2]) and len(phs) == 1:
            unit = phs[0][2]
            continue
        years = [p for p in phs if YEAR.match(p[2].strip())]
        others = [p for p in phs if not YEAR.match(p[2].strip())]
        # frases não-ano: aplicam-se às colunas que cobrem (com tolerância) ou à mais próxima
        spacing = (colx[-1] - colx[0]) / max(1, len(colx) - 1) if len(colx) > 1 else 30
        wide = len(others) >= 2 and sum(p[1] - p[0] for p in others) / len(others) > 1.6 * spacing
        if wide:
            # fronteira entre frases vizinhas = maior intervalo entre colunas
            bounds = []
            for a, b in zip(others, others[1:]):
                ca, cb = (a[0] + a[1]) / 2, (b[0] + b[1]) / 2
                ks = [k for k in range(len(colx) - 1) if ca - 10 <= colx[k] and colx[k + 1] <= cb + 10]
                if ks:
                    kb = max(ks, key=lambda k: colx[k + 1] - colx[k])
                    bounds.append((colx[kb] + colx[kb + 1]) / 2)
                else:
                    bounds.append((a[1] + b[0]) / 2)
            for k, c in enumerate(colx):
                idx = sum(c > bd for bd in bounds)
                labels[k].append(("o", others[idx][2]))
            others = []
        for p in others:
            hit = [k for k, c in enumerate(cols) if c[0] - 8 <= (p[0] + p[1]) / 2 <= c[1] + 8 or (p[0] - 4 <= c[2] <= p[1] + 4)]
            if not hit:
                hit = [col_of((p[0] + p[1]) / 2)]
            # frases largas e únicas na linha (ex. "Variação anual") cobrem um bloco
            for k in hit:
                labels[k].append(("o", p[2]))
        if not years:
            continue
        if len(years) == len(colx) or all(any(abs((p[0] + p[1]) / 2 - c) < 8 for c in colx) for p in years) and len(years) >= len(colx) - 1:
            for p in years:
                labels[col_of((p[0] + p[1]) / 2)].append(("y", p[2]))
            continue
        # anos que abrangem várias colunas
        direct = {}
        for p in years:
            k = col_of((p[0] + p[1]) / 2)
            if abs(colx[k] - (p[0] + p[1]) / 2) < 6 and seq_of_col[k] is None:
                direct[k] = p
        assigned = [None] * len(colx)
        for k, v in direct.items():
            assigned[k] = v[2]
        rest = [k for k in range(len(colx)) if assigned[k] is None]
        rest_years = [p for p in years if not any(p is v for v in direct.values())]
        if seq_line and rest_years:
            # segmenta colunas restantes por reinício da sequência
            segs, prev = [], None
            for k in rest:
                s = seq_of_col[k]
                if not segs or (s is not None and prev is not None and s <= prev) or (s is None and prev is None and False):
                    segs.append([k])
                else:
                    segs[-1].append(k)
                if s is not None:
                    prev = s
            if len(segs) == len(rest_years):
                for seg, p in zip(segs, sorted(rest_years, key=lambda p: p[0])):
                    for k in seg:
                        assigned[k] = p[2]
        if any(assigned[k] is None for k in rest):
            # ano alinhado à esquerda do bloco: vale até o próximo ano
            for k in rest:
                if assigned[k] is None:
                    lefts = [p for p in rest_years if p[0] - 6 <= colx[k]]
                    if lefts:
                        assigned[k] = max(lefts, key=lambda p: p[0])[2]
                    elif rest_years:
                        assigned[k] = min(rest_years, key=lambda p: p[0])[2]
        for k in range(len(colx)):
            if assigned[k]:
                labels[k].append(("y", assigned[k]))
    # ordena rótulos: outros (de cima) + ano + período
    col_names = []
    for k in range(len(colx)):
        parts = [t for _, t in labels[k]]
        col_names.append(" ".join(parts))

    rows_out, group = [], ""
    for ws in data:
        lab, nums = split(ws)
        label = " ".join(w[4] for w in lab).strip()
        if not nums:
            if label and not re.match(r"^(\d+/|Nota)", label):
                group = label
            continue
        cells = {}
        for p in phrases(nums, 3):
            k = col_of((p[0] + p[1]) / 2)
            cells[k] = (cells[k] + " " + p[2]) if k in cells else p[2]
        if not label:
            label = "(linha)"
        rows_out.append({"label": label, "group": group,
                         "cells": [(col_names[k], v) for k, v in sorted(cells.items())]})
    notes = []
    for lj in range(reg["li_end"] - 0, reg["li_end"] + 1):
        pass
    return {"unit": unit, "cols": col_names, "rows": rows_out}


def region_text(lines, reg):
    out = []
    for lj in range(reg["li"], reg["li_end"] + 1):
        ws = [w for w in lines[lj]["w"] if reg["x0r"] <= w[0] < reg["x1r"]]
        if ws:
            out.append(" ".join(w[4] for w in ws))
    return out


# ---------------------------------------------------------------- texto corrido
def prose_chunks(page, pno, regs, target=170):
    blocks = page.get_text("blocks")
    chunks, cur = [], []
    H = page.rect.height

    def inside(b):
        for r in regs:
            if r["x0r"] - 2 <= b[0] and b[2] <= r["x1r"] + 2 and r["y"] - 8 <= b[1] and b[3] <= r["y_end"] + 30:
                return True
        return False

    for b in blocks:
        if b[6] != 0:
            continue
        t = re.sub(r"\s+", " ", b[4]).strip()
        if not t or b[1] > H - 45:  # rodapé
            continue
        if inside(b):
            continue
        words = len(t.split())
        if words < 3 and not cur:
            continue
        cur.append(t)
        if sum(len(x.split()) for x in cur) >= target:
            chunks.append(" ".join(cur))
            cur = [cur[-1]] if len(cur[-1].split()) < 60 else []
    if cur and (not chunks or " ".join(cur) not in chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def embed(texts, bs=16):
    vecs = []
    for i in range(0, len(texts), bs):
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMB_MODEL, "input": texts[i:i + bs]}, timeout=600)
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
        print(f"  emb {i + bs}/{len(texts)}", file=sys.stderr)
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    import shutil
    shutil.rmtree(OUT / "charts", ignore_errors=True)
    (OUT / "charts").mkdir(exist_ok=True)
    doc = pymupdf.open(PDF)
    chunks = []
    for pno, page in enumerate(doc, 1):
        lines = page_lines(page)
        regs = regions(page, lines)
        for reg in regs:
            title = reg["title"]
            if reg["kind"] == "Tabela":
                tab = table_rows(lines, reg)
                if not tab:
                    chunks.append({"type": "table", "page": pno, "title": title,
                                   "text": f"{title}\n" + "\n".join(region_text(lines, reg))})
                    continue
                for row in tab["rows"]:
                    lab = (row["group"] + " – " if row["group"] else "") + row["label"]
                    cells = "; ".join(f"{c or '?'}: {v}" for c, v in row["cells"])
                    unit = f" ({tab['unit']})" if tab["unit"] else ""
                    chunks.append({"type": "table_row", "page": pno, "title": title,
                                   "text": f"{title}{unit} | {lab} | {cells}"})
            else:
                ws = [w for lj in range(reg["li"], reg["li_end"] + 1) for w in lines[lj]["w"]
                      if reg["x0r"] <= w[0] < reg["x1r"] and lines[lj]["y"] <= reg["y_src"] + 1]
                bx0 = min(w[0] for w in ws) - 4
                bx1 = max(w[2] for w in ws) + 4
                rect = pymupdf.Rect(max(bx0, 20), reg["y"] - 8, min(bx1, page.rect.width - 15), reg["y_src"] + 6)
                img = OUT / "charts" / f"p{pno}_{len(chunks)}.png"
                page.get_pixmap(clip=rect, dpi=150).save(img)
                txt = region_text(lines, reg)
                body = " ".join(t for t in txt[1:] if not re.fullmatch(r"[-\d., %IVXa-zç/]{1,12}", t))
                chunks.append({"type": "chart", "page": pno, "title": title, "img": str(img.relative_to(HERE)),
                               "text": f"{title} | {body}"[:1200]})
        for t in prose_chunks(page, pno, regs):
            chunks.append({"type": "text", "page": pno, "title": "", "text": t})
    print(f"{len(chunks)} chunks:", {k: sum(c['type'] == k for c in chunks) for k in ('text', 'table_row', 'table', 'chart')})
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=0), encoding="utf-8")
    if "--noemb" not in sys.argv:
        emb = embed([c["text"][:2000] for c in chunks])
        np.save(OUT / "emb.npy", emb)


if __name__ == "__main__":
    main()
