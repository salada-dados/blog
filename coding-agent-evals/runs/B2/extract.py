"""Extrai do PDF: texto corrido, tabelas (linearizadas célula a célula) e gráficos vetoriais
(séries reconstruídas a partir dos caminhos do PDF). Gera data/chunks.json."""
import json
import re
from pathlib import Path

import pymupdf as fitz

HERE = Path(__file__).parent
PDF = HERE / "rpm202609p.pdf"
NUM = re.compile(r"^(?:[-–−]?\d[\d.]*(,\d+)?%?|\([-–−]?\d[\d.]*(,\d+)?\)%?|\[[-–−]?\d[\d.]*(,\d+)?\]%?)$")
MONTHS = {m: i for i, m in enumerate(
    ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"], 1)}
MONTH_NAMES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def find_caps(lines, key):
    caps = []
    for ln in lines:
        ws = ln["w"]
        for i, w in enumerate(ws):
            if w[4] == key and i + 1 < len(ws) and re.match(r"\d", ws[i + 1][4]):
                cw = [w]
                for v in ws[i + 1:]:
                    if v[4] == key or v[0] - cw[-1][2] > 15:
                        break
                    cw.append(v)
                caps.append({"y": ln["y"], "w": cw})
    return caps


def parse_num(s):
    s = s.replace("−", "-").replace("–", "-").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("%", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def fmt(v):
    if v is None:
        return "?"
    a = abs(v)
    if a >= 100:
        s = f"{v:.0f}"
    elif a >= 10:
        s = f"{v:.1f}"
    else:
        s = f"{v:.2f}"
    return s.replace(".", ",")


def lines_of(words, tol=3):
    """Agrupa palavras (x0,y0,x1,y1,txt) em linhas por y."""
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if lines and abs(lines[-1]["y"] - yc) <= tol:
            lines[-1]["w"].append(w)
        else:
            lines.append({"y": yc, "w": [w]})
    for ln in lines:
        ln["w"].sort(key=lambda w: w[0])
    return lines


def phrases(ws, gap=4.5):
    out = []
    for w in sorted(ws, key=lambda w: w[0]):
        if out and w[0] - out[-1][2] <= gap:
            o = out[-1]
            out[-1] = (o[0], min(o[1], w[1]), w[2], max(o[3], w[3]), o[4] + " " + w[4])
        else:
            out.append(tuple(w[:5]))
    return out


def merge_numbers(ws):
    """Junta '2' '000' -> '2000' e '-' '1' -> '-1' quando colados."""
    out = []
    for w in sorted(ws, key=lambda w: w[0]):
        t = w[4]
        if out:
            p = out[-1]
            g = w[0] - p[2]
            if p[4] in ("-", "–", "−") and NUM.match(t) and g < 3:
                out[-1] = (p[0], p[1], w[2], w[3], "-" + t)
                continue
            if NUM.match(p[4]) and re.fullmatch(r"\d{3}(,\d+)?", t) and g < 3.5 and re.fullmatch(r"-?\d{1,3}", p[4]):
                out[-1] = (p[0], p[1], w[2], w[3], p[4] + t)
                continue
        out.append(tuple(w[:5]))
    return out


# ---------------------------------------------------------------- tabelas
def partition(cols, targets):
    """Divide colunas (ordenadas) em len(targets) grupos contíguos minimizando |média - alvo|."""
    n, k = len(cols), len(targets)
    INF = 1e18
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    arg = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0
    for j in range(1, k + 1):
        for i in range(1, n + 1):
            for s in range(j - 1, i):
                if best[j - 1][s] >= INF:
                    continue
                m = sum(cols[s:i]) / (i - s)
                c = best[j - 1][s] + abs(m - targets[j - 1])
                if c < best[j][i]:
                    best[j][i], arg[j][i] = c, s
    groups, i = [], n
    for j in range(k, 0, -1):
        s = arg[j][i]
        groups.append((s, i))
        i = s
    return groups[::-1]


def parse_table(title, words):
    lines = lines_of(words)
    rows = []
    for ln in lines:
        ws = merge_numbers(ln["w"])
        nums = []
        for w in reversed(ws):
            if NUM.match(w[4]) or w[4] in ("-", "–", "…", "...", "n.d.", "n.d"):
                nums.insert(0, w)
            else:
                break
        while nums and nums[0][4] in ("-", "–") and len(nums) < len(ws) and                 nums[0][0] - ws[ws.index(nums[0]) - 1][2] < 6:
            nums.pop(0)
        rows.append({"y": ln["y"], "ws": ws, "nums": nums})
    # primeira linha de dados: rótulo de texto à esquerda + >=2 números
    first = None
    for i, r in enumerate(rows):
        lab = [w for w in r["ws"] if w not in r["nums"]]
        if len(r["nums"]) >= 2 and lab and lab[0][0] < r["nums"][0][0] \
                and not re.fullmatch(r"(19|20)\d\d", r["nums"][0][4]):
            first = i
            break
    if first is None:
        return []
    data = rows[first:]
    # colunas: agrupa centros dos números
    centers = sorted((w[0] + w[2]) / 2 for r in data for w in r["nums"]
                     if not any(x[0] < w[0] and x not in r["nums"] and x[0] > w[0] - 1 for x in r["ws"]))
    cols = []
    for c in centers:
        if cols and c - cols[-1][-1] < 7:
            cols[-1].append(c)
        else:
            cols.append([c])
    cols = [sum(c) / len(c) for c in cols if len(c) >= 1]
    if not cols:
        return []
    x_first = min(cols) - 12
    # cabeçalhos
    headers = [[] for _ in cols]
    for r in rows[:first]:
        before = [len(h) for h in headers]
        ph = [p for p in phrases(r["ws"]) if (p[0] + p[2]) / 2 > x_first]
        if not ph:
            continue
        owned, free_ph = {}, []
        for p in ph:
            pc = (p[0] + p[2]) / 2
            j = min(range(len(cols)), key=lambda j: abs(cols[j] - pc))
            if abs(cols[j] - pc) < 5 and j not in owned:
                owned[j] = p[4]
            else:
                free_ph.append(p)
        free_cols = [j for j in range(len(cols)) if j not in owned]
        for j, t in owned.items():
            headers[j].append(t)
        if free_ph and free_cols and len(free_ph) <= len(free_cols):
            # grupos contíguos dentro dos segmentos livres
            groups = partition([cols[j] for j in free_cols], [(p[0] + p[2]) / 2 for p in free_ph])
            for (s, e), p in zip(groups, free_ph):
                for j in free_cols[s:e]:
                    headers[j].append(p[4])
        elif free_ph:
            for p in free_ph:
                pc = (p[0] + p[2]) / 2
                j = min(free_cols or range(len(cols)), key=lambda j: abs(cols[j] - pc))
                headers[j].append(p[4])
        # cabeçalho que abrange várias colunas: propaga para a direita
        got = [len(headers[j]) > before[j] for j in range(len(cols))]
        last = None
        for j in range(len(cols)):
            if got[j]:
                last = headers[j][-1]
            elif last is not None and not re.fullmatch(r"%|R\$.*", last):
                headers[j].append(last)
    out, section, parent = [], "", ""
    for r in data:
        lab = " ".join(w[4] for w in r["ws"] if w not in r["nums"] and w[2] < x_first + 8).strip()
        if not r["nums"]:
            if lab and not lab.startswith(("Fonte", "1/", "2/", "3/")):
                section = lab
            continue
        if not lab:
            lab = section
        if lab.startswith("d/q"):
            lab = f"{parent} – das quais: {lab[3:].strip()}"
        else:
            parent = lab
        if all(re.fullmatch(r"(19|20)\d\d", w[4]) for w in r["nums"]):
            continue
        cells = []
        for w in r["nums"]:
            c = (w[0] + w[2]) / 2
            j = min(range(len(cols)), key=lambda j: abs(cols[j] - c))
            h = " ".join(headers[j]) or f"col{j + 1}"
            cells.append(f"{h}: {w[4]}")
        pre = f"{section} – " if section and section != lab else ""
        out.append(f"{title} | {pre}{lab} | " + "; ".join(cells))
    return out


def extract_tables(doc):
    chunks = []
    for pn, pg in enumerate(doc):
        words = pg.get_text("words")
        lines = lines_of(words)
        caps = find_caps(lines, "Tabela")
        for ci, cap in enumerate(caps):
            x0 = cap["w"][0][0] - 15
            title = " ".join(w[4] for w in cap["w"])
            end = pg.rect.height - 40
            for ln in lines:
                if ln["y"] > cap["y"] + 5 and ln["w"][0][4].startswith("Fonte") and ln["w"][0][0] >= x0 - 20:
                    end = ln["y"] - 2
                    break
            if ci + 1 < len(caps):
                end = min(end, caps[ci + 1]["y"] - 2)
            x1 = pg.rect.width
            # tabela em meia coluna?
            for c2 in caps:
                if c2 is not cap and abs(c2["y"] - cap["y"]) < 40 and c2["w"][0][0] > x0 + 100:
                    x1 = c2["w"][0][0] - 10
            reg = [w for w in words if cap["y"] + 4 < (w[1] + w[3]) / 2 < end and x0 <= w[0] and w[2] <= x1 + 5]
            rows = parse_table(title, reg)
            # unidade
            for r in rows:
                chunks.append({"page": pn + 1, "kind": "table", "title": title, "text": r})
    return chunks


# ---------------------------------------------------------------- gráficos
def color_name(c):
    return "#%02x%02x%02x" % tuple(int(v * 255) for v in c) if c else "none"


def is_gray(c):
    return c is None or (max(c) - min(c) < 0.06)


def time_of(label_words):
    pass


def build_time_axis(xwords, px0, px1):
    """Retorna função x->(tempo decimal) e tipo, a partir dos rótulos do eixo x."""
    lines = lines_of(xwords, tol=2.5)
    years, months, quarters = [], [], []
    for ln in lines:
        for w in ln["w"]:
            t = w[4].strip().lower().rstrip(".")
            xc = (w[0] + w[2]) / 2
            m = re.fullmatch(r"(19|20)(\d\d)", t)
            if m:
                years.append((xc, int(t), w[1]))
                continue
            m = re.fullmatch(r"([a-zç]{3})[a-z]*[/ -]?(\d{2,4})?", t)
            if m and m.group(1) in MONTHS:
                yr = m.group(2)
                if yr:
                    yr = int(yr) + (2000 if len(yr) == 2 else 0)
                months.append((xc, MONTHS[m.group(1)], w[1], yr))
                continue
            m = re.fullmatch(r"([iv]+|[1-4])t(\d{2})?|t([1-4])|([iv]+)", t)
            if m:
                quarters.append((xc, t, w[1]))
    if years:
        ytop = min(y[2] for y in years)
        years = [y for y in years if y[2] < ytop + 10]
    if months:
        mtop = min(m[2] for m in months)
        months = [m for m in months if m[2] < mtop + 3]
    anchors = []
    for (xc, mo, y, yr) in months:
        if yr is None:
            cand = [yy for yy in years if abs(yy[0] - xc) < 10 and yy[2] > y]
            if cand:
                yr = cand[0][1]
        if yr:
            anchors.append((xc, yr + (mo - 1) / 12))
    if len(anchors) < 2:
        anchors = [(xc, yr + 0.0) for (xc, yr, _) in years]
        kind = "year"
    else:
        kind = "month"
    anchors = sorted(set(anchors))
    if len(anchors) < 2:
        return None, None, years
    # regressão linear
    n = len(anchors)
    mx = sum(a[0] for a in anchors) / n
    mt = sum(a[1] for a in anchors) / n
    sxx = sum((a[0] - mx) ** 2 for a in anchors)
    if sxx == 0:
        return None, None, years
    b = sum((a[0] - mx) * (a[1] - mt) for a in anchors) / sxx
    if b <= 0:
        return None, None, years
    return (lambda x: mt + b * (x - mx)), kind, years


def fit_axis(ticks):
    """ticks: lista (ycoord, valor). Retorna y->valor."""
    if len(ticks) < 2:
        return None
    n = len(ticks)
    my = sum(t[0] for t in ticks) / n
    mv = sum(t[1] for t in ticks) / n
    syy = sum((t[0] - my) ** 2 for t in ticks)
    if syy == 0:
        return None
    b = sum((t[0] - my) * (t[1] - mv) for t in ticks) / syy
    return lambda y: mv + b * (y - my)


def tdate(t, kind):
    yr = int(t + 1e-6)
    if kind == "month" or True:
        mo = int(round((t - yr) * 12)) + 1
        if mo == 13:
            yr, mo = yr + 1, 1
        return f"{MONTH_NAMES[mo - 1]}/{yr}"


def describe_line(name, pts, fy, ft, kind):
    vals = [(ft(x) if ft else None, fy(y)) for x, y in pts]
    if not vals:
        return ""
    vs = [v for _, v in vals]
    imax = max(range(len(vs)), key=lambda i: vs[i])
    imin = min(range(len(vs)), key=lambda i: vs[i])

    def at(i):
        t = vals[i][0]
        return f"{fmt(vs[i])} ({tdate(t, kind) if t is not None else f'ponto {i + 1}'})"
    s = (f"Série '{name}': máximo {at(imax)}, mínimo {at(imin)}, início {at(0)}, fim {at(len(vs) - 1)}.")
    if ft:
        by = {}
        for t, v in vals:
            by.setdefault(int(t + 1e-6), []).append(v)
        if len(by) > 1:
            parts = []
            for yr in sorted(by):
                b = by[yr]
                if len(b) > 1:
                    parts.append(f"{yr}: média {fmt(sum(b) / len(b))} (máx {fmt(max(b))}, mín {fmt(min(b))})")
                else:
                    parts.append(f"{yr}: {fmt(b[0])}")
            s += " Por ano: " + "; ".join(parts) + "."
            ymax = max(by, key=lambda y: max(by[y]))
            ymin = min(by, key=lambda y: min(by[y]))
            s += f" Ano do pico (maior valor): {ymax}. Ano do menor valor: {ymin}."
    return s


def extract_charts(doc):
    chunks = []
    for pn, pg in enumerate(doc):
        words = pg.get_text("words")
        lines = lines_of(words)
        caps = find_caps(lines, "Gráfico")
        if not caps:
            continue
        draws = pg.get_drawings()
        W = pg.rect.width
        for cap in caps:
            cx0 = cap["w"][0][0]
            title = " ".join(w[4] for w in cap["w"])
            # unidade na linha seguinte
            # região
            x0, x1 = cx0 - 25, W - 20
            for c2 in caps:
                if c2 is not cap and abs(c2["y"] - cap["y"]) < 60 and c2["w"][0][0] > cx0 + 100:
                    x1 = c2["w"][0][0] - 15
                if c2 is not cap and abs(c2["y"] - cap["y"]) < 60 and c2["w"][0][0] < cx0 - 100:
                    x0 = cx0 - 25
            y0 = cap["y"] + 5
            nxt = [ln for ln in lines if cap["y"] + 2 < ln["y"] < cap["y"] + 26]
            unit = " ".join(w[4] for ln in nxt[:2] for w in ln["w"] if x0 <= w[0] < x1 - 5 and w[4] != "Gráfico")
            y1 = pg.rect.height - 40
            for ln in lines:
                if ln["y"] > y0 + 20 and ln["w"][0][4].startswith("Fonte") and x0 - 5 <= ln["w"][0][0] <= x1:
                    y1 = ln["y"] - 3
                    break
            for c2 in caps:
                if c2["y"] > cap["y"] + 5 and x0 - 5 <= c2["w"][0][0] <= x1:
                    y1 = min(y1, c2["y"] - 3)
            src = ""
            for ln in lines:
                if abs(ln["y"] - (y1 + 3)) < 2:
                    src = " ".join(w[4] for w in ln["w"] if x0 <= w[0] <= x1)
            notes = " ".join(" ".join(w[4] for w in ln["w"] if x0 <= w[0] <= x1)
                             for ln in lines if y1 + 6 < ln["y"] < y1 + 30 and re.match(r"\d/", ln["w"][0][4]))

            def inreg(r):
                return r.x0 >= x0 - 5 and r.x1 <= x1 + 10 and r.y0 >= y0 and r.y1 <= y1 + 2
            ds = [d for d in draws if inreg(d["rect"])]
            ws = [w for w in words if x0 - 5 <= w[0] and w[2] <= x1 + 10 and y0 < w[1] and w[3] <= y1 + 2]
            # séries de linha e barras
            series, bars, swatches = [], [], []
            for d in ds:
                r = d["rect"]
                if d["type"] == "s" and d.get("color") is not None:
                    col = d["color"]
                    segs = [it for it in d["items"] if it[0] in ("l", "c")]
                    if len(segs) >= 3 and r.width > 20 and not (is_gray(col) and (r.height < 2.5 or min(col) > 0.95)):
                        pts = [(segs[0][1].x, segs[0][1].y)] + [(it[-1].x, it[-1].y) for it in segs]
                        series.append((col, pts, r, bool(d.get("dashes") and d["dashes"].strip("[] 0").strip())))
                    elif len(segs) <= 2 and r.width < 30 and r.height < 3:
                        swatches.append((col, r))
                elif d["type"] in ("f", "fs") and d.get("fill") is not None:
                    col = d["fill"]
                    if col and min(col) > 0.97:
                        continue
                    res = [it for it in d["items"] if it[0] == "re"]
                    if res and all(it[1].width < 6 and it[1].height < 6 for it in res) and len(res) <= 2:
                        swatches.append((col, r))
                        continue
                    for it in res:
                        bars.append((col, it[1]))
                    if not res and len(d["items"]) >= 3 and r.width < 10 and r.height < 10:
                        swatches.append((col, r))
            if not series and not bars:
                continue
            allr = [s[2] for s in series] + [b[1] for b in bars]
            px0 = min(r.x0 for r in allr)
            px1 = max(r.x1 for r in allr)
            py0 = min(r.y0 for r in allr)
            py1 = max(r.y1 for r in allr)
            # eixos pretos
            for d in ds:
                if d["type"] == "s" and d.get("color") is not None and max(d["color"]) < 0.05:
                    r = d["rect"]
                    if r.width < 3 and r.height > 20:
                        py0, py1 = min(py0, r.y0), max(py1, r.y1)
                    if r.height < 3 and r.width > 40:
                        px0, px1 = min(px0, r.x0), max(px1, r.x1)
                        py1 = max(py1, r.y1) if r.y1 < py1 + 15 else py1
            mws = [m for ln in lines_of(ws) for m in merge_numbers(ln["w"])]
            lt = [((w[1] + w[3]) / 2, parse_num(w[4])) for w in mws if NUM.match(w[4]) and w[2] <= px0 + 3
                  and py0 - 6 <= (w[1] + w[3]) / 2 <= py1 + 6 and parse_num(w[4]) is not None]
            rt = [((w[1] + w[3]) / 2, parse_num(w[4])) for w in mws if NUM.match(w[4]) and w[0] >= px1 - 3
                  and py0 - 6 <= (w[1] + w[3]) / 2 <= py1 + 6 and parse_num(w[4]) is not None]
            fl, fr = fit_axis(lt), fit_axis(rt)
            fy_def = fl or fr
            if fy_def is None:
                continue
            xws = [w for w in ws if py1 - 2 < w[1] < py1 + 28 and px0 - 15 < w[0] < px1 + 15]
            ft, kind, years = build_time_axis(xws, px0, px1)
            # legenda
            leg = []
            for col, r in swatches:
                if px0 + 2 < r.x0 < px1 - 2 and py0 + 2 < r.y0 < py1 - 2:
                    continue
                yc = (r.y0 + r.y1) / 2
                txt = [w for w in ws if abs((w[1] + w[3]) / 2 - yc) < 4 and w[0] >= r.x1 - 1]
                txt.sort(key=lambda w: w[0])
                name, last = [], r.x1
                for w in txt:
                    if w[0] - last > 12:
                        break
                    name.append(w[4])
                    last = w[2]
                if name:
                    leg.append((col, " ".join(name)))

            def legname(col, i):
                best = None
                for c, n in leg:
                    dd = sum(abs(a - b) for a, b in zip(c, col))
                    if dd < 0.08 and (best is None or dd < best[0]):
                        best = (dd, n)
                return best[1] if best else f"série {i + 1} (cor {color_name(col)})"

            out = [f"{title} ({unit}). {src} {notes}".strip()]
            if ft is None and years:
                out.append("Eixo x: " + ", ".join(str(y[1]) for y in years))
            for i, (col, pts, r, dashed) in enumerate(series):
                name = legname(col, i) + (" – trecho tracejado/projeção" if dashed else "")
                fy = fr if (fr and "(dir" in name.lower() or (fr and not fl)) else fy_def
                out.append(describe_line(name, pts, fy, ft, kind))
            if bars:
                # categorias: rótulos do eixo x (frases)
                legy = [(r.y0 + r.y1) / 2 for c, r in swatches if not (px0 < r.x0 < px1 and py0 < r.y0 < py1)]
                cw = [w for w in xws if (not NUM.match(w[4]) or re.fullmatch(r"(19|20)\d\d", w[4]))
                      and not any(abs((w[1] + w[3]) / 2 - ly) < 5 for ly in legy)]
                stop = min([w[1] for w in cw if w[4].startswith("Fonte")] + [1e9])
                cw = [w for w in cw if w[1] < stop - 1]
                if legy:
                    cw = [w for w in cw if w[1] < min(legy) - 3]
                # junta rótulos de várias linhas na mesma coluna
                cats = []
                for p in phrases(cw, gap=4):
                    pc = (p[0] + p[2]) / 2
                    for i, q in enumerate(cats):
                        if abs((q[0] + q[2]) / 2 - pc) < 8:
                            cats[i] = (min(q[0], p[0]), q[1], max(q[2], p[2]), p[3], q[4] + " " + p[4])
                            break
                    else:
                        cats.append(p)
                zero = None
                byser = {}
                for col, r in bars:
                    if r.width < 0.5 or r.height < 0.05:
                        continue
                    a, b = fy_def(r.y0), fy_def(r.y1)
                    v = a if abs(a) >= abs(b) else b
                    xc = (r.x0 + r.x1) / 2
                    cat = None
                    if cats:
                        c = min(cats, key=lambda p: abs((p[0] + p[2]) / 2 - xc))
                        cat = c[4]
                    elif ft:
                        cat = tdate(ft(xc), kind)
                    key = legname(col, 0)
                    if key.startswith("série "):
                        key = f"cor {color_name(col)}"
                    byser.setdefault(key, []).append((xc, cat, v))
                for name, items in byser.items():
                    items.sort()
                    m = re.fullmatch(r"(I|II|III|IV) (\d{4})", name)
                    if m:
                        name = f"{name} = {len(m.group(1)) if m.group(1) != 'IV' else 4}º trimestre de {m.group(2)}"
                    vals = "; ".join(f"{c}: {fmt(v)}" for _, c, v in items[:60])
                    mx = max(items, key=lambda t: t[2])
                    mn = min(items, key=lambda t: t[2])
                    out.append(f"Barras '{name}': {vals}. Maior: {mx[1]} ({fmt(mx[2])}); menor: {mn[1]} ({fmt(mn[2])}).")
            chunks.append({"page": pn + 1, "kind": "chart", "title": title, "text": "\n".join(out)})
    return chunks


# ---------------------------------------------------------------- texto
def extract_text(doc):
    chunks = []
    for pn, pg in enumerate(doc):
        blocks = pg.get_text("blocks")
        paras = []
        for b in blocks:
            t = b[4].strip()
            if not t or t.startswith("Setembro 2026"):
                continue
            t = re.sub(r"-\n(?=[a-zà-ú])", "", t)
            t = re.sub(r"\s*\n\s*", " ", t)
            # descarta blocos que são só números/rótulos de gráfico
            letters = sum(ch.isalpha() for ch in t)
            if len(t) < 60 or letters < 0.5 * len(t):
                continue
            paras.append(t)
        # chunks de ~900 caracteres com parágrafos inteiros
        cur = ""
        for p in paras:
            if len(cur) + len(p) > 900 and cur:
                chunks.append({"page": pn + 1, "kind": "text", "title": "", "text": cur.strip()})
                cur = ""
            if len(p) > 1200:
                sents = re.split(r"(?<=[.;])\s+", p)
                for s in sents:
                    if len(cur) + len(s) > 900 and cur:
                        chunks.append({"page": pn + 1, "kind": "text", "title": "", "text": cur.strip()})
                        cur = ""
                    cur += s + " "
            else:
                cur += p + "\n"
        if cur.strip():
            chunks.append({"page": pn + 1, "kind": "text", "title": "", "text": cur.strip()})
    return chunks


if __name__ == "__main__":
    doc = fitz.open(PDF)
    t = extract_text(doc)
    tb = extract_tables(doc)
    ch = extract_charts(doc)
    print(len(t), "text,", len(tb), "table rows,", len(ch), "charts")
    (HERE / "data").mkdir(exist_ok=True)
    json.dump(t + tb + ch, open(HERE / "data" / "chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
