"""Draw the article's charts as plain SVG, in the blog's style, in Portuguese and English.

Reads results/ (run analyze.py first). Usage: python charts.py OUTPUT_DIR
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FONT = 'font-family="Poppins, ui-sans-serif, system-ui, sans-serif"'
INK, GRAY, GRID, MAGENTA, BLUE = "#1e1e1e", "#495057", "#e9ecef", "#D9348C", "#a5d8ff"
W, H, L, R, T, B = 760, 500, 70, 730, 70, 400

TEXT = {
    "pt": {
        "forecast_file": "previsao-realizado.pt.svg",
        "forecast_label": "IPCA previsto em dezembro do ano anterior e IPCA realizado, 2001 a 2025",
        "realized": "IPCA realizado",
        "focus": "Focus em dezembro do ano anterior",
        "y_forecast": "IPCA no ano (%)",
        "horizon_file": "erro-por-horizonte.pt.svg",
        "horizon_label": "Erro médio do Focus conforme a antecedência da previsão",
        "x_horizon": "Meses antes do fim do ano",
        "y_horizon": "Erro médio absoluto (p.p.)",
        "jan": "virada do ano",
        "jul": "fim de julho",
        "source": "Dados: Banco Central do Brasil (Focus, SGS 13522)",
        "dec": ",",
    },
    "en": {
        "forecast_file": "forecast-realized.svg",
        "forecast_label": "IPCA forecast in December of the year before and IPCA realized, 2001 to 2025",
        "realized": "IPCA realized",
        "focus": "Focus in December of the year before",
        "y_forecast": "IPCA in the year (%)",
        "horizon_file": "error-by-horizon.svg",
        "horizon_label": "Mean Focus error by how early the forecast was made",
        "x_horizon": "Months before the end of the year",
        "y_horizon": "Mean absolute error (p.p.)",
        "jan": "turn of the year",
        "jul": "end of July",
        "source": "Data: Central Bank of Brazil (Focus, SGS 13522)",
        "dec": ".",
    },
}


def text(x, y, s, size=14, anchor="start", fill=GRAY, weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}"{w} text-anchor="{anchor}" fill="{fill}"{extra}>{s}</text>'


def num(v, t, digits=1):
    return f"{v:.{digits}f}".replace(".", t["dec"])


def frame(label, y_ticks, sy, y_title, t):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{label}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for v in y_ticks:
        y = sy(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(text(L - 12, y + 5, num(v, t, 0 if float(v).is_integer() else 1), anchor="end"))
    out.append(text(20, (T + B) / 2, y_title, size=15, anchor="middle", fill=INK, weight=500,
                    extra=f' transform="rotate(-90 20 {(T + B) / 2:.1f})"'))
    out.append(text(R, H - 14, t["source"], size=13, anchor="end"))
    return out


def forecast_chart(t, table):
    years = list(table.index)
    sx = lambda y: L + 20 + (y - years[0]) * (R - L - 40) / (years[-1] - years[0])
    sy = lambda v: B - v * (B - T) / 14
    out = frame(t["forecast_label"], range(0, 15, 2), sy, t["y_forecast"], t)
    for y in years:
        if y % 4 == 1:
            out.append(text(sx(y), B + 24, y, anchor="middle"))
    realized = " ".join(f"{sx(y):.1f},{sy(v):.1f}" for y, v in table["realized"].items())
    focus = " ".join(f"{sx(y):.1f},{sy(v):.1f}" for y, v in table["focus"].items())
    out.append(f'<polyline points="{realized}" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    out.append(f'<polyline points="{focus}" fill="none" stroke="{MAGENTA}" stroke-width="2.5" stroke-dasharray="7 5"/>')
    for y, v in table["realized"].items():
        out.append(f'<circle cx="{sx(y):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{INK}"/>')
    for y, v in table["focus"].items():
        out.append(f'<rect x="{sx(y) - 4:.1f}" y="{sy(v) - 4:.1f}" width="8" height="8" fill="{MAGENTA}"/>')
    for y in (2002, 2015, 2021):
        out.append(text(sx(y), sy(table.loc[y, "realized"]) - 12, y, anchor="middle", fill=INK, weight=500))
    # legend
    out.append(f'<line x1="{L + 330}" y1="34" x2="{L + 360}" y2="34" stroke="{INK}" stroke-width="2.5"/>')
    out.append(text(L + 368, 39, t["realized"], fill=INK))
    out.append(f'<line x1="{L + 330}" y1="56" x2="{L + 360}" y2="56" stroke="{MAGENTA}" stroke-width="2.5" stroke-dasharray="7 5"/>')
    out.append(text(L + 368, 61, t["focus"], fill=INK))
    return "\n".join(out + ["</svg>"]) + "\n"


def horizon_chart(t, h):
    h = h.sort_values("months_before_end", ascending=False)
    sx = lambda m: L + 20 + (23 - m) * (R - L - 40) / 23
    sy = lambda v: B - v * (B - T) / 2.2
    out = frame(t["horizon_label"], [0, 0.5, 1, 1.5, 2], sy, t["y_horizon"], t)
    for m in (23, 18, 12, 6, 0):
        out.append(text(sx(m), B + 24, m, anchor="middle"))
    out.append(text((L + R) / 2, B + 50, t["x_horizon"], size=15, anchor="middle", fill=INK, weight=500))
    pts = " ".join(f"{sx(m):.1f},{sy(v):.1f}" for m, v in zip(h["months_before_end"], h["mae"]))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{MAGENTA}" stroke-width="2.5"/>')
    for m, v in zip(h["months_before_end"], h["mae"]):
        out.append(f'<circle cx="{sx(m):.1f}" cy="{sy(v):.1f}" r="4" fill="{MAGENTA}"/>')
    # The forecast at h months is the last survey of month (December - h): 12 = end of the year before.
    for m, label in ((12, t["jan"]), (5, t["jul"])):
        x = sx(m)
        out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}" stroke="{GRAY}" stroke-width="1" stroke-dasharray="3 4"/>')
        out.append(text(x + 6, T + 14, label, size=13))
    for m in (12, 0):
        v = h.loc[h["months_before_end"] == m, "mae"].iloc[0]
        out.append(text(sx(m), sy(v) - 12, num(v, t, 2), anchor="middle", fill=INK, weight=500))
    return "\n".join(out + ["</svg>"]) + "\n"


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(HERE / "results/annual_forecasts.csv", index_col="year")
    h = pd.read_csv(HERE / "results/by_horizon.csv")
    for t in TEXT.values():
        (out_dir / t["forecast_file"]).write_text(forecast_chart(t, table), encoding="utf-8")
        (out_dir / t["horizon_file"]).write_text(horizon_chart(t, h), encoding="utf-8")
        print(t["forecast_file"], t["horizon_file"])


if __name__ == "__main__":
    main()
