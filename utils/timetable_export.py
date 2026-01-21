from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class ImageExportOptions:
    title: Optional[str] = None
    font_size: int = 10
    cell_height: float = 0.35
    cell_width: float = 1.2


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to a GitHub-flavored Markdown table."""

    # pandas to_markdown requires tabulate in some versions; avoid extra dependency.
    # Implement a small markdown renderer.
    cols = list(df.columns)
    rows = df.astype(str).values.tolist()

    def esc(s: str) -> str:
        return str(s).replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def df_to_png_bytes(df: pd.DataFrame, *, options: ImageExportOptions = ImageExportOptions()) -> bytes:
    """Render a DataFrame as a PNG image (bytes).

    Uses matplotlib's table artist.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = df.shape

    fig_w = max(6.0, float(options.cell_width) * (ncols + 1))
    fig_h = max(2.0, float(options.cell_height) * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if options.title:
        ax.set_title(options.title, fontsize=options.font_size + 2, pad=12)

    tbl = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(options.font_size)
    tbl.scale(1.0, 1.4)

    # Light styling
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0f2f6")
            cell.set_text_props(weight="bold")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
