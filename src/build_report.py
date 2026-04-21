"""把 report/report.md 转为 PDF。

用 pandoc 生成 HTML（带数学 MathJax 支持），再用 weasyprint 转 PDF。
CSS 中指定 Noto CJK 字体以支持中文。
"""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
MD = REPORT_DIR / "report.md"
HTML = REPORT_DIR / "report.html"
PDF = REPORT_DIR / "report.pdf"


CSS = """
@page {
    size: A4;
    margin: 2.2cm 1.8cm;
    @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; }
}
body {
    font-family: "Noto Serif CJK SC", "Noto Sans CJK SC", serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #222;
}
h1 { font-size: 18pt; margin-top: 0; text-align: center; }
h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1.5px solid #aaa; padding-bottom: 4px; }
h3 { font-size: 12pt; margin-top: 1.2em; }
h4 { font-size: 11pt; margin-top: 1em; color: #333; }
p  { margin: 0.5em 0; }
code {
    font-family: "Noto Sans Mono CJK SC", "DejaVu Sans Mono", monospace;
    font-size: 9.5pt;
    background: #f4f4f4;
    padding: 1px 3px;
    border-radius: 2px;
}
pre {
    background: #f4f4f4;
    padding: 8px 10px;
    border-left: 3px solid #666;
    font-size: 9.5pt;
    line-height: 1.35;
    overflow-x: auto;
}
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 0.6em 0; width: 100%; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 4px 8px; }
th { background: #e8e8e8; }
img:not(.math-inline):not(.math-display) { max-width: 100%; display: block; margin: 0.6em auto; }
img.math-display { display: block; margin: 0.8em auto; max-height: 3.5em; }
img.math-inline { display: inline; vertical-align: -0.25em; height: 1.15em; }
hr { border: none; border-top: 1px solid #aaa; margin: 1em 0; }
blockquote { border-left: 3px solid #888; padding: 0 10px; color: #444; }

/* MathJax rendered equations */
mjx-container { font-size: 1.0em !important; }
"""


def _sanitize_latex_for_mathtext(latex: str) -> str:
    """把 matplotlib mathtext 不支持的命令改写为支持的形式。"""
    s = latex
    s = s.replace(r"\displaystyle", "")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\,", " ")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\Longrightarrow", r"\Rightarrow")
    return s


def render_math_with_matplotlib(latex: str, out_path: Path,
                                 display: bool = False) -> bool:
    """用 matplotlib 的 mathtext 把单个 LaTeX 公式渲染为 PNG。

    返回是否成功渲染；失败时调用方应用文本回退方案。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cleaned = _sanitize_latex_for_mathtext(latex)
    fontsize = 18 if display else 12
    try:
        fig = plt.figure()
        t = r"$" + cleaned + r"$"
        fig.text(0.5, 0.5, t, fontsize=fontsize, ha="center", va="center")
        fig.canvas.draw()
        bbox = fig.texts[0].get_window_extent()
        fig.set_size_inches((bbox.width + 6) / 72, (bbox.height + 6) / 72)
        fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return True
    except Exception as e:
        try:
            plt.close(fig)
        except Exception:
            pass
        print(f"  [math render fail] {cleaned!r}: {e}")
        return False


def preprocess_math(md_text: str, img_dir: Path) -> str:
    """把 Markdown 中的 $...$ 和 $$...$$ 换成 <img> 标签，图片由 matplotlib 渲染。"""
    import hashlib, re
    img_dir.mkdir(exist_ok=True)

    def _render(latex: str, display: bool) -> str:
        key = hashlib.md5(f"{display}:{latex}".encode()).hexdigest()[:12]
        path = img_dir / f"math_{key}.png"
        ok = path.exists() or render_math_with_matplotlib(latex, path, display=display)
        if not ok:
            # 渲染失败回退：显示为 inline code 的 LaTeX 源码
            safe = latex.replace("<", "&lt;").replace(">", "&gt;")
            tag = "pre" if display else "code"
            return f'<{tag} class="math-fallback">{safe}</{tag}>'
        css_class = "math-display" if display else "math-inline"
        return f'<img class="{css_class}" src="math_imgs/{path.name}" alt="{latex}" />'

    # 先换显示公式 $$...$$
    def disp_sub(m: re.Match) -> str:
        return _render(m.group(1).strip(), True)
    md_text = re.sub(r"\$\$(.+?)\$\$", disp_sub, md_text, flags=re.DOTALL)

    # 再换行内 $...$（避免吞掉成对 $），允许中文环境；跳过被转义的 \$
    def inline_sub(m: re.Match) -> str:
        return _render(m.group(1).strip(), False)
    md_text = re.sub(r"(?<!\\)\$([^$\n]+?)\$", inline_sub, md_text)

    return md_text


def build() -> None:
    # 1) 读 markdown，预处理数学为 <img>
    src = MD.read_text()
    img_dir = REPORT_DIR / "math_imgs"
    src_processed = preprocess_math(src, img_dir)

    # 2) pandoc HTML
    html_raw = subprocess.check_output(
        [
            "pandoc",
            "--from=markdown-tex_math_dollars-tex_math_single_backslash",
            "--to=html5",
            "--standalone",
            "--metadata=title:报告",
        ],
        input=src_processed,
        text=True,
    )

    styled = html_raw.replace("</head>", f"<style>{CSS}</style></head>", 1)
    HTML.write_text(styled)

    from weasyprint import HTML as WP_HTML
    WP_HTML(string=styled, base_url=str(REPORT_DIR)).write_pdf(str(PDF))
    print(f"PDF written: {PDF}")


if __name__ == "__main__":
    build()
