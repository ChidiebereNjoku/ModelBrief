from pathlib import Path
import html
import json


def render_html(result, path):
    sections = []

    for s in result.sections:
        rows = ''.join(
            f'<tr><th>{html.escape(str(k))}</th>'
            f'<td><pre>{html.escape(json.dumps(v, indent=2, default=str))}</pre></td></tr>'
            for k, v in s.content.items()
        )

        sections.append(
            f'<section>'
            f'<h2>{html.escape(s.title)}</h2>'
            f'<table>{rows}</table>'
            f'{f"<p>{html.escape(s.narrative)}</p>" if s.narrative else ""}'
            f'</section>'
        )

    # Add AI recommendations
    if getattr(result, "recommendations", None):
        recommendations = ''.join(
            f'<li>{html.escape(str(rec))}</li>'
            for rec in result.recommendations
        )

        sections.append(
            f'<section>'
            f'<h2>RECOMMENDATIONS</h2>'
            f'<ul>{recommendations}</ul>'
            f'</section>'
        )

    figs = ''.join(
        f'<figure>'
        f'<img src="data:image/png;base64,{f["data"]}"/>'
        f'<figcaption>{html.escape(f.get("title", ""))}</figcaption>'
        f'</figure>'
        for f in result.figures
    )

    doc = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>ModelBrief report</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1100px;
                margin: 40px auto;
                color: #172033;
            }}

            h1 {{
                color: #185abd;
            }}

            section {{
                margin: 24px 0;
                padding: 20px;
                border: 1px solid #dde3ea;
                border-radius: 12px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}

            th, td {{
                text-align: left;
                vertical-align: top;
                border-bottom: 1px solid #eee;
                padding: 8px;
            }}

            th {{
                width: 28%;
            }}

            pre {{
                white-space: pre-wrap;
                margin: 0;
            }}

            li {{
                margin-bottom: 10px;
            }}

            img {{
                max-width: 100%;
            }}

            .note {{
                background: #eef5ff;
                padding: 12px;
            }}
        </style>
    </head>

    <body>
        <h1>ModelBrief Report</h1>

        <p class="note">
            Analysis of a supplied, already-trained model.
            ModelBrief did not retrain it.
        </p>

        {''.join(sections)}

        {figs}

    </body>
    </html>
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")

    return p