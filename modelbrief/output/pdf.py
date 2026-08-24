from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib import colors
from reportlab.lib.units import inch
import json
import base64
import io


def render_pdf(result, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

    story = [
        Paragraph("ModelBrief Report", styles["Title"]),
        Paragraph(
            "Analysis of a supplied, already-trained model. "
            "ModelBrief did not retrain it.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    # Main report sections
    for s in result.sections:
        story += [
            Paragraph(s.title, styles["Heading2"])
        ]

        data = [
            [
                Paragraph("Field", styles["BodyText"]),
                Paragraph("Value", styles["BodyText"]),
            ]
        ]

        for k, v in s.content.items():
            value = json.dumps(v, indent=2, default=str)[:5000]

            data.append([
                Paragraph(str(k), styles["BodyText"]),
                Paragraph(
                    value.replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"),
                    styles["BodyText"],
                ),
            ])

        table = Table(
            data,
            colWidths=[1.7 * inch, 5.1 * inch],
            repeatRows=1,
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#185abd"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ])
        )

        story += [
            table,
            Spacer(1, 14),
        ]

        if s.narrative:
            story += [
                Paragraph(s.narrative, styles["BodyText"]),
                Spacer(1, 10),
            ]

    # AI recommendations
    if getattr(result, "recommendations", None):
        story += [
            Paragraph("RECOMMENDATIONS", styles["Heading2"]),
            Spacer(1, 6),
        ]

        for recommendation in result.recommendations:
            story += [
                Paragraph(
                    f"• {str(recommendation)}",
                    styles["BodyText"],
                ),
                Spacer(1, 6),
            ]

    # Figures
    for f in result.figures:
        try:
            story += [
                Paragraph(
                    f.get("title", "Figure"),
                    styles["Heading3"],
                ),
                Image(
                    io.BytesIO(
                        base64.b64decode(f["data"])
                    ),
                    width=6.4 * inch,
                    height=4.2 * inch,
                ),
                Spacer(1, 10),
            ]
        except Exception:
            pass

    SimpleDocTemplate(
        str(p),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    ).build(story)

    return p