import json


class ConsoleReport(str):
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


def render_console(result):
    lines = ["=" * 72, "MODELBRIEF REPORT", "=" * 72]
    for s in result.sections:
        lines += ["", s.title.upper(), "-" * len(s.title), json.dumps(s.content, indent=2, default=str)]
        if s.narrative:
            lines.append(s.narrative)
    if result.warnings:
        lines += ["", "WARNINGS", *["- " + x for x in result.warnings]]
    if result.recommendations:
        lines += ["", "RECOMMENDATIONS", *["- " + x for x in result.recommendations]]
    return ConsoleReport("\n".join(lines))

