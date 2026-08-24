import json


def explain(result, provider):
    prompt = (
    "Explain the ML evaluation results accurately and concisely. "
    "Return exactly 3 to 5 findings. "
    "Use clean Markdown with one finding per line beginning with '- '. "
    "Do not use numbered lists, nested bullets, bullet symbols such as '•', "
    "or additional '-' characters. "
    "Focus only on the most important findings, patterns, and limitations. "
    "Do not repeat all metrics, tables, confusion matrices, or feature "
    "importance values already shown in the report. "
    "Do not provide code. "
    "Base every statement only on the supplied results. "
    "Do not invent facts or make unsupported claims."
)

    data = json.dumps(
        result.to_dict(),
        default=str,
    )[:7000]

    return provider.complete(prompt, data)
