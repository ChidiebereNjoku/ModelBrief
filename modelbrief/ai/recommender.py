import json


def recommend(result, provider):
    prompt = (
    "Provide 3 to 5 concise, practical, evidence-based recommendations. "
    "Use clean Markdown with one recommendation per line beginning with '- '. "
    "Do not use numbered lists, nested bullets, bullet symbols such as '•', "
    "or additional '-' characters. "
    "Focus only on recommendations supported by the supplied results. "
    "Do not repeat metrics or findings already shown in the report. "
    "Do not claim the model was retrained or improved. "
    "Do not provide code or tables. "
    "Prioritize the most useful recommendations and avoid repetition."
)

    data = json.dumps(
        result.to_dict(),
        default=str,
    )[:7000]

    return provider.complete(prompt, data)

