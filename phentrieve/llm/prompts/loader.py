from importlib import resources


def load_prompt_template(template_name: str) -> str:
    return (
        resources.files(__package__)
        .joinpath("templates", template_name)
        .read_text(encoding="utf-8")
    )
