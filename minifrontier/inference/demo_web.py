"""Self-contained browser pages, served without external assets or a build step."""

from importlib.resources import files


def page(name: str) -> str:
    if name not in {"checkpoint", "mf1"}:
        raise ValueError("unknown demo page")
    assets = files("minifrontier.inference").joinpath("web")
    return (
        assets.joinpath(f"{name}.html")
        .read_text(encoding="utf-8")
        .replace("<!--STYLE-->", assets.joinpath("demo.css").read_text(encoding="utf-8"))
        .replace("<!--SCRIPT-->", assets.joinpath(f"{name}.js").read_text(encoding="utf-8"))
    )
