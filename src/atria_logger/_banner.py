"""ASCII art banner for the Atria library."""

import pyfiglet
from rich.console import Console

_BANNER_FONT = "ansi_shadow"


def print_banner(
    subtitle: str | None = None, style: str = "bold cyan", font: str = _BANNER_FONT
) -> None:
    """Print a stylized ATRIA ASCII art banner to the console.

    Args:
        subtitle: Optional line printed centered beneath the banner (e.g. a version string).
        style: Rich style string used to color the banner.
        font: pyfiglet font used to render the "ATRIA" text.
    """
    console = Console()
    banner = pyfiglet.figlet_format("ATRIA", font=font)
    console.print(banner, style=style, highlight=False)
    if subtitle:
        console.print(subtitle, style="dim", justify="center")
