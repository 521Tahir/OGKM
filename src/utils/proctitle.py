"""Process-title helper — sets a consistent proctitle for every run of this project."""
from __future__ import annotations

PROJECT_PROCTITLE = "tahir_Ont-code"


def set_proctitle(name: str = PROJECT_PROCTITLE) -> None:
    """Set the OS process title so all project jobs are identifiable (e.g. in `ps`/`top`)."""
    try:
        import setproctitle
        setproctitle.setproctitle(name)
    except Exception:
        pass
