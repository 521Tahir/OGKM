"""Centralized logging configuration using loguru."""
import sys
import os
from pathlib import Path

try:
    from loguru import logger

    def get_logger(name: str = "tourism_ontology"):
        """Return a configured loguru logger with the given name context."""
        return logger.bind(module=name)

    def setup_logging(
        log_level: str = "INFO",
        log_file: str | None = None,
        rotation: str = "50 MB",
        retention: str = "14 days",
    ) -> None:
        """Configure global logging handlers."""
        logger.remove()
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[module]}</cyan> | {message}",
        )
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_file,
                level=log_level,
                rotation=rotation,
                retention=retention,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[module]} | {message}",
            )

except ImportError:
    import logging

    def get_logger(name: str = "tourism_ontology"):  # type: ignore[misc]
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        return logging.getLogger(name)

    def setup_logging(log_level: str = "INFO", log_file: str | None = None, **kwargs) -> None:  # type: ignore[misc]
        level = getattr(logging, log_level.upper(), logging.INFO)
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(level=level, handlers=handlers)
