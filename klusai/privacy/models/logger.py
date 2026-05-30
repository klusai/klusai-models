"""Shared logging setup (mirrors the convention across KlusAI repos)."""

import logging


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return logging.getLogger(name)
