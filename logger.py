import logging
import colorlog


def configlogger(
    nome: str = __name__, log_file: str = "app.log", nivel=logging.DEBUG
) -> logging.Logger:
    logger = logging.getLogger(nome)
    logger.setLevel(nivel)

    if logger.hasHandlers():
        return logger

    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(nivel)

    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
