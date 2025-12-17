import logging

def raise_log(ex_type: Exception, error_msg: str, logger: logging.Logger) -> None:
    logger.error(error_msg)
    raise ex_type(error_msg)