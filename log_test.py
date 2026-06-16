import logging
logging.basicConfig(level = logging.DEBUG, filename ="log.log", filemode="w",
                    format = "%(asctime)s - %(levelname)s - %(message)s") # mặc định level warning

logging.debug("debug")
logging.info("info")
logging.warning("warning")
logging.error("error")
logging.critical("critical")