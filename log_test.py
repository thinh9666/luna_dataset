import logging
logging.basicConfig(level = logging.DEBUG, filename ="log.log", filemode="w",
                    format = "%(asctime)s - %(levelname)s - %(message)s") # mặc định level warning
x=2
try:
    1/0
except ZeroDivisionError as e:
    logging.error("ZeroDivisionError",exc_info=False)
logger= logging.getLogger(__name__)
#__name__ trả về name của module đó
#nếu có 1 logger đã exist thì lấy cái đó, ko thì tạo cái mới

handler= logging.FileHandler("test.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("test the custom logger")
#ghi vào cả test.log và log.log vì mặc đinh propagate=True

logging.debug(f"the value of {x}")
logging.debug("debug")
logging.info("info")
logging.warning("warning")
logging.error("error")
logging.critical("critical")
print(__name__)