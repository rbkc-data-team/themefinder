import logging
import sys

logger = logging.getLogger("theme_finder.tasks")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
