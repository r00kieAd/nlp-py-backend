import os
import logging
from app.main import app

logging.basicConfig(level=logging.INFO)
env = os.getenv("FLASK_ENV", "dev")

if env == "prod":
    logging.info("Running in production...")
else:
    logging.info("Running in development...")

if __name__ == "__main__":
    app.run()