import logging
import os
import re

class SensitiveFormatter(logging.Formatter):
    """
    Masks secret env var values in all log output [T-1].
    WHY: A bare 'except Exception as exc: logger.error(exc)' call on a PRADAN
    HTTP error can include the request URL with embedded credentials in the
    traceback. This formatter is the last-resort safety net.
    """
    _PATTERN: re.Pattern | None = None

    @classmethod
    def _get_pattern(cls) -> re.Pattern | None:
        secrets = [
            os.environ.get("PRADAN_PASSWORD", ""),
            os.environ.get("PRADAN_USERNAME", ""),
            os.environ.get("GH_PAT", ""),
        ]
        active = [re.escape(s) for s in secrets if len(s) > 3]
        if not active:
            return None
        return re.compile("|".join(active))

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._PATTERN is None:
            self.__class__._PATTERN = self._get_pattern()
        if self._PATTERN:
            msg = self._PATTERN.sub("***REDACTED***", msg)
        return msg

def configure_logging(level: int = logging.INFO) -> None:
    """
    Call once at application startup (main.py, scheduler.py).
    Installs SensitiveFormatter on the root logger so ALL modules
    (including third-party libraries that use logging) are covered.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(SensitiveFormatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"))
    logging.root.setLevel(level)
    logging.root.handlers = [handler]
