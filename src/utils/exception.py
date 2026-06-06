import sys
import traceback
from src.utils.logger import logger

def error_message_details(error: Exception) -> str:
    """Extract detailed error information including file name and line number."""
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error in file [{file_name}] at line [{line_no}]: {type(error).__name__} - {error}"
    return f"{type(error).__name__}: {error}"

class CustomException(Exception):
    """Custom exception class that captures detailed error information."""
    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error_message = error_message_details(error)
        logger.error(self.error_message)
        logger.error("Full traceback:\n%s", traceback.format_exc())


    def __str__(self):
        return self.error_message
