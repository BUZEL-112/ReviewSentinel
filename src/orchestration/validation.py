import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass
from typing import List, Optional
from src.utils.logger import logger
from src.utils.exception import CustomException

class ReviewRecord(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    title: Optional[str] = None
    text: Optional[str] = None

@dataclass
class ValidationResult:
    success: bool
    failed_expectations: List[str]
    row_count: int
    class_distribution: dict
    summary: str

class DataValidator:
    def __init__(self, validation_config: dict = None):
        self.config = validation_config or {}
        self.max_null_ratio = self.config.get("max_null_ratio", 0.10)
        self.max_class_imbalance = self.config.get("max_class_imbalance", 0.80)
        self.min_row_count = self.config.get("min_row_count", 100)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        try:
            logger.info("Starting Pydantic data validation...")
            
            failed_expectations = []
            
            # Schema expectations
            if list(df.columns) != ["rating", "title", "text"]:
                failed_expectations.append("Columns mismatch: expected ['rating', 'title', 'text']")

            if "rating" in df.columns and df["rating"].isnull().any():
                failed_expectations.append("Rating column contains nulls")

            # Allow some missing text
            if "text" in df.columns:
                text_null_ratio = df["text"].isnull().mean()
                if text_null_ratio > self.max_null_ratio:
                    failed_expectations.append(f"Text column exceeds {self.max_null_ratio*100:.1f}% nulls")

            # Volume expectation
            row_count = len(df)
            if row_count < self.min_row_count:
                failed_expectations.append(f"Row count below minimum of {self.min_row_count}")

            # Class distribution check
            class_dist = {}
            if "rating" in df.columns and row_count > 0:
                class_dist = df["rating"].value_counts(normalize=True).to_dict()
                for rating, fraction in class_dist.items():
                    if fraction > self.max_class_imbalance:
                        failed_expectations.append(
                            f"Class imbalance: rating {rating} makes up {fraction:.1%} "
                            f"(max allowed: {self.max_class_imbalance:.1%})"
                        )

            # Pydantic row-level validation
            if list(df.columns) == ["rating", "title", "text"]:
                records = df.where(pd.notnull(df), None).to_dict(orient="records")
                invalid_rows = 0
                for record in records:
                    try:
                        ReviewRecord(**record)
                    except ValidationError:
                        invalid_rows += 1
                
                if invalid_rows > 0:
                    failed_expectations.append(f"{invalid_rows} rows failed Pydantic schema validation (e.g., rating outside [1, 5] or wrong types)")

            success = len(failed_expectations) == 0
            
            summary = "Validation passed." if success else f"Validation failed: {'; '.join(failed_expectations)}"
            if not success:
                summary += f" | Class distribution: {class_dist}"
                
            logger.info(summary)

            return ValidationResult(
                success=success,
                failed_expectations=failed_expectations,
                row_count=row_count,
                class_distribution=class_dist,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            raise CustomException(e)
