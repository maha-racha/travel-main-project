from pydantic import BaseModel, Field


class UpdateLanguageRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    language: str = Field(..., description="Language code (e.g., 'en', 'ro', 'ru')")


class UpdateLanguageResponse(BaseModel):
    success: bool
    message: str
    language: str
