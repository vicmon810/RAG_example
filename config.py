from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os


class Settings(BaseSettings):
    DOCS_DIR: Path = Path("note")
    INDEX_DIR: Path = Path("index")
    TOP_K: int = 3

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "deepseek-r1-1.5b"
    embed_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3

    @property
    def INDEX_FILE(self) -> Path:
        return self.INDEX_DIR / "docs.index"

    @property
    def CHUNKS_FILE(self) -> Path:
        return self.CHUNKS_FILE / "chunk.json"

    @property
    def META_FILE(self) -> Path:
        return self.INDEX_DIR / "meta.json"

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()
