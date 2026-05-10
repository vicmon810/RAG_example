from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DOCS_DIR: Path = Path("note")
    INDEX_DIR: Path = Path("index")
    TOP_K: int = 3

    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str =  "qwen2.5-coder:0.5b"
    #"deepseek-r1:1.5b"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    @property
    def INDEX_FILE(self) -> Path:
        return self.INDEX_DIR / "docs.index"

    @property
    def CHUNK_FILE(self) -> Path:
        return self.INDEX_DIR / "chunks.json"

    @property
    def META_FILE(self) -> Path:
        return self.INDEX_DIR / "meta.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True
    )


settings = Settings()