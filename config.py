from pydantic_settings import BaseSettings


class Setting(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "deepseek-r1-1.5b"
    embed_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
