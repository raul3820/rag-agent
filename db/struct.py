from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from enum import Enum

programming_languages = {
        "python",
        "java",
        "c",
        "c++",
        "javascript",
        "js",
        "ruby",
        "go",
        "rust",
        "kotlin",
        "swift",
        "php",
        "typescript",
        "plpgsql",
        "sql",
        "html",
        "css",
        "r",
        "rsharp",
        "r#",
        "scala",
        "perl",
        "haskell",
        "dart",
        "lua",
        "elixir",
        "fsharp",
        "f#",
        'csharp',
        "c#",
        "shell",
        "bash",
        "matlab",
        "vhdl",
        "verilog",
        "dreamberd",
    }

class Abstract(BaseModel):
    title: str = ""
    summary: str = ""

class ProcessedMetadata(Abstract):
    programming_languages: List[str] = []

class ProcessedChunk(BaseModel):
    source: str
    chunk_number: int
    metadata: dict
    processed_metadata: ProcessedMetadata
    embedding: List[float]
    content: str
