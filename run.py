import os

from dotenv import load_dotenv
load_dotenv()

import httpx
import logfire

from pathlib import Path
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

from pydantic_ai.models.openai import OpenAIModel

from db.func import DBFunctions
from agents.manager import AgentManager
from proxy.openai import handler

logfire.configure(send_to_logfire='never',scrubbing=False)
logfire.instrument_asyncpg()

openai_api_key = os.getenv("API_KEY_OPENAI")
openai_api_url = os.getenv("API_URL_OPENAI")
# The small model is responsible for tool calling, rag tasks and providing context for the big model
small_model = OpenAIModel(
    model_name=os.getenv("SMALL_MODEL"),
    base_url=openai_api_url,
    api_key=openai_api_key,
    )
# big modesl is responsible for main conversation        
big_model = OpenAIModel(
    os.getenv('BIG_MODEL'),
    base_url=openai_api_url,
    api_key=openai_api_key,
    )


app = FastAPI()
client = httpx.AsyncClient()
db = None
agent_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logfire.info("Starting application...")
    global db, agent_manager
    db = DBFunctions()
    await db.connect()
    await db.create_schema()
    agent_manager = AgentManager(db, small_model, big_model)
    
    yield
    
    logfire.info("Shutting down application...")
    await db.close()

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def router(request: Request, call_next):
    return await handler(
        request,
        call_next,
        openai_api_key,
        openai_api_url,
        agent_manager,
        {big_model.model_name},
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'run:app', reload=True, host="0.0.0.0", port=11433, reload_dirs=[str(Path(__file__).parent)]
    )