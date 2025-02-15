import logfire
import asyncio
import json
import uuid
from datetime import datetime
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import ChatCompletionMessageToolCallParam
from typing import List, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from itertools import chain
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.models.openai import OpenAIModel
import httpx
from agents.manager import AgentManager
from pydantic import BaseModel
from time import time
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
)


def to_openai_messages(messages: List[ModelMessage]) -> List[ChatCompletionMessageParam]:
    return list(chain(*(OpenAIModel._map_message(m) for m in messages)))


def to_pydantic_message(message_param: ChatCompletionMessageParam) -> List[ModelMessage]:
    """
    Maps an `openai.types.ChatCompletionMessageParam` back to `pydantic_ai.ModelMessage`.
    """
    role = message_param.get("role")
    content = message_param.get("content")
    tool_calls = message_param.get("tool_calls", [])
    
    if role == "user":
        return ModelRequest(parts=[UserPromptPart(content=content)]) if content else None

    elif role == "assistant":
        parts = []
        if content:
            parts.append(TextPart(content=content))

        for tool_call in tool_calls:
            parts.append(_to_pydantic_tool_msg(tool_call))

        return ModelResponse(parts=parts) if parts else None

    else:
        raise ValueError(f"Unsupported role: {role}")

def _to_pydantic_tool_msg(tool_call_param: ChatCompletionMessageToolCallParam) -> ToolCallPart:
    """
    Maps an OpenAI tool call parameter back to `pydantic_ai.ToolCallPart`.
    """
    # Assuming ToolCallPart has attributes matching the keys of the OpenAI tool call format.
    return ToolCallPart(
        tool_name=tool_call_param.get("tool_name"),
        args=tool_call_param.get("arguments"),
        tool_call_id=tool_call_param.get("id"),
    )

def to_pydantic_messages(messages: List[ChatCompletionMessageParam]) -> List[ModelMessage]:
    return [to_pydantic_message(msg) for msg in messages]

async def handler(
    request: Request,
    call_next,
    openai_api_key: str,
    openai_api_url: str,
    agent_manager: AgentManager,
    supported_models: set[str]
):
    """Middleware to proxy requests to OpenAI API or handle them manually based on the model."""
    path = request.url.path
    if path.endswith("/chat/completions"):
        try:
            body = await request.body()

            if not body:
                raise HTTPException(status_code=400, detail="Request body missing")
            chat_request = json.loads(body)
            
            if chat_request.get("model") in supported_models:
                return await to_agent_inference(chat_request, agent_manager)

        except HTTPException as he:
            logfire.exception(f'HTTPException: {he}')
            return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
        except Exception as e:
            logfire.exception(f'Exception: {e}')
            return JSONResponse(
                content={"error": f"Failed to process request: {str(e)}"}, 
                status_code=500
            )
    
    return await to_openai(request, openai_api_url, openai_api_key)


async def to_agent_inference(chat_request: dict, agent_manager: AgentManager) -> StreamingResponse:
    """Handle inference using custom models and stream results in OpenAI format as SSE."""
    
    async def stream_generator():
        try:
            model = chat_request.get("model")
            messages = to_pydantic_messages(chat_request.get("messages"))
            context_manager = await agent_manager.run(model, messages)
            id = uuid.uuid4()
            async with context_manager as result:
                async for chunk in result.stream_text(debounce_by=0.01, delta=True):
                    openai_response = {
                        "id": f"chatcmpl-{id}",
                        "object": "chat.completion.chunk",
                        "created": int(time()),
                        "model": model,
                        "choices": [{
                            "delta": {"content": chunk},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(openai_response)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            logfire.exception(f'Exception: {e}')
            error_response = {"error": str(e)}
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

async def to_openai(request: Request, api_url: str, api_key: str) -> StreamingResponse:
    """Proxy forward to OpenAI API."""
    headers = {**request.headers, "authorization": f"Bearer {api_key}"}
    request_content = await request.body()
    request_content_str = request_content.decode("utf-8")
    
    if request_content_str and json.loads(request_content_str).get("stream"):
        upstream_media_type = None

        async def stream_generator():
            nonlocal upstream_media_type
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    method=request.method,
                    url=api_url + str(request.url.path),
                    headers=headers,
                    content=request_content,
                ) as upstream_response:
                    upstream_media_type = upstream_response.headers.get(
                        "content-type", "application/octet-stream"
                    )
                    async for chunk in upstream_response.aiter_bytes():
                        yield chunk

        # prime stream_generator to capture the media type
        gen = stream_generator().__aiter__()
        try:
            first_chunk = await gen.__anext__()
        except StopAsyncIteration:
            first_chunk = b""
        
        async def final_generator():
            yield first_chunk
            async for chunk in gen:
                yield chunk

        return StreamingResponse(
            content=final_generator(),
            media_type=upstream_media_type or "application/octet-stream",
        )
    else:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=api_url + str(request.url.path),
                headers=headers,
                content=request_content,
            )
            return StreamingResponse(
                content=response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers),
            )