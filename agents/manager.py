
import json
import asyncio
import random
from pprint import pprint
from contextlib import asynccontextmanager
from typing import List, AsyncGenerator
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel, _map_usage
from pydantic_ai.result import StreamedRunResult

from fastapi.encoders import jsonable_encoder
from db.func import DBFunctions
from db.struct import ProcessedMetadata
from agents.retriever import Retriever
from agents.tool_agent import StreamToolAgent, StreamResponse
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion, ParsedChatCompletionMessage

@dataclass
class Deps:
    db: DBFunctions

class AgentManager:
    def __init__(self, db: DBFunctions, small_model: OpenAIModel, big_model: OpenAIModel):
        self.end_study_mode_msg = '--- Study mode ended ---'
        self.summarizer = Agent(
            small_model,
            retries=3,
            system_prompt="You are an AI that extracts information from documentation chunks.",
        )
        self.crawler = Retriever(self.summarizer, db, crawl_concurrent=3)
        
        self.student = StreamToolAgent(
            small_model,
            retries=3,
            system_prompt=(
                f"You are a strict AI assistant:\n"
                f"1. Show related URLs.\n"
                f"2. Ask user for URL selection.\n"
                f"3. Use tool '{self.crawler.study.__name__}'."
                # f"3. Use tool, the name of the tool is '{self.crawler.study.__name__}', put a list of the selected URL(s) as argument."
                ),
            tools=[self.crawler.study],
        )
        self.big_agent = Agent(
            big_model,
            deps_type=Deps,
            retries=3,
            system_prompt="You are an AI assistant. If user provides <context> and it is related, use it to help the user.",
            )
        
    async def run(self, model: str, messages: List[ModelMessage]):
        assert messages[-1].parts[-1].part_kind == UserPromptPart.part_kind
        user_prompt = messages.pop().parts[-1].content
        
        study_mode_msgs = 2
        if '#study' in user_prompt[:64].lower():
            urls = await self.crawler.get_related_urls(user_prompt)
            if len(urls):
                study_max_urls = 30
                related_urls = json.dumps({"related_urls": urls[:study_max_urls]})
                user_prompt = (
                    f'<context>{related_urls}</context>\n'
                    'Show me the list of related urls. Ask me if I want you to study one or more of them.'
                )
                return self.student.run_tool_stream(
                    deps=None,
                    user_prompt=user_prompt,
                    model_settings={'temperature': 0.1, 'tool_choice': 'none'},
                )
        
        elif any(['#study' in p.content[:64].lower() for m in messages[-study_mode_msgs:] for p in m.parts]):
            r = self.student.run_tool_stream(
                deps=self,
                user_prompt=user_prompt,
                message_history=messages[-study_mode_msgs:],
                model_settings={'temperature': 0.1, 'end_study_mode_msg': self.end_study_mode_msg},
            )
        
        else:
            from_index = self.arg_max_where(messages, self.study_mode_condition)
            messages = messages[from_index+1:]
            query = '\n\n'.join([p.content for m in messages for p in m.parts]) + f"\n\n{user_prompt}"
            ctx, has_meta = await self.crawler.get_db_content(query)
            user_prompt = f'<context>{json.dumps(ctx)}</context>\n{user_prompt}' if has_meta else user_prompt
            
            r = self.big_agent.run_stream(
                user_prompt=user_prompt,
                message_history=messages,
                model_settings={'temperature': 0.5},
            )

        return r

    def arg_max_where(self, iterable, condition, loop_max: int = 16):
        len_it = len(iterable)
        for i in range(len_it - 1, -1, -1):
            if condition(iterable[i]):
                return i
            
            if i < len_it - loop_max:
                return 0
        return 0
    
    def study_mode_condition(self, message):
        for part in message.parts:
            return self.end_study_mode_msg in part.content[-64:]

@asynccontextmanager
async def send_msg(msg: str) -> AsyncGenerator[StreamResponse, None]:
    async def msg_generator():
        yield msg
    yield StreamResponse(msg_generator())


