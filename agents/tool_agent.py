import os
import asyncio
import random
import json
from typing import AsyncIterator, AsyncGenerator, List, Any, Union, Sequence
from contextlib import asynccontextmanager
from itertools import chain
from openai import NOT_GIVEN
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    DocstringFormat,
    RunContext,
    Tool,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserPromptPart,
    RetryPromptPart,
)


from dotenv import load_dotenv
load_dotenv()
import logfire
logfire.configure(send_to_logfire='never', scrubbing=False)


class StreamResponse:
    def __init__(self, generator: AsyncIterator[str]):
        self._generator = generator

    async def stream_text(self, **kwargs) -> AsyncIterator[str]:
        async for chunk in self._generator:
            yield chunk

    def __aiter__(self):
        return self._generator.__aiter__()

class StreamToolAgent(Agent):
    """
    Pydantic-ai Agent that streams tool responses, only compatible with OpenAIModel.
    """
    def __init__(self,
        model: OpenAIModel,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        **kwargs
    ) -> None:
        assert len(tools) == 1, f"{self.__class__.__name__} accepts a list of only 1 tool, due to small model reliability"
        self.model: OpenAIModel = model
        self.tools = tools
        super().__init__(model, tools=tools, **kwargs)
    
    @asynccontextmanager
    async def run_tool_stream(
        self,
        deps: Any,
        user_prompt: str = None,
        message_history: List[ModelMessage] = None,
        model_settings: OpenAIModelSettings = None,
        sniff_chunks: int = 5
    ) -> AsyncGenerator[StreamResponse, None]:
        """
        Returns an async context manager that yields an async iterator of text chunks.
        If a tool is sniffed in the first n chunks, that tool is invoked and tool 
        outputs are streamed; otherwise, the model's streaming output is returned.
        """
        model_settings = model_settings or {}
        assert user_prompt or message_history
        if user_prompt:
            model_request = ModelRequest([UserPromptPart(content=user_prompt)])
            message_history = message_history or []
            message_history.append(model_request)
        async def generator() -> AsyncIterator[str]:
            retrial = -1
            while retrial < self._default_retries:
                tools = [self.model._map_tool_definition(await tool.prepare_tool_def(None)) for _,tool in self._function_tools.items()]
                openai_messages = list(chain(*(self.model._map_message(m) for m in message_history)))
                response_text = ""
                i = 0
                final_tool_calls: dict[int, ChoiceDeltaToolCall] = {}
                stream = await self.model.client.chat.completions.create(
                    model=self.model.model_name,
                    messages=openai_messages,
                    n=1,
                    parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                    tools=(None if model_settings.get('tool_choice') == 'none' else tools) or NOT_GIVEN,
                    tool_choice=model_settings.get('tool_choice', NOT_GIVEN),
                    stream=True,
                    stream_options={'include_usage': True},
                    max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                    temperature=model_settings.get('temperature', NOT_GIVEN),
                    top_p=model_settings.get('top_p', NOT_GIVEN),
                    timeout=model_settings.get('timeout', NOT_GIVEN),
                    seed=model_settings.get('seed', NOT_GIVEN),
                    presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                    frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                    logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                    reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
                )
                
                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta: ChoiceDelta = chunk.choices[0].delta
                    
                    for tool_call in delta.tool_calls or []:
                        index = tool_call.index
                        if index not in final_tool_calls:
                            final_tool_calls[index] = tool_call

                    if i < sniff_chunks or final_tool_calls:
                        response_text += delta.content
                    else:
                        if response_text:
                            yield response_text
                            response_text = None
                        yield delta.content
                    
                    i += 1
                
                if final_tool_calls:
                    for i, tool_call in final_tool_calls.items():
                        try:
                            args = json.loads(tool_call.function.arguments)
                            tool = list(self._function_tools.items())[0][1]
                            async for tool_result in (tool.function(deps, **args) if tool.takes_ctx else tool.function(**args)):
                                assert type(tool_result) == str
                                yield tool_result
                        except Exception as e:
                            logfire.info(f"tool call failed: {tool_call} -- {e}")
                            yield f"Please try again. Tool call failed: {tool_call} -- {e}"
                            # if isinstance(message_history[-1].parts[-1], RetryPromptPart):
                            #     message_history[-1].parts.pop()
                            # message_history[-1].parts.append(RetryPromptPart(
                            #     content=[f'{e}'],
                            #     tool_name=tool_call.function.name,
                            #     tool_call_id=tool_call.id
                            # ))

                if retrial < 0:
                    break
            end_study_mode_msg = model_settings.get('end_study_mode_msg')
            if end_study_mode_msg:
                yield f"\n\n{end_study_mode_msg}\n\n"
        try:
            yield StreamResponse(generator())
        finally:
            # cleanup?
            pass


openai_api_key = os.getenv("API_KEY_OPENAI")
openai_api_url = os.getenv("API_URL_OPENAI")
small_model = OpenAIModel(
    model_name=os.getenv("SMALL_MODEL"),
    base_url=openai_api_url,
    api_key=openai_api_key,
)

async def roll_dice(ctx: RunContext[str], n_times: Union[str, int]) -> AsyncGenerator[str, None]:
    """This is an asynchronous dice rolling function."""
    i = 0
    yield f"Hello {ctx}, rolling {n_times} times!"
    while i < int(n_times):
        result = random.randint(1, 6)
        yield str(result)
        await asyncio.sleep(1)
        i += 1

async def main():
    deps = "Slim Shady"
    agent = StreamToolAgent(
        small_model, 
        tools=[roll_dice],
        )

    prompt = "Please use the roll dice tool to simulate a couple (between 1 and 10) dice rolls."
    try:
        context_manager = agent.run_tool_stream(deps, prompt)
        async with context_manager as result:
            print('result type is: ', type(result))
            async for text_chunk in result.stream_text(debounce_by=0.1, delta=True):
                print(f'[Agent Stream] {text_chunk}')
    except Exception as e:
        logfire.exception(f"Stream error: {e}")
    finally:
        logfire.exception("Stream completed")

# if __name__ == '__main__':
#     asyncio.run(main())


