
# inspired in work from https://github.com/coleam00

import os
import logging
import re
import asyncio
import logfire

from urllib.parse import urlsplit
from collections import namedtuple
from typing import List, Dict, AsyncGenerator
from dotenv import load_dotenv
from db.func import DBFunctions
from db.struct import Abstract, programming_languages, ProcessedMetadata, ProcessedChunk
from dataclasses import dataclass
from infinity_client.client import Client as EmbeddingsClient
from infinity_client.models import OpenAIEmbeddingInputText, OpenAIEmbeddingResult
from infinity_client.api.default import embeddings
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from pydantic_ai import Agent
from pydantic_ai.usage import Usage

load_dotenv()
@dataclass
class CleanUrl:
    clean: str
    base: str

# This agent is responsible for extracting metadata for embeddings
# Divided into two agents so they are only called once per task, otherwise agent has to decide to use a tool
class Retriever:
    def __init__(self, agent: Agent, db: DBFunctions, crawl_concurrent=3):
        self.embeddings_ndim = int(os.getenv("EMBEDDINGS_NDIM"))
        self.embeddings_model = os.getenv("EMBEDDINGS_MODEL")
        self.embeddings_client = EmbeddingsClient(base_url=os.getenv("EMBEDDINGS_URL"))

        self._chunk_char_size = int(os.getenv("CHUNK_SIZE")) * 3
        self.embeddings_ndim = int(os.getenv("EMBEDDINGS_NDIM"))
        self.db = db
        self.agent = agent
        
        self._browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        self._crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            check_robots_txt=True,
            semaphore_count=crawl_concurrent
            )

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from EMBEDDINGS_MODEL."""
        try:
            embeds: OpenAIEmbeddingResult = await embeddings.asyncio(client=self.embeddings_client, body=OpenAIEmbeddingInputText.from_dict({
                "input": [text],
                "model": self.embeddings_model,
            }))                

            return embeds.data[0].embedding
        except Exception as e:
            logfire.exception(f"Error getting embedding: {e}")
            return [0] * self.embeddings_ndim

    def _get_clean_url(self, url: str, keep_query: bool) -> CleanUrl:
        parsed_url = urlsplit(url.lower())
        path = parsed_url.path.rstrip('/') if parsed_url.path != '/' else parsed_url.path
        
        clean = f"{parsed_url.scheme}://{parsed_url.netloc}{path}"
        if keep_query and parsed_url.query:
            clean += f"?{parsed_url.query}"
        
        base = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return CleanUrl(clean, base)

    def _unique_urls(self, urls: List[str], must_contain: str = None, must_contain_any: List[str] = None, keep_query = False) -> List[str]:
        seen_urls = set()
        unique_ordered_urls = []
        for u in urls:
            clean_url = self._get_clean_url(u, keep_query).clean
            if clean_url not in seen_urls and (must_contain is None or must_contain in u) and (must_contain_any is None or any(s in u for s in must_contain_any)):
                seen_urls.add(clean_url)
                unique_ordered_urls.append(clean_url)
        
        return unique_ordered_urls

    def _parse_urls_from_text(self, text: str) -> List[str]:
        # Regular expression to match URLs
        url_pattern = re.compile(
            r"""
            (?:https?://)?                  # Optional http:// or https:// at the start
            (?:www\.)?                     # Optional www. prefix
            [a-zA-Z0-9\-\.]+               # Domain name, including letters, numbers, hyphens, and periods
            \.[a-zA-Z]{2,}                # Top-level domain, e.g., .com, .org
            (?:/[a-zA-Z0-9+&@#/%?=$~_.]*)*   # Optional path with allowed characters
            (?:\?[a-zA-Z0-9+&@#/%?=$~_.]*)?  # Optional query parameters
            (?:\#[a-zA-Z0-9+&@#/%?=$~_.]*)?   # Optional fragments
            """,
            re.VERBOSE
        )
        # Find all matches of the pattern in the input text
        urls = re.findall(url_pattern, text)
        return self._unique_urls(urls)

    async def get_related_urls(self, prompt: str) -> List[str]:
        """Function to get related URLs from text."""
        urls_to_crawl = []
        r = []
        
        try:
            for u in self._parse_urls_from_text(prompt):
                clean_url = self._get_clean_url(u, keep_query=True)
                urls_to_crawl.extend([clean_url.clean,clean_url.base])
                r.append(clean_url.clean)

            urls_to_crawl = self._unique_urls(urls_to_crawl, keep_query=True)
            r = self._unique_urls(r, keep_query=False)
            new_r = []

            async with AsyncWebCrawler(config=self._browser_config) as crawler:
                results = await crawler.arun_many(urls_to_crawl, config=self._crawl_config)
                for result in results:
                    if result.success:
                        logfire.info(f"Successfully crawled {result.url}")
                    elif result.status_code == 403 and "robots.txt" in result.error_message:
                        logfire.warning(f"Skipped {result.url} - blocked by robots.txt")
                        continue
                    else:
                        logfire.exception(f"Failed to crawl {result.url}: {result.error_message}")
                        continue
                    new_r.extend([l['href'] for l in result.links['internal']])
                    new_r = self._unique_urls(new_r, must_contain_any=r, keep_query=False)
            
            r.extend(new_r)
            r = self._unique_urls(r)

        except Exception as e:
            logfire.exception(f"{e}")
        
        return r

    async def _get_abstract(self, chunk: str, source: str) -> tuple[Abstract, bool]:
        """Extract title and summary using SMALL MODEL."""
        try:
            r = await self.agent.run(
                f"""
                    Return a JSON object with "title", "summary" keys.
                    For "title": Extract or derive a descriptive title.
                    For "summary": What are the main points in this chunk? Include keywords.

                    Chunk: {source} {chunk}
                """,
                result_type=Abstract,
                model_settings={'temperature': 0.25, 'max_tokens': 1024},
                
            )
            return r.data, True
        except Exception as e:
            logfire.exception(f"{e}")
        return Abstract(), False

    def _extract_languages(self, input: str, known_languages: set) -> set[str]:
        normalized_data = re.sub(r'\\n+', ' ', input.strip())
        cleaned_data = re.sub(r'[^a-zA-Z0-9,#\+]+', '', normalized_data)
        languages = [lang.strip().lower() for lang in cleaned_data.split(',') if lang.strip()]
        return set(languages).intersection(known_languages)

    async def _get_languages(self, chunk: str, source: str) -> tuple[List[str], bool]:
        retries: int = self.agent._max_result_retries
        
        for attempt in range(retries):
            try:
                r = await self.agent.run(
                    f"""
                        Return a list of programming languages used in this chunk, if any. Only the list, CSV format, no extra characters, no Markdown.
                        Chunk: {source} {chunk}
                    """,
                    model_settings={'temperature': 0.25, 'max_tokens': 1024},
                )
                r = self._extract_languages(r.data, programming_languages)
                if r:
                    return (r, True)
                chunk = chunk[:int(len(chunk)*0.5)]
            except Exception as e:
                logfire.exception(f"_get_languages attempt {attempt + 1}/{retries} failed: {e}")
                
        # If all retries fail or no valid results
        return [], False

    def _remove_markdown_links(self, text):
        # This pattern matches an optional exclamation mark, a '[',
        # then any characters until the first ']', then '(' and any characters until ')'
        pattern = r'!?\[([^\]]*)\]\([^\)]*\)'
        previous_text = None
        # Loop until applying the regex makes no further changes
        while previous_text != text:
            previous_text = text
            text = re.sub(pattern, lambda m: m.group(1).strip(), text)
        return text

    def _find_split(self, window: str, delimiters: list[str], min_window_size: int) -> int:
            
        window_size = len(window)
        # Try each delimiter in order of priority
        for delimiter in delimiters:
            idx = window.rfind(delimiter)
            if min_window_size < idx:
                window_size = idx
                break
        # If no suitable split point is found, use the original chunk size
        return window_size

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks, respecting code blocks and paragraphs."""
        chunks = []
        start = 0
        text_length = len(text)
        in_code_block = False  # Track whether currently inside a code block

        while start < text_length:
            end = start + self._chunk_char_size
            if text_length < start + self._chunk_char_size*0.5:
                chunks.append(text[start:])
                break

            current_window = text[start:end]
            tentative_end = start + self._find_split(current_window, ['#', '##', '###', '\n\n\n', '\n\n', '\n', '. ', ' '], self._chunk_char_size*0.5)
            # try splitting by code blocks
            code_split = False
            if in_code_block:
                # Look for closing ``` within the current chunk
                closing_pos = current_window.find('```', 3)
                if closing_pos != -1:
                    # Include the closing ```
                    end = start + closing_pos + 3
                    in_code_block = False
                    code_split = True
            else:
                # Look for opening ``` within the current chunk
                opening_pos = current_window.find('```')
                if opening_pos != -1:
                    # split just before code block
                    window_end = self._find_split(current_window[:opening_pos], ['#', '##', '###', '\n\n\n', '\n\n', '\n'], self._chunk_char_size*0.1)
                    # Split before the opening ```
                    if opening_pos <= window_end: 
                        end = start + opening_pos
                        in_code_block = True
                    else:
                        end = start + window_end
                    code_split = True
            
            if not code_split:
                end = tentative_end

            chunk = text[start:end]
            if chunk:
                if 0 < len(chunks) and (len(chunks[-1]) + len(chunk)) < self._chunk_char_size and ('```' in chunks[-1][-16:] or not in_code_block):
                    chunks[-1] += chunk
                else:
                    chunks.append(chunk)
            start = end

        return chunks

    async def _get_processed_meta(self, chunk: str, source: str = '') -> tuple[ProcessedMetadata, List[float], bool]:
        abstract, is_meta_success = await self._get_abstract(chunk, source)
        languages, _  = await self._get_languages(chunk, source)
        
        embedding_string = f"{source} {abstract.title} {abstract.summary} {chunk}"
        processed_metadata = ProcessedMetadata(
            title=abstract.title,
            summary=abstract.summary,
            programming_languages=languages
        )

        embedding = await self.get_embedding(embedding_string)

        return processed_metadata, embedding, is_meta_success

    async def _process_chunk(self, chunk: str, chunk_number: int, source: str) -> ProcessedChunk:
        """Process a single chunk of text."""

        logfire.info('Hello', name='world')

        try:
            processed_metadata, embedding, is_meta_success = await self._get_processed_meta(chunk, source)

            metadata = {
                "chunk_size": len(chunk),
                "model": self.agent.model.name(),
                "is_processed": is_meta_success
            }
            
            return ProcessedChunk(
                source=source,
                chunk_number=chunk_number,
                metadata=metadata,
                processed_metadata=processed_metadata,
                embedding=embedding,
                content=chunk
            )
        except Exception as e:
            logfire.exception(f"Error processing chunk {chunk_number} from {source}: {e}")
            return ProcessedChunk(
                source,
                chunk_number,
                {'Error': 'Not embedded'},
                {'Error': 'Not embedded'},
                [0] * self.embeddings_ndim,
                chunk,
            )

    async def _process_and_store_document(self, url: str, markdown: str) -> AsyncGenerator[str, None]:
        """Process a document and store its chunks in parallel.
        
        Yields progress updates during processing and storage.
        """
        try:
            yield f"Processing {url} ...\n"
            chunks = self._chunk_text(markdown)
            
            tasks = [
                self._process_chunk(chunk, i, url)
                for i, chunk in enumerate(chunks)
            ]
            
            processed_chunks = []
            for completed in asyncio.as_completed(tasks):
                try:
                    chunk = await completed
                    processed_chunks.append(chunk)
                except Exception as chunk_error:
                    logfire.exception(f"Failed to process chunk for {url}: {chunk_error}")
            
            successful_chunks = [chunk for chunk in processed_chunks if not isinstance(chunk, Exception)]
            failed_chunks = [chunk for chunk in processed_chunks if isinstance(chunk, Exception)]
            
            if failed_chunks:
                logfire.exception(f"Failed to process {len(failed_chunks)} chunks for {url}")
                
            await self.db.insert_chunks(successful_chunks)
            yield f"OK {url}\n"
            
        except Exception as e:
            logfire.exception(f"Error processing document from {url}: {e}")
            yield f"Issues with {url}\n"

    async def _crawl_parallel(self, urls: List[str]) -> Dict[str, str]:
        """Crawl multiple URLs in parallel with a concurrency limit."""
        r = {}
        try:
            async with AsyncWebCrawler(config=self._browser_config) as crawler:
                results = await crawler.arun_many(urls, config=self._crawl_config)
                for result in results:
                    if result.success:
                        logfire.info(f"Successfully crawled {result.url}")
                    elif result.status_code == 403 and "robots.txt" in result.error_message:
                        logfire.warning(f"Skipped {result.url} - blocked by robots.txt")
                        continue
                    else:
                        logfire.exception(f"Failed to crawl {result.url}: {result.error_message}")
                        continue
                        
                    r |= {result.url: self._remove_markdown_links(result.markdown)} 

        except Exception as e:
            logfire.exception(f"{e}")
        
        return r

    async def study(self, urls: List[str]) -> AsyncGenerator[str, None]:
        try:
            assert isinstance(urls, list) and all(isinstance(u, str) for u in urls)
            url_markdown = await self._crawl_parallel(urls)
            # Create a shared queue for status messages.
            queue: asyncio.Queue = asyncio.Queue()
            # A unique object used as a sentinel to mark when a worker is done.
            sentinel = object()
            total_tasks = len(url_markdown)

            # This worker consumes an async generator and pushes messages to the queue.
            async def worker(url: str, markdown: str):
                async for status in self._process_and_store_document(url, markdown):
                    await queue.put(status)
                # Signal that this worker is done.
                await queue.put(sentinel)

            # Create a background task for each URL.
            tasks = [
                asyncio.create_task(worker(url, markdown))
                for url, markdown in url_markdown.items()
            ]

            # As long as not all workers have finished, yield messages as they arrive.
            finished = 0
            while finished < total_tasks:
                msg = await queue.get()
                if msg is sentinel:
                    finished += 1
                else:
                    yield msg

            # Optionally, ensure all tasks have completed (also raises exceptions if any occurred)
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logfire.exception(f"{e}")

    async def get_db_content(self, query, min_similarity = 0.6) -> tuple[dict[str, str], bool]:
        processed_metadata, embedding, is_meta_success = await self._get_processed_meta(query)
        content = await self.db.get_content_chunks(embedding, processed_metadata.programming_languages, 3)
        filtered_content = []
        for c in content:
            logfire.info(f"Similarity: {c['similarity']:.4f}, Source: {c['source']}")
            if min_similarity < c['similarity']:
                filtered_content.append(c)
        
        is_meta_success = is_meta_success and len(filtered_content)

        return filtered_content, is_meta_success

            

