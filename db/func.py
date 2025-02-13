import os
import asyncio
import asyncpg
import logfire
import json
from asyncpg import Pool
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from db.struct import ProcessedChunk
load_dotenv()


class DBFunctions:
    def __init__(self):
        self.pool: Optional[Pool] = None

    async def connect(self):
        """Initialize the database connection pool."""
        postgres_url = os.getenv("POSTGRES_URL")

        if not postgres_url:
            raise ValueError("POSTGRES_URL environment variable not set")

        try:
            pool = await asyncpg.create_pool(dsn=postgres_url)
            logfire.info("Database pool created successfully")
            self.pool = pool
        except Exception as e:
            logfire.exception(f"Error creating DB pool: {e}")
            raise

    async def create_schema(self):
        """Create database schema using the SQL script."""
        if not self.pool:
            raise ValueError("Database pool not initialized")

        script_path = os.path.abspath(__file__)
        db_dir = os.path.dirname(script_path)
        sql_file = os.path.join(db_dir, 'create.sql')

        with open(sql_file, 'r') as f:
            sql = f.read()
        
        embeddings_ndim = os.getenv("EMBEDDINGS_NDIM")
        if embeddings_ndim is None:
            logfire.exception("Error: EMBEDDINGS_NDIM environment variable is not set.")
            return
        sql = sql.replace("${embeddings_ndim}", embeddings_ndim)
        
        postgres_url = os.getenv("POSTGRES_URL")
        if postgres_url is None:
            logfire.exception("Error: POSTGRES_URL environment variable is not set.")
            return
        
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(sql)
            logfire.info("Schema created successfully")
        except Exception as e:
            logfire.exception(f"Error executing SQL: {e}")
            raise

    async def _insert_chunk(self, chunk: ProcessedChunk) -> Optional[dict]:
        """Insert a processed chunk into the database."""
        if not self.pool:
            logfire.exception("Database pool not initialized")
            raise ValueError("Database pool not initialized")

        try:
            query = """
                INSERT INTO site_pages (
                    source, chunk_number, metadata, processed_metadata, embedding, content
                ) VALUES (
                    $1, $2, $3, $4, $5, $6
                )
                ON CONFLICT (source, chunk_number) 
                DO UPDATE SET
                    updated_at = timezone('utc'::text, now()),
                    metadata = EXCLUDED.metadata,
                    processed_metadata = EXCLUDED.processed_metadata,
                    embedding = EXCLUDED.embedding,
                    content = EXCLUDED.content
                RETURNING id;
            """
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    query,
                    chunk.source,
                    chunk.chunk_number,
                    json.dumps(chunk.metadata),
                    chunk.processed_metadata.model_dump_json(),
                    json.dumps(chunk.embedding),
                    chunk.content,
                )
                
            logfire.info(f"Inserted chunk {chunk.chunk_number} for {chunk.source}")
            return {"id": row["id"]}
        
        except Exception as e:
            logfire.exception(f"Error inserting chunk: {e}", exc_info=True)  # Include exception info
            return None

    async def insert_chunks(self, chunks: List[ProcessedChunk]):
        """Insert multiple chunks into the database in parallel."""
        if not self.pool:
            raise ValueError("Database pool not initialized")

        insert_tasks = [
            self._insert_chunk(chunk)
            for chunk in chunks
        ]
        await asyncio.gather(*insert_tasks)

    async def get_content_chunks(self, query_embedding: List[float], langs: List[str] = [], match_count: int = 3) -> List[Dict[str, str]]:
        """
        Call the get_chunks PostgreSQL function to retrieve relevant chunks based on the query embedding.

        Args:
            param query_embedding: The embedding vector to query against.
            param match_count: The number of matching chunks to return.
            return: A list of dictionaries containing 'source' and 'content' for each chunk.
        """
        if not self.pool:
            raise RuntimeError("Database connection pool is not initialized")

        try:
            query = """
                with 
                sources as (
                    select
                    source,
                    max(1 - cosine_distance) as similarity
                    from (
                        select
                            source,
                            site_pages.embedding <=> $2::vector as cosine_distance
                        from site_pages
                        where exists (
                            select 1
                            from jsonb_array_elements_text(processed_metadata->'programming_languages') as langs
                            where (case when array_length($1::text[], 1) is null then true else langs = any($1::text[]) end)
                            )
                        order by cosine_distance
                        limit $3
                    ) s
                    group by 1                
                ) 
                select
                source,
                string_agg(content, '' order by chunk_number) as content,
                max(similarity) as similarity
                from site_pages
                inner join sources using(source)
                group by source 
                order by similarity desc;
            """

            async with self.pool.acquire() as connection:
                # Execute the query directly with the vector embedding and match count
                result = await connection.fetch(query, langs, json.dumps(query_embedding), match_count)

                # Convert the result to a list of dictionaries
                chunks = [dict(row) for row in result]
                return chunks
        except Exception as e:
            logfire.exception(f"Error retrieving chunks: {e}")
            raise

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logfire.info("Database pool closed successfully")
