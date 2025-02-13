-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table if not exists site_pages (
    id bigserial primary key,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    source varchar not null,
    chunk_number integer not null,
    metadata jsonb not null default '{}'::jsonb,  -- general metadata
    processed_metadata jsonb not null default '{}'::jsonb,  -- processed metadata
    embedding vector(${embeddings_ndim}),  -- embeddings are ${embeddings_ndim} dimensions
    content text not null,  -- Added content column
    
    -- unique constraint to prevent duplicate chunks for the same URL
    unique(source, chunk_number)
  );

-- create an index for better vector similarity search performance  
create index if not exists site_pages_ivfflat_index on site_pages using ivfflat (embedding vector_cosine_ops);  

