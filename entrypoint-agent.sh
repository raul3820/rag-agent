#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# --- Check for POSTGRES_URL ---
if [ -z "$POSTGRES_URL" ]; then
  echo "Error: POSTGRES_URL environment variable is not set."
  exit 1
fi

# --- Wait for Postgres to be ready ---
echo "Waiting for Postgres at $POSTGRES_URL..."
# The -q flag makes pg_isready quiet; it returns a zero exit code when ready.
while ! pg_isready -d "$POSTGRES_URL" -q; do
  echo "$(date) - Postgres is unavailable - sleeping"
  sleep 2
done
echo "Postgres is up and running."

# --- Start the uvicorn server ---
# AGENT_PORT defaults to 11433 if not provided.

if [ "$DEBUG" = "true" ]; then
  echo "DEBUG mode is enabled, using --reload"
  uvicorn --host 0.0.0.0 --port "${AGENT_PORT:-11433}" --reload run:app
else
  uvicorn --host 0.0.0.0 --port "${AGENT_PORT:-11433}" run:app
fi