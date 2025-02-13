#!/bin/sh

# Function to wait until the Ollama service is available
wait_for_ollama() {
  echo "Waiting for Ollama service to start..."
  until curl --silent http://localhost:11434 > /dev/null; do
    sleep 2
  done
  echo "Ollama service is up."
}

# Start Ollama in the background and wait
ollama serve &

# Wait for Ollama before pulling models
wait_for_ollama

# Pull models
models=" 
$BIG_MODEL 
$SMALL_MODEL 
"

echo "$models" | sort | uniq | while read -r model; do
  if [ -n "$model" ]; then
    echo "Pulling model: $model"
    ollama pull "$model"
  fi
done

# Keep the container running by keeping ollama serve running in the foreground
wait $!
