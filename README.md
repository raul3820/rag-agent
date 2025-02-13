# rag-agent

This service enables Large Language Models (LLMs) to "study" web pages.

By routing your OpenAI API chat completion requests through this service, you can enable the following workflow:

  * From your chat interface, the `#study site.com` command allows you to crawl and process web pages.
  * In subsequent chat conversations, relevant context from the studied pages will be automatically incorporated into your prompts before they are sent to the OpenAI API.

This service intercepts OpenAI API calls specifically for chat completion requests, and only when the model name in the request matches the model specified by the `BIG_MODEL` variable.

## How to Install

This project utilizes [Docker Compose](https://docs.docker.com/compose/install/), allowing for easy local deployment.

To get started:

1.  **Clone the repository:**
  ```bash
  git clone https://github.com/raul3820/rag-agent.git
  ```

2.  **Create an `.env` file** in the repository root directory and define the following environment variables:

  ```env
  POSTGRES_PASSWORD = ...
  API_URL_OPENAI = http://ollama:11434 or [https://api.openai.com](https://api.openai.com) or any other service compatible with the OpenAI API
  API_KEY_OPENAI = ...
  BIG_MODEL=deepseek-r1:14b  # Model for general chat interactions
  SMALL_MODEL=llama3.2:3b   # Model for smart parsing, query generation, and tool utilization
  ```

3.  **Run the services:**

  You can choose to run only the essential services:
  ```bash
  docker compose up agent postgres infinity ...
  ```

  Alternatively, to run all services locally, use:
  ```bash
  docker compose up
  ```

### Installation on Ubuntu with Docker and NVIDIA GPU Support

For users with Ubuntu, Docker, and NVIDIA GPUs, follow these steps:

1.  **Install Docker:**

  ```bash
  sudo apt-get update
  sudo apt-get install ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  ```

**Install Docker Engine:**

  ```bash
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
  ```

**Start and enable Docker:**

  ```bash
  sudo systemctl start docker
  sudo systemctl enable docker
  ```


2.  **Install NVIDIA Container Toolkit:**

Refer to the official NVIDIA documentation for detailed information:
[NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Execute the following commands for installation:**

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  ```

3.  **Configure Docker to use NVIDIA runtime:**

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

4.  **Verify NVIDIA Docker setup:**

  ```bash
  sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
  ```

## Contributions

This is an open work in progress, feel free to fork or propose changes.