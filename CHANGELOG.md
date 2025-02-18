## v0.1.0 - 2025-02-12

**Features:**

* Initial release of rag-agent!
* Implements `#study site.com` command for crawling web pages.
* Automatically adds context to prompts for chat conversations.
* Supports OpenAI API and compatible services.
* Automatically adds context to prompts during chat conversations.
* Supports OpenAPI API and compatible services.
* Docker Compose setup for easy local installation.

**Bug Fixes:**

* None in this initial release.

**Known Issues:**

* Sometimes the `SMALL_MODEL` fails to call available tools


## v0.1.1 - 2025-02-14

**Bug Fixes:**

* Fixes issue where streaming responses were not handled correctly in the OpenAI proxy.

**Improvements:**

* Optimizes the Docker Compose setup for Ollama service.


## v0.1.2 - 2025-02-18

**Performance Improvements:**

* Improved efficiency of the crawling algorithm for `#study site.com` command.

**Bug Fixes:**

* Fixed an issue with RAG programming language classification.