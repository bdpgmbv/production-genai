# 0.1 Python for GenAI

`needs: openai, pydantic, pydantic-settings, python-dotenv, httpx, requests, tenacity, aiolimiter, structlog, pyright, uv, OPENAI_API_KEY`

One notebook per bullet of the master list. Each item is one page: problem · idea · use when · diagram · how it works · ✗/✓ · production code with real output · what the output shows · in practice · alternatives · terms.

- [async/await, asyncio.gather, concurrency for API calls](01_async_await_gather_concurrency.ipynb) — async/await · asyncio.gather · concurrency for API calls
- [Pydantic v2 models, validators, JSON schema generation](02_pydantic_models_validators_schema.ipynb) — Pydantic v2 models · validators · JSON schema generation
- [Generators / streaming iterators](03_generators_streaming.ipynb) — Generators · streaming iterators
- [Typing, dataclasses, Protocols](04_typing_dataclasses_protocols.ipynb) — Typing · dataclasses · Protocols
- [Environment/config management (dotenv, pydantic-settings)](05_config_dotenv_settings.ipynb) — dotenv · pydantic-settings
- [Packaging (uv, poetry), virtual envs](06_packaging_uv_poetry_venv.ipynb) — uv · poetry · virtual envs
- [HTTP clients (httpx, requests), retries with tenacity](07_http_httpx_requests_tenacity.ipynb) — httpx · requests · retries with tenacity
- [Logging, structlog](08_logging_structlog.ipynb) — Logging · structlog
- [Break → Fix](09_break_fix.ipynb) — Break → Fix
