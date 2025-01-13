# 0.1 Python for GenAI

One notebook per topic. Each one starts from a real production situation, shows the code that fixes it, and reads the output back.

| # | Notebook | What you will see |
|---|---|---|
| 01 | [async/await](01_async_await.ipynb) | A FastAPI service: three customers wait in a queue with a blocking endpoint, and are served together with an async one |
| 02 | [asyncio.gather](02_asyncio_gather.ipynb) | 40 tickets embedded one by one vs all at once |
| 03 | [concurrency for API calls](03_concurrency_for_api_calls.ipynb) | 60 tickets under a semaphore and a rate limiter; reading the provider's rate-limit headers |
| 04 | [Pydantic models](04_pydantic_models.ipynb) | A signup endpoint with and without a model: 200 vs 422, `"30"` → 30 |
| 05 | [validators](05_validators.ipynb) | An invoice rule that lives on the model, not in three scripts |
| 06 | [JSON schema generation](06_json_schema_generation.ipynb) | Extracting an invoice from an email with a schema, no parser |
| 07 | [Generators](07_generators.ipynb) | 100,000 tickets through read → filter → count with one in memory |
| 08 | [streaming iterators](08_streaming_iterators.ipynb) | A chat answer that appears word by word; first-word time vs finish time |
| 09 | [Typing](09_typing.ipynb) | pyright finding a renamed field before the code runs |
| 10 | [dataclasses](10_dataclasses.ipynb) | A cost record with no validation cost; frozen price table |
| 11 | [Protocols](11_protocols.ipynb) | One `ChatModel` capability, an OpenAI adapter and a fake for tests |
| 12 | [dotenv](12_dotenv.ipynb) | Keys in `.env`, never in code; per-environment files |
| 13 | [pydantic-settings](13_pydantic_settings.ipynb) | `REQUEST_TIMEOUT_SECONDS=30s` stopped at start-up, not in a request |
| 14 | [uv](14_uv.ipynb) | A project created, locked and run; what the lock file pins |
| 15 | [poetry](15_poetry.ipynb) | The same project with Poetry, and the command-for-command mapping |
| 16 | [virtual envs](16_virtual_envs.ipynb) | Where this notebook runs; a fresh environment that has nothing |
| 17 | [httpx](17_httpx.ipynb) | The raw request the SDK sends, its headers, and a 404 |
| 18 | [requests](18_requests.ipynb) | Session, retry adapter and the timeout that saves the 3 a.m. job |
| 19 | [retries with tenacity](19_retries_with_tenacity.ipynb) | A timeout retried four times; a wrong key not retried at all |
| 20 | [Logging](20_logging.ipynb) | Real log lines around a model call; silencing the SDK's HTTP chatter |
| 21 | [structlog](21_structlog.ipynb) | JSON events with a request id bound once |
| 22 | [Break → Fix](22_break_fix.ipynb) | `await` in a loop (slow, correct) vs gather + semaphore |
