# production-genai

Every layer of a production GenAI system as code you can run: foundations, model training & serving, providers, prompting, RAG, memory, tools, agents, multi-agent, multimodal, evaluation, safety and LLMOps. 17 layers, 65 subsections, 600 topics — real APIs, real outputs saved in every notebook, one page per topic with diagrams and trade-offs.

## How every topic is written

One page per topic, always in the same order:

| part | what you get |
|---|---|
| **Problem** | the concrete situation, in everyday words |
| **Idea** | one sentence |
| **Use when / Not when** | the decision a senior engineer makes |
| **Diagram** | a flowchart or arrow drawing that carries the explanation |
| **How it works** | 3–5 steps |
| **✗ / ✓** | the wrong way and the right way, with numbers |
| **Production code** | the real library, real settings, executed — output saved |
| **What the output shows** | the printed result, read back to you |
| **In practice** | what bites in real systems, and why |
| **Alternatives** | the other ways, and why this one |
| **Terms** | the two or three words you need |

Production code only: the libraries real teams ship with, typed settings, timeouts, retries, structured logs. No toy code, no mocks — real API calls, real models, real outputs.

## The layers

Follows [GenAI_Master_Topic_List.md](GenAI_Master_Topic_List.md). One folder per layer, one folder per subsection, one notebook per bullet.

| layer | status |
|---|---|
| [0.1 Python for GenAI](00_foundations/0_1_python_for_genai/README.md) | done — 22 notebooks, one per topic |
| [0.2 Math / ML basics](00_foundations/0_2_math_ml_basics/README.md) | done — 7 notebooks |
| 0.4 Deep learning basics | in progress |
| 1 Model layer · 2 Provider layer · 3 Prompt engineering · 4 Data & ingestion · 5 RAG | next |
| 6 Memory · 7 Tools & protocols · 8 Agents · 9 Agentic RAG · 10 Multi-agent | planned |
| 11 Multimodal · 12 Evaluation · 13 Safety · 14 LLMOps · 15 Apps · 16 Cost · 17 Emerging | planned |

## Run it

```bash
git clone https://github.com/bdpgmbv/production-genai
cd production-genai
uv sync
cp .env.example .env        # add your OPENAI_API_KEY
uv run jupyter lab
```

Every notebook runs on its own: the first cell only loads `.env` and picks the model. Running all of 0.1 costs about $0.03 in API calls.

## License

MIT
