# Observability & Evaluation for the Multi-Agent RAG System

> 🪟 **On Windows?** This README and the `docs/` guides use bash/zsh. A full PowerShell edition lives in **[`README_powershell.md`](README_powershell.md)** (with [`docs/local_deployment_powershell.md`](docs/local_deployment_powershell.md) and [`docs/gcp_deployment_powershell.md`](docs/gcp_deployment_powershell.md)).

In week 4 we shipped the multi-agent **Agentic RAG** system: a supervisor that delegates to a
knowledge-base agent (which calls an **MCP Search sidecar** over Streamable HTTP, backed by Qdrant)
and a web-search agent, on **Cloud Run**. This week we make that same system **observable** and
**evaluated** — without changing what the agent *does*. (Wiring the eval into a **CI/CD** pipeline
is **Week 6 — AgentOps**; this week you run the eval commands yourself.)

Same "one more thing at a time" spirit:

- ✅ **Observability** — ADK traces via **OpenTelemetry + OpenInference**: **Phoenix** for local dev
  (free) and **Langfuse** for production (**required** for the Cloud Run deploy). Same agent code,
  swap the endpoint.
- ✅ **Evaluation** — a `pytest` suite in [`eval/`](eval/) scoring both the **trajectory** (did it
  take the right path?) and the **final response** (grounded? correct?).
- ✅ **A regression gate** — a baseline the suite is compared against (`make gate`), so a change that
  makes the agent worse **fails the gate** when you run it. This is "preparing for when things
  change." (Week 6 runs this gate automatically in CI.)
- ⚠️ **You redeploy this week.** The change lives *inside the image* (observability deps + the
  instrumentation), so you rebuild and roll a new revision — locally first, then Cloud Run.

## What's in this folder

```
week5/
├── README.md                 ← you're reading it (overview)
├── README_powershell.md      ← Windows / PowerShell edition
├── Makefile                  ← local docker-compose workflow + trace + eval targets
├── .env.example              ← template (Cohere, Qdrant, Exa, GCP, + Week-5 OTEL_*/GOOGLE_API_KEY)
├── docker-compose.yml        ← local dev: agent + mcp + qdrant + postgres + ingestion (+ phoenix profile)
├── service.yaml              ← Cloud Run: one service, two containers (agent + mcp sidecar)
├── pytest.ini                ← pytest config for the eval suite
├── docs/
│   ├── local_deployment.md       ← step-by-step local stack + tracing + verify + eval  (bash)
│   ├── gcp_deployment.md          ← Cloud Run deploy + observability + ONLINE EVAL (bash)
│   ├── local_deployment_powershell.md
│   ├── gcp_deployment_powershell.md
│   ├── observability_concepts.md ← concepts: the telemetry stack (OTel, OTLP, OpenInference) + glossary
│   └── evaluation_concepts.md    ← concepts: offline gates, online scoring & the eval metrics
├── agent/                    ← the ADK agent (its own image)
│   ├── Dockerfile            ← agent image (uv-based)
│   ├── pyproject.toml        ← agent deps + observability; the [eval] extra (host tooling)
│   ├── main.py               ← FastAPI entrypoint; calls init_observability()
│   └── src/
│       ├── agent.py          ← supervisor + kb_agent + web_search_agent
│       ├── exa_tool.py       ← web search tool
│       └── observability.py  ← OpenInference → OTLP wiring (Week 5)
├── mcp/                      ← the Streamable-HTTP MCP Search server (its own image)
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── ...
├── ingestion/                ← Cloud Run Job that loads the corpus into Qdrant
└── eval/                     ← the evaluation suite + regression gate (see eval/README.md)
```

## The system

One Cloud Run **service**, two **containers** (separate images) sharing `localhost`:

- **agent** — the supervisor (`general_assistant`) delegating to **`kb_agent`** and
  **`web_search_agent`**, on port 8080.
- **mcp** — the Search server (`search_article` over Qdrant), reached by the agent at
  `MCP_SERVER_URL` (`http://localhost:3000/mcp` in the same pod).

Plus Cloud SQL (sessions), Secret Manager (keys), Qdrant (vectors), Cohere (LLM + embeddings),
Exa (web). Locally, `docker-compose` mirrors this exactly: `agent` + `mcp` + a local `qdrant` +
`postgres` + a one-shot `ingestion` job.

> New to the telemetry vocabulary (OTel, OTLP, OpenInference)? See
> [`docs/observability_concepts.md`](docs/observability_concepts.md) — a short explainer + glossary
> (the wiring itself is in [`agent/src/observability.py`](agent/src/observability.py)).

## Prerequisites

1. A Google Cloud account with billing enabled, and the **gcloud CLI**.
2. **Docker Desktop** running locally.
3. A **Cohere API key** — <https://dashboard.cohere.com/> — the agent's model + embeddings.
4. An **Exa API key** — <https://dashboard.exa.ai/> — the web-search tool.
5. A **Qdrant Cloud** cluster for the cloud deploy (local dev uses the Qdrant container).
6. **[uv](https://docs.astral.sh/uv/)** — the Python package manager, needed to **run the eval suite**
   on your host (`make eval-deps` uses it to build an isolated `.venv`). Install once:
   `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux, or `brew install uv`) — see the
   [install docs](https://docs.astral.sh/uv/getting-started/installation/). *(Not needed for the
   deploy itself — the Docker build carries its own uv; this is only for local eval.)*
7. *(Optional)* a **Gemini API key** — <https://aistudio.google.com/apikey> — for the LLM-judge
   eval criteria. The judge model is independent of the Cohere agent.
8. *(Optional)* a free **Langfuse** project — for production tracing + online evaluation.

## Setup (one-time)

Copy the env template and fill in your own values — **don't edit `.env.example` directly**:

```bash
cp .env.example .env
# then edit .env and replace every placeholder
```

Load it into your shell so `docker compose`, `gcloud`, and `make` all inherit it:

```bash
set -a; source .env; set +a
```

> 💡 **New terminal later?** Re-run `set -a; source .env; set +a` — env vars don't persist across shells.

## Run it

Two paths. **Run it locally first** — same image, same code, just on your laptop. You'll catch
missing env vars and broken imports in seconds instead of waiting for a Cloud Build.

### Locally with `docker-compose`

See **[`docs/local_deployment.md`](docs/local_deployment.md)** — `make up`, talk to it at `/dev-ui`,
turn on local **Phoenix** tracing, run the **eval suite**, tear down.

### On Google Cloud Run

See **[`docs/gcp_deployment.md`](docs/gcp_deployment.md)** — every `gcloud` command spelled out:
APIs, Cloud SQL, secrets, IAM, build, ingest, deploy the two-container service, production tracing,
and online evaluation.

## Observability & evaluation at a glance

- **Tracing** is governed by one env toggle (`OBSERVABILITY_ENABLED` + an OTLP endpoint).
  **Local dev → Phoenix:** it's **on by default** — `make up` starts the stack *and* a Phoenix UI,
  with the agent already exporting to it (local guide §3); set `OBSERVABILITY_ENABLED=false` to opt out.
  **Production → Langfuse (required):** the Cloud Run `service.yaml` ships with the same toggle on,
  pointed at Langfuse, so the deploy needs a Langfuse secret (deploy guide Steps 3b + 11). Same agent
  code; only the endpoint changes. The trace shows the supervisor → `transfer_to_agent` → the
  sidecar's `search_article` → synthesis.
- **Evaluation** lives in [`eval/`](eval/) — see [`eval/README.md`](eval/README.md). Bring the stack
  up with **`make up`** (tracing on, so eval runs land in Phoenix), then score the running system:
  **`make eval`** (fast deterministic suite) and **`make gate`** (regression gate), or **`make
  eval-judge`** / **`make gate-judge`** to add the Gemini-judge suite over the grounded AI-Index
  evalset. **You run these yourself this week** — in **Week 6 (AgentOps)** the same commands run
  automatically in **GitHub Actions** on every push/PR.
- **Online evaluation** scores **live traffic** (not fixtures) in **production**: **Langfuse**'s
  managed **Context Relevance** evaluator samples incoming traces and scores **retrieval relevance**
  (are the chunks `search_article` retrieved relevant to the query?) on real sessions
  automatically — no script, no judge call in the request path. It's set up at the end of the cloud
  deploy — see [`docs/gcp_deployment.md`](docs/gcp_deployment.md) **Step 12** (online eval lives in
  its own section there). (Online eval is a production concern, so there's no local version — locally
  you only have your own dev-UI clicks.)

## Evaluation: judge on Vertex AI (and free-tier fallback)

The eval suite drives two **rate-limited** providers — the Cohere agent and the Gemini judge — so the
defaults are tuned to keep `make eval-judge` / `gate-judge` runnable.

**Gemini judge → Vertex AI (the configured default).** Google's *AI-Studio* free tier is brutal:
~**20 requests per DAY** for `gemini-2.5-flash` (quota `GenerateRequestsPerDayPerProjectPerModel-FreeTier`),
which a single `num_samples=5` run exhausts in one shot. So the judge is pointed at **Vertex AI**
instead, which bills to your GCP project / free-trial credits and has production-scale quota. Config
lives in `.env`:

```bash
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=<your-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

Auth is via Application Default Credentials — run this once (and re-run if you see a `RefreshError`):

```bash
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com   # one-time, enables Vertex AI on the project
```

The eval host loads `.env` itself (via `eval/_envload.py`), so you don't need to `source .env` first.
With Vertex, the judge uses the full config — `gemini-2.5-flash`, `num_samples: 5`.

**Cohere agent → client-side rate limit.** Trial keys cap at **20 requests/min**. The agent's
`LiteLlm` wrapper has a sliding-window limiter, **off by default** (production on a paid key runs full
speed); the `make eval*` / `gate*` targets switch it on via `LLM_MAX_RPM` (default `18`). On a paid
Cohere key: `make eval-judge LLM_MAX_RPM=0`.

**Small judge set on purpose.** `make eval-judge` runs `report_qa.evalset.json` (4 happy-path cases,
scored by `final_response_match_v2` only — reliably green) — kept small not for Gemini quota (Vertex
has plenty) but because the **Cohere agent** is slow and rate-limited. Groundedness (`hallucinations_v1`)
is **opt-in** (it's flaky — returns NOT_EVALUATED on a hiccup); enable it via the `_groundedness_optin`
block in `eval/fixtures_judge/test_config.json`. Need heavier coverage? Add cases or point
`JUDGE_DATASET` at your own evalset.

**Free-tier fallback (no GCP billing).** Set `GOOGLE_GENAI_USE_VERTEXAI=FALSE` + a `GOOGLE_API_KEY`
in `.env` to use AI Studio instead — but mind the ~20 req/day cap: you'd want `num_samples: 1` and a
small dataset, and you'll typically get one run/day.

## What changes when you go from local to cloud

The agent code does not change — env vars route to the right backends.

| | Local (`docker-compose`) | Cloud Run |
|---|---|---|
| MCP transport | `agent` → `mcp` container (`http://mcp:3000/mcp`) | `agent` → `mcp` sidecar (`http://localhost:3000/mcp`) |
| Vector store | local `qdrant` container | **Qdrant Cloud** |
| Sessions | local `postgres` container | **Cloud SQL** (Auth Proxy socket) |
| Keys | `.env` → docker-compose | **Secret Manager** → Cloud Run |
| Tracing backend | **Phoenix** (compose service, on by default with `make up`) | **Langfuse** — required (deploy guide Steps 3b + 11) |
| Online eval | — (not run locally) | Langfuse UI online evaluators (managed) |

**One agent image. Two configurations. That's portable infrastructure.**

## What's still out of scope

- 🔜 **CI/CD** — this week you run the eval commands by hand. **Week 6 (AgentOps)** wires them into a
  GitHub Actions pipeline (and covers turning this into a proper repo, branch protection, etc.).
- 🔜 **Auth on the endpoint** — we still deploy public for the lab; real auth (IAP/OAuth) comes later.
