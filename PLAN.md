# Phase-Wise Development Plan
### Flipkart GRiD 8.0 · Return Triage Multi-Agent System

Companion to `DESIGN.md`. 4 people · ~1–2 days. Each phase has a **goal**, **tasks + owner**, **exit criteria** (definition of done), and a **time box**.

**Team roles** (from DESIGN.md):
- **A** — ML / Data (SQL, Data Agent, Risk model)
- **B** — Agents / Orchestration (LangGraph, Text Agent, fusion)
- **C** — Data generation + RAG (synthetic data, policy corpus, Chroma)
- **D** — Eval + Demo + Deck (harness, adversarial sets, UI, dial, presentation)

**Golden rule:** never leave a phase without its exit criteria met. Stub-first, integrate early.

---

## PHASE 0 — Pre-flight / Setup (2–3 h, whole team, BEFORE any feature code)

**Goal:** every blocker that could stall the team mid-build is cleared, every shared decision is frozen, and everyone can run the repo. Do NOT skip this — a missing API key or an undecided schema at hour 10 costs more than this whole phase.

### 0.1 Accounts, keys & external access
- [ ] **LLM API key** obtained and **tested with a real call** (Gemini billing enabled *or* Groq key). Confirm it returns valid **structured/JSON output** in your SDK — this is a hard dependency for the text agent.
- [ ] Note the provider's **rate limits** (RPM/TPM) and set up a `.env` + provider-agnostic client with **retry/backoff**; add **key rotation** if teammates each make a free key.
- [ ] **Kaggle account** — only if using IEEE-CIS/PaySim (optional). Download now if yes; skip if no.
- [ ] **Google Cloud account** created + **billing enabled** to unlock the $300 free credit; install `gcloud` CLI; enable Cloud Run API. (Primary deploy target, same ecosystem as Gemini. Note: Cloud Run = one port per service — deploy as **Pattern B**, a single Streamlit service that imports the orchestrator; see DESIGN §10.)
- [ ] **HuggingFace account** created as **fallback**; a Streamlit-SDK Space scaffolded (empty is fine). On this path the dashboard deploys to HF and FastAPI runs locally.

### 0.2 Repo, environment & dependencies
- [ ] **GitHub repo** created, `DESIGN.md` + `PLAN.md` committed, branch strategy agreed (feature branches per agent, frequent merges).
- [ ] Repo skeleton from DESIGN.md §11 scaffolded (empty modules).
- [ ] **Python 3.11** env pinned; `requirements.txt` with: `langgraph, xgboost, chromadb, sentence-transformers, streamlit, fastapi, uvicorn, pandas, scikit-learn, python-dotenv` + LLM SDK.
- [ ] **Smoke test:** everyone runs `pip install -r requirements.txt` and imports all libs clean. Confirm the embedding model downloads and XGBoost runs on each laptop.

### 0.3 Information gathering
- [ ] **Scrape Flipkart's public return-policy pages** and drop raw text into `data/policies/`.
- [ ] **Draft 3–5 mock policy docs** (non-returnable categories, condition rules, time windows) — assign each a `clause_id`.
- [ ] Decide the **business constants** for the dial: `CUSTOMER_LTV`, `REVIEW_COST` (state them on screen later).

### 0.4 Freeze these decisions (write them in the repo README — do not reopen)
- [ ] **SQLite schema** (DESIGN §3.1) — exact columns.
- [ ] **Shared-state object** (DESIGN §4) — what each agent reads/writes.
- [ ] **Output JSON contract** (DESIGN §5.2) — exact keys.
- [ ] **Verdict mapping** — how risk score + flags → Auto-Approve/Auto-Reject/Escalate, and the dual-label scheme (nature vs verdict).
- [ ] **Compute/demo machine** designated (which laptop runs the live demo).

**Exit criteria:** ✅ every teammate can clone the repo, install, make a successful LLM call, and everyone has signed off on the four frozen decisions. Deployment target exists as an empty Space.

---

## PHASE 1 — Stubbed end-to-end skeleton + data kickoff (3–4 h)

**Goal:** a pipeline that runs start-to-finish with every agent returning fake data, and synthetic-data generation underway. Working demo by end of this phase.

**Tasks:**
- **B** — LangGraph graph wiring all agents as stubs (each returns hardcoded output) → produces a valid output JSON. Deterministic pre-check layer as a stub too.
- **A** — SQLite schema created + a handful of seed rows so the Data Agent stub can read real SQL.
- **C** — start `generate_synthetic.py`: batched LLM reasons + numeric fields + dual labels + `clause_id` tags; begin the Chroma index over policy docs.
- **D** — start eval harness skeleton reading the stubbed output; hand-author the first ~10 adversarial cases.

**Exit criteria:** ✅ `run_batch.py` processes a batch of mock requests end-to-end and emits valid structured JSON (with stub logic). The repo "works." Data generation running.

---

## PHASE 2 — Real components in parallel (6–8 h)

**Goal:** each stub replaced by a real agent, tested in isolation.

**Tasks:**
- **A** — Data Agent (real SQL feature pulls) + Risk Scoring Agent (train XGBoost on synthetic behavioral features; output calibrated probability).
- **C** — finish 5–10k synthetic dataset (3 classes, all archetypes, tonal reasons); finish Chroma RAG returning the governing clause + `clause_id`.
- **B** — Text Reasoning Agent (LLM intent: implausible/contradictory/sarcastic) with JSON validation + retry.
- **D** — grow adversarial set toward 50–100 + escalation set 30–50; wire each real agent's output into the harness as it lands.

**Exit criteria:** ✅ each agent works standalone on real data; risk model trained; RAG retrieves correct clauses on spot checks; full synthetic dataset + 20% stratified holdout ready.

---

## PHASE 3 — Orchestration & decision fusion (3–4 h)

**Goal:** swap stubs for real agents inside LangGraph; implement the real deterministic layer + fusion rules.

**Tasks:**
- **B** — deterministic initial-validation gates; fusion logic (thresholds → 3 verdicts) including the two precedence rules (current evidence overrides history; suspicious current signal overrides trust); **grounded justification** (facts templated programmatically, LLM writes prose only); escalation dossier generator.
- **A** — feed real features into fusion; help calibrate thresholds.
- **D** — run the harness after each swap; report metrics back to B for tuning.

**Exit criteria:** ✅ real end-to-end pipeline produces verdict + grounded justification + policy citation + dossier, all agents completing in sequence with valid structured output.

---

## PHASE 4 — Evaluation & tuning (3–4 h)

**Goal:** hit the graded targets; produce the deliverable reports.

**Tasks:**
- **D** — harness emits all 5 metrics: Recall@≤5% FPR, escalation coverage + automation rate, explainability faithfulness (0–3), RAG adherence, orchestration reliability + latency. Plus plain Precision/Recall and a quantified FP-risk number.
- **B + A** — tune fusion thresholds to **≤5% FPR** and **>70% automation**.
- **D** — draft the **Failure Analysis** note from adversarial-set results (sarcastic reasons, deceptive photos, quantified FP risk).

**Exit criteria:** ✅ all 5 metrics computed and reported; automation >70% and FPR ≤5% achieved (or the tradeoff consciously documented); Failure Analysis drafted.

---

## PHASE 5 — UI + Wow dial (3–4 h)

**Goal:** the demo layer.

**Tasks:**
- **D** — Streamlit dashboard: single-case adjudication view (watch a case flow, see the trail), escalation queue with dossiers, metrics panel.
- **D + B** — the **Risk Appetite dial**: precompute every eval case's risk score + flags ONCE, cache, slider re-applies threshold only → live recalculation of recall / FPR / automation / ₹ net value. (Never re-run agents on slider move.)

- *(optional, LOW priority)* — escalation-queue alert/badge as a lightweight "notification" (DESIGN §15). Only if ahead of schedule.

**Exit criteria:** ✅ dashboard runs; dial recalculates instantly and correctly; a full case demoable through the UI.

---

## PHASE 6 — Deploy, deck & rehearsal (2–3 h)

**Goal:** shippable and pitch-ready.

**Tasks:**
- **C/D** — deploy to **Google Cloud Run** (primary, **Pattern B**): one Streamlit service that imports the orchestrator (no CORS hop); `Dockerfile` listens on `$PORT` → `gcloud run deploy` → set `GEMINI_API_KEY` secret → region `us-central1` → `min-instances=1` to avoid cold start. Confirm public URL loads. FastAPI runs local for the "API-speed" moment. *Fallback:* HF Streamlit Space (also Pattern B).
- Run the **pre-deploy checklist** (DESIGN §10): warm the URL, key as secret, **bundle SQLite + `risk_model.pkl` + persisted Chroma index inside the image**, pin `requirements.txt`, cache for rate limits, container listens on `$PORT`.
- **D** — build the deck: problem → architecture → the 5 metrics you hit → live demo → the dial → failure analysis → future work.
- **Whole team** — run the demo end-to-end **twice**; record a **backup video** in case venue networking fails.

**Exit criteria:** ✅ public demo URL live; deck done; demo rehearsed twice; backup recording saved.

---

## Cut-list (shed in this order if time runs short)
1. Image handling → replace with an honest note in Failure Analysis.
2. Public deployment → demo locally instead (URL is a nice-to-have, not graded).
3. FastAPI hosting → keep it local only.
**Never cut:** eval harness, grounded explainability, the risk dial, the 3-verdict logic.

---

## One-line phase summary
**0** Pre-flight & freeze decisions → **1** Stubbed skeleton → **2** Real agents in parallel → **3** Orchestrate & fuse → **4** Eval & tune → **5** UI & dial → **6** Deploy, deck, rehearse.
