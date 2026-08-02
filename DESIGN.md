# Return Triage Multi-Agent System — Design & Deployment Spec
### Flipkart GRiD 8.0 · Data Science Track

**Team size:** 4 · **Timeline:** ~1–2 days · **Deliverable focus:** working model/demo + code/notebook

This document is the single source of truth. Freeze it before writing code. Everything here maps back to a specific line in the case study PDF, and every design choice targets one of the five graded metrics.

---

## 1. Problem framing (what we are actually building)

We are building an **autonomous multi-agent triage system** that adjudicates e-commerce return requests into one of three **verdicts** — **Auto-Approve / Auto-Reject / Escalate to Human** — with a fully auditable justification trail. It is *not* a binary fraud classifier. The system must reason like a senior human reviewer (context + history + policy) at the speed of an API call, auto-deciding the clear cases and escalating the genuinely ambiguous ones.

Two vocabularies exist in the brief and must not be confused:

| Axis | Values | Purpose |
|---|---|---|
| **Ground-truth nature** (label) | Legitimate / Fraud / Borderline | What the case *actually is* — used only for scoring |
| **System verdict** (prediction) | Auto-Approve / Auto-Reject / Escalate | What our system *outputs* — the deliverable |

The system never sees "nature." It infers the verdict. Scoring compares the two.

---

## 2. System architecture

```
                        RETURN REQUEST
      (order metadata + behavioral history + free-text reason + optional image)
                              │
                    ┌─────────▼──────────┐
                    │  0. DETERMINISTIC   │   Hard policy gates BEFORE agents:
                    │   INITIAL VALIDATION │   return window? non-returnable category?
                    └─────────┬──────────┘   (mirrors status-quo journey step 2)
                              │  grey area / passes gate
                    ┌─────────▼──────────────────────────────┐
                    │        ORCHESTRATOR (LangGraph)          │
                    │  routes → gathers → fuses → decides       │
                    └──┬─────────┬──────────┬──────────┬───────┘
                       ▼         ▼          ▼          ▼
                 1.Data     2.Risk     3.Text      4.Policy-RAG
                   Agent      Scoring    Reasoning   Agent
                 (SQL pull)  (XGBoost)  (LLM intent) (Chroma retrieve
                  features    fraud prob  vs order)    governing clause)
                       │         │          │          │
                 5.(opt) Image Consistency check ──────┤
                       └─────────┴────┬─────┴──────────┘
                                      ▼
                        DECISION FUSION + THRESHOLD RULES
                                      ▼
              VERDICT + grounded justification trail + policy citation
                                      ▼
                    if Escalate → auto-generated HUMAN DOSSIER
```

Seven components: a deterministic pre-check, four core agents, an optional image check, and the orchestrator that fuses everything.

---

## 3. Data design

### 3.1 SQLite schema (the mock order backend)

```sql
accounts(
  account_id PK, account_age_days, total_orders, total_returns,
  return_to_order_ratio, prior_fraud_flags, trust_tier
)
orders(
  order_id PK, account_id FK, sku, category, order_value,
  order_date, delivery_scan_confirmed BOOL, event_adjacent BOOL
)
return_requests(
  return_id PK, order_id FK, reason_text, reason_tone,
  days_since_delivery, image_ref NULLABLE,
  -- scoring fields (never shown to the system):
  true_nature ENUM(Legitimate,Fraud,Borderline),
  fraud_archetype ENUM(none,wardrobing,empty_box,serial_returner),
  governing_policy_clause_id NULLABLE   -- for RAG adherence scoring
)
```

The **Data Agent** reads `accounts` + `orders` via SQL to build the behavioral feature vector — this is where your SQL skills apply.

### 3.1a Provided Kaggle datasets — our stance (all optional)

The brief offers three; our development data is **synthetic** (§3.2). Treatment:
- **IEEE-CIS Fraud Detection** — optional warm-up only. It's *financial-transaction* fraud, not return fraud, so features don't map cleanly. Use at most to sanity-check the XGBoost pipeline or as a "validated against real fraud data" credibility point. Not required.
- **PaySim** — offered for class imbalance; **skip**, we control class ratio directly in generation.
- **Home Credit** — feature-engineering *inspiration* only; already absorbed into our behavioral features. **Skip.**

Recommendation for a 1–2 day build: skip all three, invest the time in synthetic-data quality and the adversarial set.

### 3.1b How we obtain the data (there is nothing to download for the primary set)

- **Primary dataset = self-generated.** Run `data/generate_synthetic.py` → it calls Gemini for the free-text reasons and fills numeric/behavioral fields by rule → writes 5–10k labeled rows into `mock_orders.db` (SQLite) + a CSV. **No Kaggle, no download, no manual labeling** — return-fraud data doesn't exist publicly, so we manufacture it. Cache the LLM output to disk so it's generated once.
- **Policy corpus** (for RAG) = scraped Flipkart return-policy pages + 3–5 hand-written mock docs, dropped into `data/policies/`.
- **IEEE-CIS (optional only):** if used, it *is* a Kaggle download — create a Kaggle account, accept the competition rules on the page, then download the CSV in-browser **or** via the Kaggle API: `pip install kaggle`, place `kaggle.json` token in `~/.kaggle/`, run `kaggle competitions download -c ieee-fraud-detection`. Skip for the standard 1–2 day build.

### 3.2 Synthetic generation (5k–10k rows, 3 classes)

Generated by template, so labels are free. Per archetype, fill fields to the exact signatures in the brief:

- **Wardrobing (Fraud):** high `order_value`, `event_adjacent=true`, purchase→return gap short, reason "doesn't fit," first return.
- **Empty-box (Fraud):** `delivery_scan_confirmed=true`, high `account_age_days` but sudden claim spike, vague reason.
- **Serial returner (Fraud):** `return_to_order_ratio > 0.40`, many low-review SKUs.
- **Legitimate:** normal ratios, plausible reason, healthy account.
- **Borderline:** deliberately mixed signals (see §7 adversarial).

**LLM is used only for `reason_text`** — batch 50–100 per call, vary `reason_tone` across normal / evasive / sarcastic / contradictory, cache to disk. Every other field is sampled numerically.

**Critical:** each policy-relevant row records `governing_policy_clause_id` so RAG adherence is measurable, and every row records both `true_nature` and (at eval time) the predicted verdict.

**Splits:** hold out a **20% stratified classification set** that preserves both the class ratio and fraud-archetype diversity (brief Evaluation Dataset #1). The remaining 80% trains the risk model and tunes fusion thresholds. The adversarial (§7) and escalation-quality (§7) sets are separate, hand-authored, and never used for training.

### 3.3 Policy corpus (RAG source)

Scrape Flipkart's public return-policy pages **plus** author 3–5 mock policy docs covering: non-returnable categories, condition requirements, and time windows (e.g., 10-day electronics window). Chunk → embed (`sentence-transformers`, local, free) → store in **Chroma**. Each chunk keeps a `clause_id` so citations are checkable.

---

## 4. Agent specifications

| # | Agent | Input | Output | Tech |
|---|---|---|---|---|
| 0 | Deterministic pre-check | order meta | pass / hard-reject (window, non-returnable) | pure Python |
| 1 | Data Agent | account_id, order_id | feature vector (age, ratio, value, prior flags) | SQL over SQLite |
| 2 | Risk Scoring Agent | feature vector | calibrated fraud probability [0–1] | XGBoost/LightGBM |
| 3 | Text Reasoning Agent | reason_text, category, order facts | intent flags: implausible / contradictory / sarcastic + rationale | LLM (Gemini/Groq) |
| 4 | Policy-RAG Agent | case facts | retrieved governing clause + clause_id | Chroma + embeddings |
| 5 | Image check (optional) | image_ref | consistency flag (or "no image") | lightweight heuristic / CLIP |

Each agent writes into a **shared state object** the orchestrator owns. No agent talks to another directly.

---

## 5. Orchestration & decision logic

Built with **LangGraph**. Flow: `initial_validation → (parallel: data, risk, text, policy, image) → fusion → verdict → [dossier]`.

### 5.1 Fusion rules (where Precision↔Recall lives)

The fusion step combines the risk probability, the text-intent flags, and the policy verdict into a **single continuous `fused_fraud_confidence` ∈ [0,1]**, then derives the verdict by thresholds on it (this continuous score is what Criterion 1 sweeps — see §8.0):

- **High risk + policy violation + contradictory reason → Auto-Reject**
- **Low risk + clean reason + policy-compliant → Auto-Approve**
- **Conflicting signals, or confidence in the "grey band," or policy edge case → Escalate**

Two design rules that directly protect graded metrics:

1. **Strong current evidence can override a bad history.** A serial returner with a genuine, policy-compliant current claim must not be auto-rejected on profile alone — otherwise we fail the "serial returner with legitimate claim" adversarial case *and* the ≤5% FPR metric.
2. **Behavioral trust can be overridden by a suspicious current signal.** An old trusted account suddenly claiming empty-box on a delivery-scanned order should escalate, not auto-approve.

Thresholds are tuned against the eval harness (§8) to hit **≤5% false-positive rate** and **>70% automation rate**.

### 5.2 Output contract (structured JSON — deliverable 1)

```json
{
  "return_id": "R12345",
  "verdict": "Escalate",
  "risk_score": 0.62,
  "fused_fraud_confidence": 0.58,
  "policy_citation": {"clause_id": "RET-10D", "text": "10-day window for electronics"},
  "justification_trail": [
    "account_age_days = 14 (below trust threshold of 90)",
    "return_to_order_ratio = 0.62 (exceeds 0.40 serial-returner flag)",
    "reason 'arrived broken' contradicts delivery_scan_confirmed = true",
    "policy RET-10D satisfied (day 6 of 10)"
  ],
  "escalation_dossier": "…summary for human reviewer…",
  "latency_ms": 840
}
```

### 5.3 Grounded justification (protects the Explainability metric)

The rubric scores **0 for hallucinated numbers**. Therefore the factual lines in `justification_trail` are **templated from the real feature values and the retrieved clause programmatically** — the LLM only writes connecting prose, never invents figures. This is how we score 3/3 (grounded + policy-cited).

---

## 6. Deliverable → component mapping

| Brief deliverable | Produced by |
|---|---|
| **Working Pipeline** (batch → structured decisions) | Orchestrator + FastAPI `POST /adjudicate` + batch runner script |
| **Justification Log** (reasoning chain per case) | `justification_trail` written per case to a log file / table |
| **Performance Evaluation** (Precision & Recall) | Eval harness §8 — P/R report + Recall@FPR curve |
| **Failure Analysis** (edge cases, quantified FP risk) | Technical note driven by adversarial-set results §7 (§6.2) |

### 6.1 Performance Evaluation — how we demonstrate it
The literal deliverable is **plain Precision & Recall on a labeled Fraud-vs-Legitimate sample** (simpler than the graded Recall@FPR). We present it as a report/notebook + dashboard metrics panel with, layered:
1. **3×3 confusion matrix** — predicted verdict vs expected verdict.
2. **Plain Precision / Recall / F1 table** (the required artifact) — Fraud-vs-Legitimate framing: Auto-Reject = "predicted fraud," Auto-Approve = "predicted legit," **Escalate = abstain** (reported separately, not scored as wrong — per §8.0).
3. **Recall @ ≤5% FPR curve** + the full **5-criteria scoreboard** (beyond the ask).
4. The **risk dial** (§9b) as the interactive P/R tradeoff.

### 6.2 Failure Analysis — the technical note (structure)
Half qualitative, half quantitative. The note contains:
1. **Failure taxonomy** — for each of the 5 adversarial modes (sarcasm, deceptive image, contradictory signals, 1-day policy edge, serial-returner-with-legit-claim): what the system does, **where it breaks**, and a concrete example case with the system's *actual* output — including cases it gets **wrong** (honest failures score better than none).
2. **Quantified false-positive risk** (the number the brief demands): FP rate on the holdout (`legit Auto-Rejected / all legit`), FP rate **under adversarial stress** (typically higher), and the **₹ cost** of those FPs via the dial's `CUSTOMER_LTV` formula — turning "risk" into a business figure.
3. **Residual risk & mitigations** — deceptive-image weakness, sarcasm edge cases, what we'd fix with more time.
- *How we show it:* short markdown/PDF note + one slide; **every claim backed by a harness number.**

---

## 7. Evaluation datasets & edge-case coverage

Three distinct eval sets (brief, pages 7–8), none of which train the model:

1. **Classification holdout** — 20% stratified split from the synthetic data, preserving class ratio + fraud-pattern diversity. **Labeled by verdict** (Auto-Approve / Auto-Reject / Escalate) via the mapping Legitimate→Approve, Fraud→Reject, Borderline→Escalate. (Split defined in §3.2.)
2. **Adversarial probe set** — 50–100 hand-authored cases (below).
3. **Escalation quality set** — 30–50 borderline cases whose ground truth is "should escalate," measuring whether the system correctly *abstains* instead of making a confident wrong call.

### 7.1 Adversarial probe set — the five failure modes

Every stress case from the brief, mapped to how the system handles it:

| Adversarial case | Handling |
|---|---|
| Contradictory signals (trusted acct, empty-box) | Text agent flags contradiction vs delivery scan → fusion escalates |
| Sarcastic reason ("arrived broken lol") | LLM intent detection classifies tone → escalate/flag |
| Policy edge (window expired 1 day; grey category) | Deterministic layer + RAG → **Escalate**, not hard-reject |
| Deceptive image (undamaged, misleading angle) | Image consistency flag + explicit failure-analysis treatment |
| Serial returner w/ legitimate current claim | Fusion rule 1 — current evidence overrides history |

---

## 8. Evaluation harness — the 5 graded criteria (ELIMINATION-CRITICAL)

These five criteria (brief pages 8–9) are how the project is judged. The harness computes **all five** on every run. Each row below states the exact definition, how we measure it, the target, and the design feature that earns it. **Nothing here is optional.**

### 8.0 Correctness foundations (read first — these make the numbers valid)

Our system emits **three** verdicts, but several metrics are **binary**. Two design rules make the measurement correct and defensible:

- **The orchestrator emits a continuous `fused_fraud_confidence` ∈ [0,1]**, not just a discrete verdict. The verdict is derived by thresholds on it: `< low → Auto-Approve`, `> high → Auto-Reject`, `in between → Escalate`. Recall/FPR curves are swept over this continuous score; the operating verdict is just a chosen point on that curve.
- **Escalate is never a "wrongful rejection."** Only **Auto-Reject of a truly Legitimate** case is a false positive. Escalating a legit case is a safe, non-penalized outcome (a human will confirm).

**Ground-truth field each metric reads (from §3.1 schema):**

| Criterion | Ground-truth used | Eval set |
|---|---|---|
| 1. Recall@FPR | `true_nature` = Fraud vs Legitimate | holdout |
| 2a. Escalation coverage | "should escalate" label | escalation-quality set (§7) + Borderline in holdout |
| 2b. Automation rate | `true_nature` = Fraud/Legitimate (clear only) + correct verdict | holdout |
| 3. Faithfulness | input record (feature values) + policy corpus | all decisions (+ sample for LLM-judge) |
| 4. Policy adherence | `governing_policy_clause_id` + expected verdict | policy-grounded subset |
| 5. Orchestration | run logs (errors, order, schema, latency) | full batch |

### Criterion 1 — Fraud Recall @ controlled FPR (primary safety metric)
- **Definition:** % of actual fraud caught (Recall) at a fixed FPR (**≤5%** wrongful *auto-rejections* of legitimate customers).
- **Exact computation:** sweep threshold `t` over `fused_fraud_confidence`. At each `t`: `Recall = fraud with conf ≥ t / all fraud`; `FPR = legit with conf ≥ t / all legit`. Report **Recall at the largest t where FPR ≤ 5%**, plus the full Recall–FPR (ROC) curve. Escalated cases sit below the Auto-Reject threshold, so escalating a legit case is *not* an FP.
- **Target:** maximize Recall subject to FPR ≤ 5%.
- **Earned by:** calibrated XGBoost risk + fused confidence (§5.1); the risk dial (§9b) *is* this curve made interactive.

### Criterion 2 — Escalation Calibration (two sub-metrics)
- **(a) Escalation Coverage:** `cases the system Escalated / all truly-borderline cases` — should be **high**. Measured on the escalation-quality set (§7) + Borderline rows in the holdout.
- **(b) Automation Rate:** `clear cases given the CORRECT auto-verdict / all clear cases` — **target >70%**. A clear fraud that is Auto-*Approved* is automated but **wrong**, so it does **not** count; nor does any Escalated clear case (not automated). Both conditions — auto-decided **and** correct — must hold.
- **Tension to watch:** widening the escalate band raises coverage but lowers automation. Tune to satisfy both.
- **Earned by:** the 3-verdict fusion logic and the deliberately-tuned escalate band (§5.1).

### Criterion 3 — Explainability Faithfulness (rubric 0–3)
- **Definition:** does each justification cite the *specific signals / feature values* **that actually drove the decision**, and are those signals *factually correct from the input*?
- **Exact rubric:** **0** = generic/hallucinated · **1** = partially grounded · **2** = fully grounded · **3** = grounded **+ policy-cited**.
- **How we measure (two parts):** (1) **programmatic** — every number in `justification_trail` must match the source record exactly (catches hallucination, the 0-trap) and a valid `clause_id` must be present (the +1 for level 3); (2) **"driving-signal" check** — number-matching alone can't prove the cited signals are the ones that *drove* the verdict, so run an **LLM-as-judge (or manual rubric) on a sample** to confirm the cited evidence aligns with the fusion decision. Combine both into the 0–3 score.
- **Target: 3 on every decision.**
- **Earned by:** grounded justification (§5.3) — the driving signals come straight from the fusion step's inputs, templated programmatically + the retrieved clause; LLM writes prose only. Designed to hit level 3 and structurally avoid the 0-trap.

### Criterion 4 — RAG Policy Adherence
- **Definition:** % of decisions that correctly **cite AND apply** the relevant policy clause (e.g., applying the 10-day window; flagging a non-returnable category).
- **Exact computation:** on the policy-grounded subset, a case scores a **pass only if BOTH**: (1) cited `clause_id` == ground-truth `governing_policy_clause_id`, **and** (2) the verdict is *consistent* with that clause's implication (e.g., window-expired clause ⇒ verdict is Reject or Escalate, not Approve). Citing the right clause but reaching an inconsistent verdict = **fail**.
- **Earned by:** Policy-RAG agent (§4) + `clause_id`-tagged corpus (§3.3) + deterministic window/category gates (§5) that make the "apply" step deterministic.

### Criterion 5 — Agent Orchestration Reliability
- **Definition:** % of requests where **all** required agents complete without error, tool calls run in the **correct sequence**, and the orchestrator returns a **structured output (verdict + justification)** in **< X seconds**.
- **"Correct sequence" defined:** the LangGraph DAG's dependency order is honored — the parallel agents (data/risk/text/policy) all complete *before* fusion, and fusion runs before the verdict/dossier. A request passes only if the DAG executed as specified with no agent erroring.
- **Exact computation:** per request log four booleans — all-agents-succeeded, order-honored, output-schema-valid, latency < X — and require **all four true** to count as reliable. Report the % and the latency distribution. Budget **X = 5s** (state the measured p95 in the report).
- **Earned by:** LangGraph orchestration (§5) + strict output-schema validation + retry/backoff on the LLM.

### Also emitted (deliverables, not the 5 criteria)
- **Plain Precision/Recall** table (deliverable 3).
- **Quantified false-positive-risk** number (deliverable 4).

**Build the harness against stub outputs on Day 1** so it's live the moment real agents land — you tune *toward these numbers*, not toward vague "accuracy." Every criterion has a design feature behind it; if any column above is ever blank, that's a red flag to fix before the pitch.

---

## 9. UI (reviewer dashboard — recommended, not graded)

A thin **Streamlit** layer, built last. Three views:
1. **Single-case adjudication** — paste/select a return, watch it flow through the agents, see the verdict + justification trail live.
2. **Escalation queue** — list of escalated cases each with its auto-generated dossier.
3. **Metrics panel** — the eval scoreboard rendered as a dashboard.

It imports the orchestrator as a module (or calls the FastAPI endpoint). Purpose is the *demo*, not the grade — keep it minimal.

---

## 9b. The Wow Moment — live "Risk Appetite" dial

A single slider on the dashboard, **Customer-Friendly ⟷ Fraud-Strict**, that re-adjudicates the entire eval set in real time as the judge drags it. Four figures recalculate live:

- **Fraud caught** (recall)
- **Good customers wrongly rejected** (false-positive rate)
- **Automation rate** (% auto-decided vs escalated)
- **₹ Net business value**

**Why it wins for this problem:** the case study's core is the false-positive trap vs false-negative trap (page 3). This hands the judge the exact risk knob a Flipkart risk manager would want, and makes the precision-recall frontier *visible and playable*. It proves business understanding, ties to the primary metric (Recall @ controlled FPR) and the >70% automation target, and reframes the system as a controllable business instrument rather than a black box.

**₹ Net business value formula** (computed per case at threshold `t`, summed):

```
+ order_value         for each fraud correctly Auto-Rejected      (loss prevented)
− order_value         for each fraud wrongly Auto-Approved        (fraud leaked)
− CUSTOMER_LTV        for each legit wrongly Auto-Rejected         (churn cost)
− REVIEW_COST         for each case Escalated                      (human handling)
```

Report the net vs a naive baseline (e.g., all-manual or all-approve) so the number has a reference point. Use sensible constants (e.g., `CUSTOMER_LTV ≈ ₹5,000`, `REVIEW_COST ≈ ₹50`) and state them on screen.

**Critical implementation note — precompute, don't re-run agents.** The slider must NOT re-invoke the LLM/RAG/agents on every drag. Run all agents **once** per eval case, cache each case's `risk_score` + intent flags + policy result to memory, and have the slider only re-apply the fusion **threshold** over those cached scores. That makes the dial instant and demo-safe. Re-running agents on slider move would be slow, rate-limited, and non-deterministic.

**Pitch line:** *"We didn't just automate the decision — we gave the business a dial to tune its own risk appetite, and here's the money it saves at each setting."*

---

## 10. Deployment plan

**LOCKED: Google Cloud Run (primary) · HF Streamlit Space (fallback).** Heavy compute is the LLM (external API) and XGBoost (trivial), so free CPU hosting is sufficient. Cloud Run is primary because Docker gives full control (it can host the FastAPI service *and* a Streamlit service — as separate services, see the architecture note below) and it stays in the Gemini/Google ecosystem. HF free tier supports the whole app *except* Docker — so on the fallback path the Streamlit dashboard deploys publicly and FastAPI runs locally (no graded loss).

| Target | Use | Free? | Notes |
|---|---|---|---|
| **Google Cloud Run** ← PRIMARY | FastAPI + Streamlit (see arch note below) | ✅ $300 credit/90 days + always-free 2M req/mo | Docker supported (unlike HF free); coherent with Gemini. Free-tier region: `us-central1`/`us-east1`/`us-west1`. |
| **HF Spaces (Streamlit SDK)** ← FALLBACK | Public Streamlit demo | ✅ CPU-Basic 2 vCPU / 16 GB | Fits Chroma + embeddings; built-in secrets for API key. Docker Spaces need PRO, so on this path FastAPI runs local. |
| **Streamlit Community Cloud** ← 2nd fallback | Public Streamlit demo | ✅ | Dead simple GitHub deploy; sleeps after 12 h idle, smaller RAM |
| **FastAPI** | "API-speed" deliverable | run **local**, or on Cloud Run | Local is fastest with no cold starts; Cloud Run if you want it public |

**Cloud Run architecture note (important — one service = one port):** Cloud Run routes to a single port per service, so you cannot expose FastAPI *and* Streamlit from one container. Two correct patterns:
- **(A) Two services** — deploy `api` (FastAPI) and `ui` (Streamlit) separately; the UI calls the API's public URL. Most idiomatic. **Requires CORS enabled on FastAPI** so the browser allows the cross-service call.
- **(B) One service (simpler, recommended for the demo)** — deploy only the Streamlit `ui`, which **imports the orchestrator as a Python module** (no HTTP hop, no CORS). Run FastAPI locally / as an optional 2nd service purely for the "API-speed" deliverable.

Pattern **B** is the lower-risk choice for a hackathon. Since the LLM is Gemini, Cloud Run keeps everything in the Google ecosystem. Use it if Phase 6 has the hour; otherwise ship the HF Streamlit Space (which is pattern B by nature).

Deploy steps (HF): push repo → create Streamlit Space → add `GEMINI_API_KEY` as a Space secret → `requirements.txt` auto-installs → live URL.
Deploy steps (Cloud Run): write `Dockerfile` → `gcloud run deploy` from source → set `GEMINI_API_KEY` via env var / Secret Manager → region `us-central1` → set `min-instances=1` during the demo to avoid cold start.

**Pre-deploy checklist:** (1) cold start — warm the URL / `min-instances=1` before presenting; (2) API key as a secret, `.env` git-ignored; (3) bundle the SQLite file + trained XGBoost artifact (`.pkl`) + **persisted Chroma index** (persist-dir committed) inside the image/repo — a fresh container has no local files; (4) pin `requirements.txt` (+ `Dockerfile` for Cloud Run); (5) if using the two-service pattern (A), **enable CORS on FastAPI**; (6) cache to survive Gemini rate limits under demo load; (7) container must listen on `$PORT` (Cloud Run sets it); (8) keep a **recorded backup demo** in case venue networking fails.

---

## 11. Repository structure

```
return-triage/
├── common/                        # shared, imported everywhere
│   ├── llm_client.py              # provider-agnostic Gemini/Groq/OpenAI + retry/backoff/key-rotation
│   ├── schemas.py                 # pydantic: shared-state object + §5.2 output contract
│   └── config.py                  # constants (thresholds, CUSTOMER_LTV, REVIEW_COST, X=5s)
├── data/
│   ├── generate_synthetic.py      # batched LLM + cache + backoff
│   ├── mock_orders.db             # SQLite (generated)
│   ├── llm_cache/                 # cached reason_text (git-ignored or committed)
│   └── policies/                  # scraped + mock policy docs (clause_id tagged)
├── agents/
│   ├── data_agent.py
│   ├── risk_agent.py              # XGBoost train + predict
│   ├── text_agent.py              # uses common/llm_client
│   ├── policy_rag_agent.py        # Chroma index + retrieve
│   └── image_check.py             # optional
├── orchestrator/
│   ├── graph.py                   # LangGraph flow
│   ├── fusion.py                  # fused_fraud_confidence + thresholds + verdict + justification + dossier
│   └── deterministic.py           # initial-validation gates
├── models/
│   └── risk_model.pkl             # trained XGBoost (committed/bundled)
├── chroma/                        # persisted Chroma index (committed/bundled)
├── api/main.py                    # FastAPI POST /adjudicate
├── ui/app.py                      # Streamlit dashboard + risk dial (§9b)
├── eval/
│   ├── harness.py                 # all 5 metrics + P/R + latency
│   ├── adversarial_set.json       # 50–100 hand-authored
│   └── escalation_set.json        # 30–50
├── tests/                         # §16 test suite
│   ├── conftest.py                # seeded test SQLite + mocked LLM fixtures
│   ├── test_agents.py
│   ├── test_fusion.py
│   ├── test_orchestrator.py
│   └── test_nfr.py                # latency, malformed-JSON, missing-image, rate-limit
├── outputs/                       # run_batch structured decisions land here
├── run_batch.py                   # batch runner → structured decisions
├── Dockerfile                     # Cloud Run (listens on $PORT)
├── Makefile                       # `make test` → pytest && python eval/harness.py
├── requirements.txt               # pinned
├── .env.example                   # GEMINI_API_KEY=...
├── .gitignore                     # .env, __pycache__, local artifacts
├── README.md                      # frozen decisions (Phase 0.4) + run instructions
├── DESIGN.md
└── PLAN.md
```

Note: `chroma/` and `models/risk_model.pkl` must be committed/bundled so a fresh Cloud Run deploy has them (no local build step at runtime).

---

## 12. Tech stack & environment

Python 3.11 · LangGraph (orchestration) · XGBoost/LightGBM (risk) · **Gemini (default LLM, Flash tier)** via a provider-agnostic wrapper (Groq/OpenAI = one-line swaps) · Chroma + sentence-transformers (RAG, local/free) · FastAPI (API) · Streamlit (UI) · SQLite (mock data). All open-source except the LLM API. Provider-agnostic LLM client with retry/backoff + optional key rotation so rate limits never break a run.

---

## 13. Build order (de-risking rule)

Build a **thin end-to-end skeleton with every agent stubbed** first (working pipeline by hour 3), then upgrade each stub to real in parallel, running the eval harness after every swap. Guarantees a demoable system at all times and surfaces integration bugs early.

---

## 14. Implementation challenges & risks

Known hard spots, ranked by likelihood of biting us, each with a mitigation:

1. **Synthetic-data circularity (highest risk).** If we generate data by rule and the system learns those exact rules, metrics look artificially perfect and a judge will call it out. *Mitigate:* add noise and overlapping distributions between classes, make Borderline genuinely ambiguous, and keep the 50–100 adversarial cases hand-authored so they break naive rule-matching.
2. **LLM non-determinism & malformed output.** The text agent may return invalid JSON or vary run-to-run, hurting the orchestration-reliability metric. *Mitigate:* JSON/structured-output mode, schema validation + auto-retry, low temperature, cache every result.
3. **Threshold tuning into 3 verdicts.** Mapping a continuous risk score + flags into Approve/Reject/Escalate is a 2-boundary tuning problem; the "escalate band" can collapse (nothing escalates) or explode (automation <70%). *Mitigate:* tune on the holdout with automation-rate >70% as an explicit constraint; the risk dial actually helps visualize this.
4. **RAG retrieving the wrong clause.** Small corpus + embedding retrieval can grab an irrelevant policy chunk, tanking the adherence metric. *Mitigate:* clean chunking with `clause_id` metadata, filter by category/window before ranking, consider hybrid keyword+vector.
5. **Grounded-justification plumbing.** Templating facts programmatically while the LLM writes prose is fiddly; the LLM may still restate a number wrong. *Mitigate:* inject feature values/citations post-hoc; the LLM only writes connective text, never the figures. Auto-check numbers against source in the harness.
6. **Latency & rate limits under load.** Multiple LLM calls × full eval set = slow and throttled. *Mitigate:* only the text agent needs the LLM; parallelize agents, cache, batch, prefer Groq for speed. (See the dial's precompute note.)
7. **Fusion logic getting hacky.** Encoding "current evidence overrides history" cleanly is non-trivial. *Mitigate:* keep the rules few and explicit with a documented precedence order.
8. **Image handling.** Detecting a deceptive camera angle is genuinely hard; don't over-promise. *Mitigate:* ship a lightweight flag/heuristic and treat it honestly in the failure-analysis note.
9. **4-person integration drift.** Mismatched interfaces and merge pain. *Mitigate:* freeze the shared-state schema (§4) and output contract (§5.2) before coding; stub-first skeleton.
10. **Time budget.** Six components + eval + UI + dial + deck in ~1–2 days is genuinely a lot. *Mitigate:* stub-first, cut image if needed, keep FastAPI local, protect eval + explainability + the dial as the non-negotiables.

---

## 15. Optional enhancements (LOW priority — non-graded, build only if time remains)

Nice-to-haves that don't affect the five graded criteria. Attempt only after the core + eval + dial are done (Phase 6 spare time). **Not required.**

### 15.1 Reviewer notification system — priority: LOW
- **Purpose:** ping the human reviewer when a case is escalated. Note: the **escalation queue in the dashboard already surfaces escalated cases with their dossier**, so this is largely redundant — treat as demo garnish, not a real advantage.
- **Build tiers (go only as far as time allows):**
  1. *In-dashboard alert* (default, ~free) — a badge/toast in the Streamlit escalation queue ("N new cases need review"). This alone reads as "notifications" in a demo.
  2. *Slack/Discord webhook* (~20 min) — on escalate, POST the dossier to a reviewer channel; shows a live ping in the room.
  3. *Email (SMTP)* — skip unless specifically requested; most setup, least flashy.
- **Recommendation:** rely on the escalation queue (tier 1). Add the Slack webhook only as a spare-time flourish. Do not plan around it.

### 15.2 UI polish — priority: LOW
- Beyond the minimal 3-view dashboard (§9): cleaner styling, agent-by-agent "reasoning replay" animation, nicer metric charts. Pure presentation value; do last.

---

## 16. Requirements & Testing

### 16.1 Sample input (one return request)
Every test feeds a record shaped like this (the optional image may be null):
```json
{
  "return_id": "R12345",
  "order": {"order_id": "O987", "sku": "ELEC-778", "category": "electronics",
            "order_value": 45000, "order_date": "2026-07-20",
            "delivery_scan_confirmed": true, "event_adjacent": false,
            "days_since_delivery": 12},
  "account": {"account_id": "A55", "account_age_days": 14, "total_orders": 3,
              "total_returns": 2, "return_to_order_ratio": 0.67, "prior_fraud_flags": 0},
  "reason_text": "item arrived broken",
  "image_ref": null
}
```

### 16.2 Functional requirements (FR) — what the system must do
- **FR1** Accept the input package (order + account + reason + optional image) and return exactly one verdict: Auto-Approve / Auto-Reject / Escalate.
- **FR2** Run the deterministic pre-check *before* agents; hard-reject out-of-window or non-returnable categories.
- **FR3** Each agent writes its output into shared state; no agent calls another directly.
- **FR4** Orchestrator returns the full structured JSON of §5.2 (verdict, risk_score, fused_fraud_confidence, policy_citation, justification_trail, escalation_dossier if escalated, latency_ms).
- **FR5** Justification trail cites only feature values that exist in the input record (no invented numbers).
- **FR6** Policy-RAG cites the correct clause_id **and** the verdict is consistent with it.
- **FR7** Conflicting / grey-band cases return Escalate **with** a populated dossier.
- **FR8** Batch runner processes N requests → one structured decision per request, no crash.

### 16.3 Technical / non-functional requirements (NFR)
- **NFR1** Latency < 5 s per request (report p95).
- **NFR2** Orchestration reliability: all agents complete, correct DAG order, schema-valid output (target high %).
- **NFR3** Every LLM call returns schema-valid JSON — enforced by validation + retry/backoff.
- **NFR4** Reproducibility: numeric generation seeded; LLM outputs cached so reruns are stable.
- **NFR5** Graceful degradation: missing image, missing field, or a failed agent must not crash the pipeline (fall back / escalate).
- **NFR6** Rate-limit resilience: a 429 triggers backoff+retry, not a failure.
- **NFR7** Metric targets: Recall @ ≤5% FPR maximized; automation rate > 70%.

### 16.4 Test matrix (what to test · input · expected · how)

| Level | Test | Input | Expected | How |
|---|---|---|---|---|
| **Unit** | Data Agent SQL | seeded account_id | exact known feature vector | `pytest` + seeded test SQLite |
| Unit | Risk Agent range/monotonicity | feature vectors | prob ∈ [0,1]; higher ratio → higher risk | pytest asserts |
| Unit | Text Agent — contradiction | "arrived broken" + delivery_scan=true | `contradiction=true` | pytest (cached/mock LLM) |
| Unit | Text Agent — sarcasm | "yeah it totally 'broke' lol" | `tone=sarcastic` | pytest (cached LLM) |
| Unit | Policy-RAG | electronics, day 12 | retrieves 10-day-window clause_id | pytest assert clause_id |
| Unit | Deterministic gate | out-of-window / non-returnable | hard-reject before agents | pytest |
| Unit | Fusion truth table | (high risk+violation+contradiction) / (low+clean+ok) / (conflicting) | Reject / Approve / Escalate | pytest parametrized |
| Unit | Justification grounding | any decision | every number in trail exists in input | automated string/number check |
| **Integration** | Orchestrator on known case | full sample record | correct verdict + schema-valid JSON | pytest + pydantic schema |
| Integration | Agent sequencing | any request | all agents ran before fusion | assert on run log |
| Integration | Escalation path | borderline record | verdict=Escalate + dossier non-empty | pytest |
| **End-to-end** | Batch run | holdout set | one valid decision per row, no crash | `run_batch.py` + schema check |
| **Eval** | 5 graded metrics | labeled holdout | Recall@≤5%FPR; automation >70% | eval harness §8 |
| Eval | Adversarial modes | 50–100 hand-authored | each mode handled per §7.1 (e.g., sarcasm not auto-approved) | harness assertions |
| Eval | Escalation quality | 30–50 borderline | system escalates ≥ target % | harness |
| **NFR** | Latency | batch | p95 < 5 s | timing in harness |
| NFR | Malformed LLM output | injected bad JSON | retry recovers, no crash | fault-injection test |
| NFR | Missing image / field | null image_ref | "no image" flag, pipeline completes | pytest |
| NFR | Rate limit | simulated 429 | backoff retries, succeeds | mock test |

### 16.5 How to run tests
- **Unit + integration:** `pytest` over `agents/`, `orchestrator/`; use a **seeded test SQLite** fixture and **cached/mocked LLM responses** so results are deterministic (never hit the live API in unit tests).
- **Schema validation:** a `pydantic` model of the §5.2 output contract validates every decision.
- **Metrics + adversarial:** the eval harness (`eval/harness.py`) is the integration test for the graded criteria — run it after every agent swap (per the build order).
- **CI-lite:** a single `make test` (or `pytest && python eval/harness.py`) that runs before each merge keeps the 4-person repo from drifting.
