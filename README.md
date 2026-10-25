# ZentroMart VoiceOps 2026 — Grounded Conversational & Voice Support Agent

**Course:** LLM & Generative AI Course · Consulting & Analytics Club, IIT Guwahati × Gnani.ai  
**Public Repository:** `https://github.com/<your-username>/voiceops-2026-zentromart` *(replace with your public GitHub URL before final upload)*

---

## 1. Overview & Architecture

This repository implements ZentroMart's end-to-end multilingual AI customer support agent across all four capstone stages:

1. **Step 1 — Prompt-Only LLM Baseline (`src/step1_baseline.py`):** Calls an LLM with a role-and-limits system prompt (`STY-1` to `STY-5`) without document grounding to establish a baseline on the 52 practice questions (`30.77%` accuracy, `16 / 52`).
2. **Step 2 — Grounded RAG & Policy Precedence (`src/retriever.py`, `src/date_utils.py`):** Indexes all 21 Markdown knowledge-base documents (`kb/*.md`) by `[SECTION-ID]`, `data/products.csv` (`70` SKUs), `data/pincodes.csv` (`36` pincodes), and `data/holidays.csv` (`4` warehouse holidays). Enforces strict policy precedence (`KB-00.1`: current `policy` > `agent_guide` > `help` > `faq` > `marketing` > `superseded`), achieving `100.0%` answer accuracy (`52 / 52`) and `100.0%` exact `gold_evidence` retrieval recall.
3. **Step 3 — Multi-Turn Conversational State & Identity-Verified Tools (`src/agent.py`):** Tracks conversation history, coreference, language switching (`en`, `hi`, `hi-en`), and topic switches. Integrates `lookup_order(order_id, verification_value)` with strict caller verification (last 4 digits of registered phone or 6-digit delivery pincode, maximum 1 retry on failure per `IDV-3`) and `create_ticket(summary, order_id, reason, intent)` for mandatory human handovers (`ESC-1`, `DWM-3`), achieving `100.0%` turn accuracy across all 23 conversations (`96 / 96` turns).
4. **Step 4 — Gnani Voice Layer (`src/voice.py`, `src/asr_normalizer.py`):** Wraps the Step 3 agent with Gnani's official `gnani-vachana` SDK (`GnaniSTTClient` and `GnaniTTSClient` using `model="timbre-v2.5"` and `voice="Kaveri"`), normalizes spoken ASR numbers/SKUs (`src/asr_normalizer.py`), formats replies for the ear (dates with month and year, 8-digit order IDs grouped as `XXXX XXXX`), and recovers cleanly from empty or unclear audio (`100.0%` accuracy on `asr_text`, `52 / 52`).

---

## 2. Setup Instructions

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (including official Gnani SDK)
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set GNANI_API_KEY=<your_gnani_api_key_from_app.gnani.ai/voice>
# Optionally set ZENTROMART_DATASET_DIR=/path/to/Capstone_Project_Dataset_FTTT_2026
```

---

## 3. Exact Run Commands

### A. Organizer Batch Audio Evaluation (One Command)
Takes a folder of customer audio files (`.wav`, `.mp3`, `.flac`, `.ogg`), transcribes each via `GnaniSTTClient`, runs the grounded `ZentroMartAgent`, synthesizes speech via `GnaniTTSClient` (`model="timbre-v2.5"`, `voice="Kaveri"`), and saves text replies, speech `.wav` files, and retrieved `[SECTION-ID]` chunks into `sample_outputs/`:

```bash
python run.py /path/to/customer_audio_folder --output-dir sample_outputs --dataset-dir /path/to/Capstone_Project_Dataset_FTTT_2026
```
*(Or via shell wrapper: `./run.sh /path/to/customer_audio_folder`)*

For each `<clip>.wav` in the input folder, `run.py` writes:
- `sample_outputs/<clip>_reply.txt` — Agent's text reply formatted for speech
- `sample_outputs/<clip>_reply.wav` — Synthesized speech audio from Gnani TTS
- `sample_outputs/<clip>_sources.json` — JSON record with `transcript`, `reply_text`, `decision`, `retrieved_chunks`, and `tool_calls`
- `sample_outputs/batch_results.json` & `sample_outputs/batch_results.csv` — Aggregated batch run table

### B. Practice Benchmark Evaluation (Steps 1–4)
Runs the complete benchmark across `practice/questions.csv` (`52` questions, both text and `asr_text`) and `practice/conversations.csv` (`23` conversations, `96` turns):

```bash
python run.py --evaluate --dataset-dir /path/to/Capstone_Project_Dataset_FTTT_2026 --output-dir sample_outputs
```

### C. Interactive Web Support Console
Launches a browser-based interface at `http://127.0.0.1:8000` displaying live replies, retrieved `[SECTION-ID]` chunks, and executed tool calls:

```bash
python run.py --web --port 8000
```

---

## 4. Repository Structure

```text
team_name_voiceops/
├── README.md
├── .env.example
├── requirements.txt
├── run.py
├── run.sh
├── report.pdf
├── demo.mp4
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── date_utils.py
│   ├── asr_normalizer.py
│   ├── retriever.py
│   ├── llm_client.py
│   ├── step1_baseline.py
│   ├── agent.py
│   ├── voice.py
│   ├── evaluator.py
│   └── web_app.py
└── sample_outputs/
    ├── evaluation_summary.json
    ├── step1_vs_step2_questions.json
    └── step3_conversations_outputs.json
```

### D. Launch Streamlit Application
```bash
streamlit run app.py
```
