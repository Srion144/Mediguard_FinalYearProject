# Track B — Weeks 1, 2 & 3, fully explained

Your roadmap splits Weeks 1–3 of Track B into three stages: scaffold both
apps (Week 1), build the frontend against fake/mocked data (Week 2), build
the backend against a fake `/predict` endpoint and then swap in the real
model at Hand-off 1 (Week 3). Below is what was built, why it's structured
this way, and exactly how to run it.

**One thing worth knowing up front:** because Track A's real model
(`predict.py` + the trained `.joblib` file) already existed when I built
this, I connected the backend to the *real* model directly, instead of
building a temporary fake one first. The "mock" step still exists — it's
just on the frontend side now (see Week 2 below) — because that's the part
that's genuinely useful even when nothing's mocked on the backend: it lets
you work on the UI without needing the backend server running at all.

---

## Part 1 — The Backend (`backend/`)

### What FastAPI is and why we're using it

FastAPI is a Python web framework: it takes HTTP requests (like a POST
request with a patient's data as JSON) and runs your Python code in
response. We're using it because your roadmap specifies it, and because it
does one very convenient thing for a student project: it auto-generates
interactive API documentation you can click through in a browser, with zero
extra work.

### Folder structure

```
backend/
├── app/
│   ├── main.py              <- creates the FastAPI app, wires everything together
│   ├── schemas.py            <- defines the exact shape of requests/responses
│   ├── routers/
│   │   └── predict.py         <- the actual POST /predict endpoint
│   └── ml/
│       ├── predict_core.py     <- Track A's predict() logic, with one small fix
│       └── multioutput_xgb.joblib   <- a copy of the trained model
├── tests/
│   └── test_predict.py        <- automated tests (see below)
└── requirements.txt
```

### `predict_core.py` — Track A's model, one bug fixed

This file is your `src/predict.py`, copied in almost unchanged. There is
exactly **one** intentional difference: how it finds the model file.

The original loads the model with a plain string:
```python
_model = joblib.load("models/multioutput_xgb.joblib")
```
That path is relative to whatever folder you happen to be sitting in when
you run the command. It works fine when you always run things from the
repo root — but the moment someone runs `uvicorn` from inside `backend/`
(which is the normal way to start a FastAPI app), Python looks for
`backend/models/multioutput_xgb.joblib`, which doesn't exist, and it
crashes. This is one of the most common real-world Python bugs — code that
"works on my machine" because of which folder you happened to be in.

The fix:
```python
MODEL_PATH = Path(__file__).resolve().parent / "multioutput_xgb.joblib"
```
This says "look for the model file next to *this Python file itself*, no
matter where the terminal's current folder is." That's why the model file
is copied into `backend/app/ml/` right alongside `predict_core.py`, instead
of being loaded from the original `models/` folder.

### `schemas.py` — a deliberate choice worth understanding

FastAPI uses a library called Pydantic to define what a valid request looks
like. The obvious thing to do would be to tell Pydantic "age must be
between 1 and 120" and let it reject anything outside that range
automatically. **I didn't do that, on purpose.**

Here's why: your `INPUT_OUTPUT_SPEC.md` already defines exactly what should
happen on bad input — a normal (HTTP 200) response that looks like
`{"error": true, "messages": [...]}`, so your frontend can show it as a
friendly form validation message. If Pydantic rejected the bad input
first, the person would instead get FastAPI's generic, differently-shaped
422 error — breaking the contract Track A already carefully designed and
documented. So `schemas.py` only checks "is this actually a number," and
lets `predict_core.py`'s own `validate_input()` — which Track A already
wrote — be the single source of truth for range checking. One rulebook,
not two that could disagree.

### `routers/predict.py` and `main.py`

`predict.py` is intentionally almost empty — it just takes the request,
turns it into a plain dict, and hands it to `predict()`. All the real logic
lives in one place (`predict_core.py`), not scattered across the API layer.

`main.py` creates the app, and adds one important thing: **CORS**
(Cross-Origin Resource Sharing) middleware. Without this, a browser will
*refuse* to let your React app (running on `localhost:5173`) call your
API (running on `localhost:8000`) — browsers block requests between
different ports by default, for security. The CORS middleware explicitly
tells the browser "requests from `localhost:5173` are allowed."

### `tests/test_predict.py`

Three automated checks: the health-check route responds, a valid patient
gets a real prediction back, and an invalid patient (age 999) gets the
documented error shape instead of a server crash. I ran these myself
before handing this to you — see "What I actually tested" below.

### How to run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** in a browser — FastAPI builds you
a full interactive test page automatically. You can expand `POST /predict`,
click "Try it out," type in patient values, and see the real prediction
come back, without writing any frontend code at all.

To run the automated tests:
```bash
cd backend
pytest
```

---

## Part 2 — The Frontend (`frontend/`)

### What React + Vite is and why we're using it

React is a JavaScript library for building UIs out of reusable pieces
("components") that each manage their own little bit of the page. Vite is
the tool that runs your React code in a browser during development, and
bundles it into fast, deployable files when you're done. Your roadmap
specifies both.

### Folder structure

```
frontend/
├── src/
│   ├── App.jsx                    <- the page itself: holds state, wires everything together
│   ├── index.css                   <- design tokens (colors, fonts) + Tailwind
│   ├── components/
│   │   ├── PatientForm.jsx           <- the 5-field intake form
│   │   ├── ResultsPanel.jsx           <- shows empty / error / results states
│   │   ├── RiskGauge.jsx                <- the semicircular risk dial (hand-drawn SVG)
│   │   └── DriverBars.jsx                <- the "what's driving this" mini bar chart
│   ├── api/
│   │   └── predictApi.js                  <- calls either the mock or the real backend
│   └── mocks/
│       └── mockPrediction.js                <- Week 2's fake data
├── .env                                       <- controls mock vs. real backend
└── package.json
```

### Design choices, briefly

This is a clinical screening tool, so the visual direction is deliberately
calm rather than flashy: a cool, sage-tinted off-white background, a
medical-teal accent color, a serif display face (Fraunces) for headings
that adds a bit of warmth without looking corporate, and a monospace face
(IBM Plex Mono) specifically for numbers — risk percentages, lab values —
so they read as measured data rather than marketing copy. The risk gauge
itself is hand-drawn with SVG math (not a chart library default) so it has
its own identity: a filled semicircular arc, colored by risk band, with the
percentage in the center.

### Week 2's mock, and why it's still here even though the real backend works

`src/mocks/mockPrediction.js` returns fake numbers shaped **exactly** like
`INPUT_OUTPUT_SPEC.md`'s real output. `src/api/predictApi.js` is the one
place that decides whether to use that mock or call the real backend —
controlled by a single line in `.env`:

```
VITE_USE_MOCK=false   # currently set to false: calls the real backend
```

Every component (`PatientForm`, `ResultsPanel`, `RiskGauge`, `DriverBars`)
only ever talks to `getPrediction()` from `predictApi.js` — they have no
idea whether the data came from the mock or the real model. This is
exactly the kind of decoupling your roadmap's whole Track A/Track B split
is built on, just applied one level down. Set `VITE_USE_MOCK=true` any
time you want to work on the UI without needing the backend server
running at all (e.g. while riding a bus with no way to run Python).

### How to run the frontend

```bash
cd frontend
npm install
npm run dev
```

Then open the URL it prints (normally **http://localhost:5173**). Make
sure the backend (Part 1) is also running in a separate terminal first, or
set `VITE_USE_MOCK=true` in `.env` to test the UI without it.

---

## Part 3 — Hand-off 1, actually completed

Your roadmap describes Hand-off 1 as "about 15 minutes of work if the
format was agreed on clearly." Since the format (`INPUT_OUTPUT_SPEC.md`)
was already written precisely, that promise held up: the frontend calls
the backend, the backend calls Track A's real model, and the response
flows all the way back to the risk gauges and driver charts, with zero
translation code anywhere in between.

## What I actually tested (not just wrote)

I don't want to hand you code I merely *believe* works — here's what I
actually ran, in this order, before giving it to you:

1. **Backend automated tests** (`pytest`) — all 3 passed.
2. **Backend live server** — started `uvicorn` for real and hit it with
   `curl`, for a valid patient, an invalid patient, and the root health
   check. All three returned exactly what the spec says they should.
3. **Frontend build** (`npm run build`) — compiles cleanly with no errors
   or warnings.
4. **Frontend → backend call, simulated exactly** — I ran the *same* fetch
   call `predictApi.js` makes (same URL, same headers, same JSON body)
   against the live backend and printed the response, to confirm the shape
   matches what `RiskGauge` and `DriverBars` expect.
5. **CORS preflight** — checked that a browser request from
   `localhost:5173` (Vite's dev port) is actually allowed by the backend's
   CORS settings, since this is a common "works in curl, fails in the
   browser" trap.

The one thing I could *not* test from here is the actual rendered page in
a browser (this environment doesn't have one) — so when you run
`npm run dev` for the first time, look the page over for anything visually
off, and let me know if anything doesn't look right so I can fix it.

## Next steps

- Week 4: Track A work continues (LangGraph agents) — see the earlier
  `WEEK4_NOTES.md` I gave you, which is unaffected by any of this.
- Before your demo: swap the Cholesterol input from a dropdown to
  accepting a real mg/dL number if you want it to feel more like a lab
  report (the conversion rule is already documented in
  `INPUT_OUTPUT_SPEC.md`: <200→Normal, 200–239→Above normal, ≥240→Well
  above normal).
- Voice interface and bilingual (Hindi/English) support are still ahead of
  you per the roadmap (Track B, Week 4) — this frontend doesn't touch
  either yet.
