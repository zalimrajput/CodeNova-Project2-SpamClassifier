# SMS Spam Message Classifier

An end-to-end SMS spam detector: a Next.js frontend talks to a FastAPI backend
that loads a pre-trained **CountVectorizer + Multinomial Naive Bayes** model from
`backend/model/spam_classifier.joblib` and classifies pasted messages as **spam**
or **ham**. The model was trained and evaluated in Google Colab on the UCI SMS
Spam Collection dataset and is used as-is at inference time — no retraining.

## Project layout

```
.
├── README.md
├── backend
│   ├── main.py                  FastAPI app + prediction endpoints
│   ├── requirements.txt         Python dependencies
│   └── model
│       └── spam_classifier.joblib   pre-trained model bundle
└── frontend
    ├── package.json
    ├── next.config.mjs
    ├── tsconfig.json
    ├── .gitignore
    └── app
        ├── globals.css          page styles
        ├── layout.tsx           root layout
        └── page.tsx             classify UI (input, result card, examples)
```

## Running it

Start the backend first.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000      # or: python main.py
```

`python main.py` defaults to port **8000**; override it with `APP_PORT`. Swagger
docs are at http://localhost:8000/docs.

Then the frontend:

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

The frontend calls the API at `http://localhost:8000` by default. Override with
`NEXT_PUBLIC_API_URL` at build/runtime (e.g.
`NEXT_PUBLIC_API_URL=http://localhost:8010 npm run dev`). Because this env var is
injected at build time, a rebuild is needed after changing it when using
`npm start`/production.

### Endpoints

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| `GET` | `/api/health` | — | `{status, model_loaded, model_name}` |
| `GET` | `/api/model` | — | `{model_name, positive_class, classes, test_metrics}` |
| `POST` | `/api/predict` | `{"message": "..."}` (≤10000 chars, non-empty) | `{"message","label","is_spam","confidence","probabilities":{"ham","spam"}}` |

`POST /api/predict` returns `422` when the message is empty (or whitespace-only).
Inference latency is dominated by Python model load on startup; per-prediction cost
is negligible for a single message.

## Notes

- The bundle was pickled with scikit-learn 1.6.1; loading with a newer scikit-learn
  prints an `InconsistentVersionWarning` at startup (silenced in `main.py`). It
  predicts correctly; pin `scikit-learn==1.6.1` for exact parity.
