"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Prediction = {
  message: string;
  label: string;
  is_spam: boolean;
  confidence: number;
  probabilities: Record<string, number>;
};

const EXAMPLES = [
  "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
  "Hey, are we still meeting for lunch tomorrow at noon?",
  "URGENT: Your bank account has been locked. Verify your PIN at http://bit.ly/xyz",
  "Can you pick up some milk on your way home? Thanks!",
];

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function Home() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  async function classify() {
    const text = message.trim();
    if (!text || loading) return;

    setLoading(true);
    setResult(null);
    setError(null);
    const started = Date.now();

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Request failed (${res.status})`);
      }
      const data: Prediction = await res.json();
      const elapsed = Date.now() - started;
      const remaining = Math.max(0, 350 - elapsed);
      timerRef.current = setTimeout(() => {
        setResult(data);
        setLoading(false);
      }, remaining);
    } catch (err) {
      setError(
        err instanceof Error && err.message === "Failed to fetch"
          ? `Could not reach the backend at ${API_URL}. Make sure it is running (uvicorn main:app --port 8000).`
          : err instanceof Error
            ? err.message
            : "Something went wrong. Please try again.",
      );
      setLoading(false);
    }
  }

  function reset() {
    setMessage("");
    setResult(null);
    setError(null);
  }

  const hamProb = result ? (result.probabilities.ham ?? 0) : 0;
  const spamProb = result ? (result.probabilities.spam ?? 0) : 0;

  return (
    <main className="container">
      <header className="header">
        <h1>
          📱 SMS <span>Spam</span> Classifier
        </h1>
        <p className="subtitle">
          Paste an unseen SMS message and the pre-trained model will tell you
          whether it is <strong>spam</strong> or <strong>ham</strong>.
        </p>
      </header>

      <section className="card">
        <label className="label" htmlFor="message">
          Message
        </label>
        <textarea
          id="message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classify();
          }}
          placeholder="Type or paste an SMS message here…"
          rows={5}
          maxLength={10000}
        />

        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              className="chip"
              onClick={() => {
                setMessage(ex);
                setResult(null);
                setError(null);
              }}
            >
              {ex.length > 45 ? `${ex.slice(0, 45)}…` : ex}
            </button>
          ))}
        </div>

        <div className="actions">
          <button
            type="button"
            className="btn primary"
            onClick={classify}
            disabled={loading || !message.trim()}
          >
            {loading ? "Classifying…" : "Classify message"}
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={reset}
            disabled={loading || (!message && !result && !error)}
          >
            Reset
          </button>
        </div>
      </section>

      {loading && (
        <section className="card result" aria-live="polite">
          <div className="spinner" />
          <p>Analyzing message…</p>
        </section>
      )}

      {error && (
        <section className="card error" role="alert">
          <strong>Something went wrong</strong>
          <p>{error}</p>
        </section>
      )}

      {result && !loading && (
        <section className={`card result ${result.is_spam ? "spam" : "ham"}`} aria-live="polite">
          <div className="badge">{result.is_spam ? "🚫 SPAM" : "✅ HAM"}</div>
          <p className="verdict">
            This message is classified as{" "}
            <strong>{result.label.toUpperCase()}</strong> with{" "}
            <strong>{pct(result.confidence)}</strong> confidence.
          </p>

          <div className="bars">
            <div className="bar-row">
              <span className="bar-label">ham</span>
              <div className="track">
                <div className="fill ham-fill" style={{ width: pct(hamProb) }} />
              </div>
              <span className="bar-value">{pct(hamProb)}</span>
            </div>
            <div className="bar-row">
              <span className="bar-label">spam</span>
              <div className="track">
                <div className="fill spam-fill" style={{ width: pct(spamProb) }} />
              </div>
              <span className="bar-value">{pct(spamProb)}</span>
            </div>
          </div>

          <p className="echo">&ldquo;{result.message}&rdquo;</p>
        </section>
      )}

    </main>
  );
}
