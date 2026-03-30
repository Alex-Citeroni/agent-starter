"""
Agents Society — Challenge Agent (con LLM)

Usa GitHub Models (gpt-4o-mini, gratuito) per decidere cosa dire a Renato.
Il GITHUB_TOKEN è disponibile automaticamente in GitHub Actions — non serve
configurare nulla.

Uso locale:
  AGENTS_SOCIETY_API_KEY=<chiave> PARTICIPANT_ID=<id> GITHUB_TOKEN=<token> python agent.py
"""

import os
import re
import sys
import json
import time
import requests

# ── Configurazione ─────────────────────────────────────────────
BASE_URL     = os.environ.get("AGENTS_SOCIETY_URL", "https://agentssociety.ai")
API_KEY      = os.environ.get("AGENTS_SOCIETY_API_KEY", "")
PARTICIPANT  = os.environ.get("PARTICIPANT_ID", "")
CHALLENGE    = os.environ.get("CHALLENGE_SLUG", "sales")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

if not API_KEY or not PARTICIPANT:
    print("❌ AGENTS_SOCIETY_API_KEY e PARTICIPANT_ID sono obbligatori.")
    sys.exit(1)

if not GITHUB_TOKEN:
    print("❌ GITHUB_TOKEN è obbligatorio per usare GitHub Models.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

# ── System prompt dell'agente venditore ────────────────────────
AGENT_SYSTEM_PROMPT = """Sei Marco, un agente commerciale che vuole vendere "GestioCarni Pro"
a Renato, un macellaio tradizionale di Milano.

GestioCarni Pro è un software gestionale per macellerie che:
- Automatizza gli ordini ai fornitori
- Monitora le scorte in tempo reale
- Riduce gli sprechi del 30%
- Costa 89€/mese con 30 giorni di prova gratuita

Il tuo obiettivo è vendere seguendo le fasi: qualificazione → scoperta bisogni →
presentazione soluzione → gestione obiezioni → chiusura.

Regole:
- Risposte brevi e naturali (max 3 frasi)
- Adatta il tono in base alle risposte di Renato
- Se Renato è scettico, usa prove sociali (altri macellai milanesi)
- Cerca di chiudere con una prova gratuita o un appuntamento
- Scrivi solo in italiano"""


def call_llm(conversation_history: list[dict]) -> str:
    """Chiama GitHub Models (gpt-4o-mini) per generare la prossima risposta."""
    response = requests.post(
        "https://models.inference.ai.azure.com/chat/completions",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *conversation_history,
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def post(endpoint: str, payload: dict) -> dict:
    url = f"{BASE_URL}/api/v1/challenges/{CHALLENGE}/{endpoint}"
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_scores(text: str) -> dict | None:
    """Estrae i punteggi dal formato VALUTAZIONE FINALE."""
    mapping = {
        "qualification":      r"Qualification:\s*(\d+)/10",
        "outreach":           r"Outreach:\s*(\d+)/10",
        "discovery":          r"Discovery:\s*(\d+)/10",
        "solution":           r"Solution:\s*(\d+)/10",
        "objection_handling": r"Objection Handling:\s*(\d+)/10",
        "closing":            r"Closing:\s*(\d+)/10",
    }
    scores = {}
    for key, pattern in mapping.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            scores[key] = int(m.group(1))
    return scores if len(scores) == 6 else None


def run():
    print(f"🚀 Avvio challenge '{CHALLENGE}' | participant_id: {PARTICIPANT}")
    print(f"🤖 Modello LLM: gpt-4o-mini (GitHub Models)\n")

    # 1. Avvia la sessione
    start = post("start", {"participant_id": PARTICIPANT})
    if not start.get("success"):
        print(f"❌ Start fallito: {start}")
        sys.exit(1)

    session_id    = start["data"]["session_id"]
    first_message = start["data"]["first_message"]
    print(f"🤌 Renato: {first_message}\n")

    # 2. Loop conversazione — l'LLM decide cosa dire ad ogni turno
    conversation_history = [
        {"role": "assistant", "content": first_message},
    ]
    score_breakdown = None

    for turn in range(20):  # max 20 turni
        # L'LLM genera la prossima mossa dell'agente
        my_message = call_llm(conversation_history)

        print(f"🤖 Agente: {my_message}\n")
        time.sleep(1)

        result = post("message", {
            "session_id": session_id,
            "message":    my_message,
        })

        if not result.get("success"):
            print(f"❌ Messaggio fallito al turno {turn}: {result}")
            sys.exit(1)

        reply    = result["data"]["reply"]
        is_ended = result["data"]["conversation_ended"]

        print(f"🤌 Renato: {reply}\n")
        print("-" * 60)

        # Aggiorna la storia della conversazione per il prossimo turno
        conversation_history.append({"role": "assistant", "content": my_message})
        conversation_history.append({"role": "user", "content": reply})

        if is_ended:
            print("\n📊 Conversazione conclusa — parsing punteggi...")
            score_breakdown = parse_scores(reply)
            break

        time.sleep(2)

    # 3. Invia i punteggi
    if not score_breakdown:
        print("⚠️  Punteggi non trovati. Inserisco punteggi nulli.")
        score_breakdown = {k: 0 for k in ["qualification", "outreach", "discovery",
                                           "solution", "objection_handling", "closing"]}

    total = sum(score_breakdown.values())
    print(f"\n🏆 Punteggio finale: {total}/60")
    print(json.dumps(score_breakdown, indent=2))

    submit = post("submit", {
        "participant_id":  PARTICIPANT,
        "score_breakdown": score_breakdown,
    })

    if submit.get("success"):
        print(f"\n✅ Punteggio salvato! Total: {submit['data']['total_score']}/60")
    else:
        print(f"❌ Submit fallito: {submit}")
        sys.exit(1)


if __name__ == "__main__":
    run()
