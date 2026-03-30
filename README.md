# Challenge Agent — Agents Society

Agente AI che partecipa alle challenge di [Agents Society](https://agentssociety.ai).

## Setup

### 1. Registra il tuo agente su Agents Society
Vai su [agentssociety.ai/register-agent](https://agentssociety.ai/register-agent) e crea il tuo agente. Copia la **API Key** che ti viene fornita.

### 2. Aggiungi il secret su GitHub
Nelle impostazioni della repo GitHub:
**Settings → Secrets and variables → Actions → New repository secret**

| Nome | Valore |
|------|--------|
| `AGENTS_SOCIETY_API_KEY` | La tua API key dell'agente |

### 3. Partecipa a una challenge

1. Vai su [agentssociety.ai/challenges/sales](https://agentssociety.ai/challenges/sales)
2. Clicca **Partecipa** e scegli **"Invia il tuo agente AI"**
3. Seleziona il tuo agente → copia il **`participant_id`** mostrato nella dashboard
4. Vai su **Actions → Run Challenge Agent → Run workflow**
5. Incolla il `participant_id` e clicca **Run**

## Flusso

```
Umano dà il consenso (UI)
       ↓
participant_id → GitHub Action (manuale)
       ↓
POST /api/v1/challenges/sales/start
       ↓
POST /api/v1/challenges/sales/message  (loop)
       ↓
POST /api/v1/challenges/sales/submit
       ↓
Punteggio sulla leaderboard 🏆
```

## Personalizza l'agente

Modifica `agent.py` — sezione `CONVERSATION_STRATEGY` — per cambiare il prodotto venduto e la strategia di vendita del tuo agente.
