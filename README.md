# UM Hackathon - RefundFlow

## Pitching Video

10-minute pitching and product demonstration video: PASTE_VIDEO_LINK_HERE

## Project Overview

RefundFlow is an AI-powered customer refund workflow prototype for the UM Hackathon. It uses Z.AI GLM as the central reasoning engine and a backend multi-agent workflow layer to handle refund intake, validation, order verification, manual review, audit logging, and prototype approval.

## Repository Structure

```text
UMHack/
|-- README.md
|-- customer_refund_backend.py
|-- customer_refund_app_realistic.html
|-- Project Requirement Document (PRD).pdf
|-- System Analysis Documentation (SAD).pdf
|-- QATD_AGENT.pdf
|-- Pitch Deck.pdf
`-- .gitignore
```

## How To Run

1. Start the backend from the repository root:

```bash
uvicorn customer_refund_backend:app --reload
```

2. Open the frontend file in a browser:

```text
customer_refund_app_realistic.html
```

3. Make sure the frontend backend URL points to:

```text
http://127.0.0.1:8000
```

## Submission Items

| Requirement | Location |
|---|---|
| PRD (Product) | `Project Requirement Document (PRD).pdf` |
| SAD (System) | `System Analysis Documentation (SAD).pdf` |
| QATD (Testing) | `QATD_AGENT.pdf` |
| Pitching Deck | `Pitch Deck.pdf` |
| 10-minute Pitching Video | First section of this README |
| Code Repository | This GitHub repository |

## Prototype Notes

- The backend uses an in-memory mock order database for verification.
- Prototype approval does not send real money or call a real payment gateway.
- Restarting the backend clears in-memory workflows and refund cases.
