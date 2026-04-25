# UM Hackathon - RefundFlow

RefundFlow is an AI-powered customer refund workflow prototype for the UM Hackathon. It uses Z.AI GLM as the central reasoning engine and a backend multi-agent workflow layer to handle refund intake, validation, order verification, manual review, audit logging, and prototype approval.

## Repository Structure

```text
UMHack/
|-- src/
|   |-- customer_refund_backend.py
|   |-- customer_refund_app_realistic.html
|   `-- __init__.py
|-- docs/
|   |-- Project Requirement Document (PRD).pdf
|   |-- System Analysis Documentation (SAD).pdf
|   `-- QATD_AGENT.pdf
|-- pitch/
|   |-- Pitch Deck.pdf
|   `-- README.md
|-- README.md
`-- .gitignore
```

## How To Run

1. Start the backend from the repository root:

```bash
uvicorn src.customer_refund_backend:app --reload
```

2. Open the frontend file in a browser:

```text
src/customer_refund_app_realistic.html
```

3. Make sure the frontend backend URL points to:

```text
http://127.0.0.1:8000
```

## Submission Items

| Requirement | Location |
|---|---|
| PRD (Product) | `docs/Project Requirement Document (PRD).pdf` |
| SAD (System) | `docs/System Analysis Documentation (SAD).pdf` |
| QATD (Testing) | `docs/QATD_AGENT.pdf` |
| Pitching Deck | `pitch/Pitch Deck.pdf` |
| 10-minute Pitching Video | Add video link to `pitch/` |
| Code Repository | This GitHub repository |

## Prototype Notes

- The backend uses an in-memory mock order database for verification.
- Prototype approval does not send real money or call a real payment gateway.
- Restarting the backend clears in-memory workflows and refund cases.
