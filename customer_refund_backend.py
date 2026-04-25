from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import os
import json
import requests
import re

load_dotenv()

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_API_URL = os.getenv("ZAI_API_URL", "https://api.z.ai/api/paas/v4/chat/completions")
ZAI_MODEL = os.getenv("ZAI_MODEL", "glm-4.5")

app = FastAPI(
    title="Multi-Agent Customer Refund System Backend",
    description="Hackathon backend using Z.AI GLM plus specialist refund agents for task dispatching",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory prototype databases
workflow_db: Dict[str, Dict[str, Any]] = {}
refund_case_db: Dict[str, Dict[str, Any]] = {}
mock_order_db: Dict[str, Dict[str, Any]] = {
    "ORD12345": {
        "order_id": "ORD12345",
        "customer_name": "Jamie Lee",
        "customer_contact": "customer@email.com",
        "product_or_service": "Wireless Mouse",
        "payment_status": "paid",
        "delivery_status": "delivered",
        "refund_eligible": True,
        "refund_window_days": 30,
        "amount": "89.90",
        "currency": "MYR",
        "risk_level": "low"
    },
    "ORD99881": {
        "order_id": "ORD99881",
        "customer_name": "Alex Tan",
        "customer_contact": "alex@example.com",
        "product_or_service": "Bluetooth Headphones",
        "payment_status": "paid",
        "delivery_status": "delivered_damaged",
        "refund_eligible": True,
        "refund_window_days": 14,
        "amount": "129.90",
        "currency": "MYR",
        "risk_level": "medium"
    },
    "ORD00000": {
        "order_id": "ORD00000",
        "customer_name": "Test User",
        "customer_contact": "blocked@example.com",
        "product_or_service": "Final Sale Voucher",
        "payment_status": "paid",
        "delivery_status": "delivered",
        "refund_eligible": False,
        "refund_window_days": 0,
        "amount": "49.90",
        "currency": "MYR",
        "risk_level": "low"
    },
    "ORD77777": {
        "order_id": "ORD77777",
        "customer_name": "Morgan Lim",
        "customer_contact": "morgan@example.com",
        "product_or_service": "Premium Laptop",
        "payment_status": "paid",
        "delivery_status": "delivered",
        "refund_eligible": True,
        "refund_window_days": 7,
        "amount": "4999.00",
        "currency": "MYR",
        "risk_level": "high"
    }
}


class WorkflowRequest(BaseModel):
    user_input: str
    user_type: Optional[str] = "customer"


class FollowUpRequest(BaseModel):
    workflow_id: str
    answer: str


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    intent: Optional[str]
    current_stage: str
    missing_information: List[str]
    next_action: str
    refund_case_id: Optional[str] = None
    assigned_agents: List[str] = Field(default_factory=list)
    agent_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    agent_task_summary: Dict[str, Any] = Field(default_factory=dict)
    mapped_action: Dict[str, Any]
    structured_output: Dict[str, Any]
    created_at: str
    updated_at: str


def build_system_prompt() -> str:
    return """
You are the coordinator for a MULTI-AGENT CUSTOMER REFUND SYSTEM.

Your job:
1. Understand messy customer refund messages.
2. Identify the refund-related intent.
3. Decide the refund workflow stage.
4. Detect missing information needed to process a refund.
5. Choose ONE allowed next_action.
6. Generate structured actionable output that can be assigned to specialist backend agents.

You MUST return valid JSON only.

Allowed next_action values:
- ask_follow_up_questions
- generate_refund_checklist
- create_refund_case
- validate_refund_request
- calculate_refund_estimate
- approve_refund_prototype
- reject_refund_request
- manual_review_required

Common refund intents:
- request_refund
- check_refund_status
- return_item
- cancel_order_and_refund
- damaged_or_defective_item
- wrong_item_received
- duplicate_payment
- subscription_cancellation_refund
- refund_complaint
- unclear_request

Refund workflow stages:
- intake
- collecting_information
- validating_order_details
- checking_refund_eligibility
- calculating_refund
- ready_to_process_refund
- refund_case_created
- approved_prototype
- rejected
- manual_review
- failed

The JSON format must be:

{
  "intent": "string",
  "confidence": 0.0,
  "workflow_stage": "string",
  "missing_information": ["item1", "item2"],
  "next_action": "one_allowed_action_only",
  "structured_output": {
    "summary": "string",
    "refund_details": {
      "order_id": "string_or_unknown",
      "customer_name": "string_or_unknown",
      "customer_contact": "string_or_unknown",
      "product_or_service": "string_or_unknown",
      "refund_reason": "string_or_unknown",
      "purchase_date": "string_or_unknown",
      "payment_method": "string_or_unknown",
      "refund_amount": "string_or_unknown",
      "currency": "string_or_unknown"
    },
    "eligibility_assessment": {
      "is_likely_eligible": true,
      "reason": "string",
      "policy_flags": ["flag1", "flag2"]
    },
    "tasks": ["task1", "task2"],
    "agent_tasks": [
      {
        "agent": "intake_agent",
        "task": "task description",
        "priority": "low_or_normal_or_high",
        "status": "pending"
      }
    ],
    "questions_to_user": ["question1", "question2"],
    "recommended_tools_or_apis": ["tool1", "tool2"],
    "final_message": "string"
  }
}

Rules:
- Return JSON only. No markdown.
- Do not invent order details, dates, payment information, or refund amount.
- Do not say a real refund has been completed. This is prototype mode only.
- Do not approve a refund if required information is missing.
- Do not repeat the same question if the user already answered it.
- Required information for most refund requests: order_id, refund_reason, product_or_service, and customer_contact.
- If information is missing, use next_action = ask_follow_up_questions.
- If the customer wants steps to request a refund, use next_action = generate_refund_checklist.
- If the customer clearly wants to start a refund case and enough basic information exists, use next_action = create_refund_case.
- If order/payment/refund eligibility needs checking, use next_action = validate_refund_request.
- If the user asks how much will be refunded and amount details exist, use next_action = calculate_refund_estimate.
- If all required information is collected and the request appears eligible, use next_action = approve_refund_prototype and workflow_stage = ready_to_process_refund.
- If the request is clearly outside refund policy, fraudulent, abusive, impossible, or unsafe, use next_action = manual_review_required or reject_refund_request.
- For damaged, defective, wrong item, missing item, duplicate payment, or high-value refund cases, prefer manual_review_required unless the request is simple and low-risk.
"""


def call_glm(user_prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    if not ZAI_API_KEY:
        raise HTTPException(status_code=500, detail="ZAI_API_KEY is missing in .env file")

    headers = {
        "Authorization": f"Bearer {ZAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": ZAI_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                ZAI_API_URL,
                headers=headers,
                json=payload,
                timeout=90
            )

            if response.status_code != 200:
                raise Exception(f"GLM API error: {response.text}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return parse_glm_json(content)

        except Exception as e:
            if attempt == max_retries:
                return fallback_response(str(e))

    return fallback_response("Unknown GLM error")


def parse_glm_json(content: str) -> Dict[str, Any]:
    try:
        content = content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        parsed = json.loads(content)

        required_fields = [
            "intent",
            "confidence",
            "workflow_stage",
            "missing_information",
            "next_action",
            "structured_output"
        ]

        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing field: {field}")

        allowed_actions = {
            "ask_follow_up_questions",
            "generate_refund_checklist",
            "create_refund_case",
            "validate_refund_request",
            "calculate_refund_estimate",
            "approve_refund_prototype",
            "reject_refund_request",
            "manual_review_required"
        }

        if parsed["next_action"] not in allowed_actions:
            parsed["next_action"] = "manual_review_required"

        if not isinstance(parsed.get("missing_information"), list):
            parsed["missing_information"] = []

        if not isinstance(parsed.get("structured_output"), dict):
            parsed["structured_output"] = {}

        return parsed

    except Exception as e:
        raise ValueError(f"Invalid GLM JSON response: {str(e)}")


def fallback_response(error_message: str) -> Dict[str, Any]:
    return {
        "intent": "unknown",
        "confidence": 0.0,
        "workflow_stage": "failed",
        "missing_information": [],
        "next_action": "manual_review_required",
        "structured_output": {
            "summary": "The refund workflow engine could not complete AI reasoning for this request.",
            "refund_details": {
                "order_id": "unknown",
                "customer_name": "unknown",
                "customer_contact": "unknown",
                "product_or_service": "unknown",
                "refund_reason": "unknown",
                "purchase_date": "unknown",
                "payment_method": "unknown",
                "refund_amount": "unknown",
                "currency": "unknown"
            },
            "eligibility_assessment": {
                "is_likely_eligible": False,
                "reason": "Unable to assess refund eligibility.",
                "policy_flags": ["processing_error"]
            },
            "tasks": [],
            "questions_to_user": [
                "Can you provide your order ID, refund reason, product name, and contact email or phone number?"
            ],
            "recommended_tools_or_apis": [],
            "final_message": "Sorry, the refund workflow engine could not process this request right now. Please try again, or ask support to review it manually.",
            "error": error_message
        }
    }


def check_input_limit(text: str) -> None:
    max_chars = 6000

    if len(text) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long. Maximum allowed is {max_chars} characters."
        )

    if len(text.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Input is too short. Please provide more refund details."
        )


REQUIRED_REFUND_FIELDS = {
    "order_id": "order ID or receipt number",
    "refund_reason": "refund reason",
    "product_or_service": "product or service name",
    "customer_contact": "customer email or phone number"
}

RISKY_REFUND_KEYWORDS = [
    "fraud",
    "scam",
    "duplicate payment",
    "charged twice",
    "chargeback",
    "bank dispute",
    "legal",
    "police",
    "high value",
    "expensive"
]


def is_known_value(value: Any) -> bool:
    if value is None:
        return False

    text = str(value).strip().lower()
    return bool(text) and text not in {
        "unknown",
        "n/a",
        "na",
        "none",
        "not provided",
        "not_provided",
        "missing",
        "string_or_unknown"
    }


def get_refund_details(glm_result: Dict[str, Any]) -> Dict[str, Any]:
    structured_output = glm_result.setdefault("structured_output", {})
    refund_details = structured_output.setdefault("refund_details", {})

    if not isinstance(refund_details, dict):
        refund_details = {}
        structured_output["refund_details"] = refund_details

    return refund_details


def find_missing_required_fields(glm_result: Dict[str, Any]) -> List[str]:
    refund_details = get_refund_details(glm_result)
    missing_fields = []

    for field_name in REQUIRED_REFUND_FIELDS:
        if not is_known_value(refund_details.get(field_name)):
            missing_fields.append(field_name)

    existing_missing = glm_result.get("missing_information", [])
    if isinstance(existing_missing, list):
        for item in existing_missing:
            normalized = str(item).strip()
            if (
                normalized in REQUIRED_REFUND_FIELDS
                and normalized not in missing_fields
                and not is_known_value(refund_details.get(normalized))
            ):
                missing_fields.append(normalized)

    return missing_fields


def request_missing_refund_info(glm_result: Dict[str, Any], missing_fields: List[str]) -> None:
    structured_output = glm_result.setdefault("structured_output", {})
    readable_missing = [
        REQUIRED_REFUND_FIELDS.get(field, field.replace("_", " "))
        for field in missing_fields
    ]

    glm_result["workflow_stage"] = "collecting_information"
    glm_result["missing_information"] = missing_fields
    glm_result["next_action"] = "ask_follow_up_questions"

    structured_output["questions_to_user"] = [
        f"Please provide your {item}."
        for item in readable_missing
    ]
    structured_output["final_message"] = (
        "I need a little more information before I can validate this refund request: "
        + ", ".join(readable_missing)
        + "."
    )


def contains_risky_refund_signal(glm_result: Dict[str, Any]) -> bool:
    structured_output = glm_result.get("structured_output", {})
    text = json.dumps(structured_output).lower()
    return any(keyword in text for keyword in RISKY_REFUND_KEYWORDS)


def verify_refund_against_order_db(refund_details: Dict[str, Any]) -> Dict[str, Any]:
    order_id = str(refund_details.get("order_id", "")).strip().upper()
    customer_contact = str(refund_details.get("customer_contact", "")).strip().lower()

    if not is_known_value(order_id):
        return {
            "verification_status": "blocked_missing_order_id",
            "is_verified": False,
            "is_refund_eligible": False,
            "reason": "Order ID is required before the backend can verify this refund.",
            "order_record": None,
            "policy_flags": ["missing_order_id"]
        }

    order_record = mock_order_db.get(order_id)
    if not order_record:
        return {
            "verification_status": "order_not_found",
            "is_verified": False,
            "is_refund_eligible": False,
            "reason": f"Order {order_id} was not found in the prototype order database.",
            "order_record": None,
            "policy_flags": ["order_not_found"]
        }

    expected_contact = str(order_record.get("customer_contact", "")).strip().lower()
    if is_known_value(customer_contact) and customer_contact != expected_contact:
        return {
            "verification_status": "customer_contact_mismatch",
            "is_verified": False,
            "is_refund_eligible": False,
            "reason": "The provided customer contact does not match the order record.",
            "order_record": order_record,
            "policy_flags": ["customer_contact_mismatch"]
        }

    if order_record.get("payment_status") != "paid":
        return {
            "verification_status": "payment_not_confirmed",
            "is_verified": False,
            "is_refund_eligible": False,
            "reason": "Payment is not confirmed for this order.",
            "order_record": order_record,
            "policy_flags": ["payment_not_confirmed"]
        }

    if not order_record.get("refund_eligible", False):
        return {
            "verification_status": "not_refund_eligible",
            "is_verified": True,
            "is_refund_eligible": False,
            "reason": "The order exists, but it is not eligible for refund under the prototype policy.",
            "order_record": order_record,
            "policy_flags": ["not_refund_eligible"]
        }

    if order_record.get("risk_level") == "high":
        return {
            "verification_status": "verified_requires_manual_review",
            "is_verified": True,
            "is_refund_eligible": False,
            "reason": "The order is verified but high-value, so it requires manual review before approval.",
            "order_record": order_record,
            "policy_flags": ["high_value_manual_review"]
        }

    return {
        "verification_status": "verified_refund_eligible",
        "is_verified": True,
        "is_refund_eligible": True,
        "reason": "Order, payment, customer contact, and prototype refund policy were verified.",
        "order_record": order_record,
        "policy_flags": []
    }


def get_workflow_verification(workflow: Dict[str, Any]) -> Dict[str, Any]:
    refund_details = workflow.get("structured_output", {}).get("refund_details", {})
    if not isinstance(refund_details, dict):
        refund_details = {}

    return verify_refund_against_order_db(refund_details)


def build_compact_workflow_context(workflow: Dict[str, Any]) -> Dict[str, Any]:
    structured_output = workflow.get("structured_output", {})

    return {
        "workflow_id": workflow.get("workflow_id"),
        "status": workflow.get("status"),
        "intent": workflow.get("intent"),
        "current_stage": workflow.get("current_stage"),
        "missing_information": workflow.get("missing_information", []),
        "next_action": workflow.get("next_action"),
        "summary": structured_output.get("summary", ""),
        "refund_details": structured_output.get("refund_details", {}),
        "eligibility_assessment": structured_output.get("eligibility_assessment", {}),
        "last_customer_message": workflow.get("history", [{}])[-1].get("content", "")
    }


def extract_refund_details_from_text(refund_details: Dict[str, Any], text: str) -> Dict[str, Any]:
    answer_text = text.strip()
    answer_lower = answer_text.lower()

    email_match = re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", answer_text)
    if email_match:
        refund_details["customer_contact"] = email_match.group(0)

    order_match = re.search(r"\bORD\d+\b", answer_text, re.IGNORECASE)
    if order_match:
        refund_details["order_id"] = order_match.group(0).upper()

    for order_record in mock_order_db.values():
        product = str(order_record.get("product_or_service", ""))
        if product and product.lower() in answer_lower:
            refund_details["product_or_service"] = product
            break

    if not is_known_value(refund_details.get("product_or_service")):
        product_match = re.search(
            r"(?:product|item)\s+(?:was|is)\s+(.+?)(?:\s+and\s+my\s+email|\s+and\s+my\s+phone|\.|$)",
            answer_text,
            re.IGNORECASE
        )
        if product_match:
            refund_details["product_or_service"] = product_match.group(1).strip()

    name_match = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s'.-]{1,60})", answer_text, re.IGNORECASE)
    if name_match:
        refund_details["customer_name"] = name_match.group(1).strip().rstrip(".")

    reason_map = {
        "wrong item": "wrong_item_received",
        "received the wrong": "wrong_item_received",
        "damaged": "damaged_or_defective_item",
        "defective": "damaged_or_defective_item",
        "charged twice": "duplicate_payment",
        "duplicate payment": "duplicate_payment"
    }
    for phrase, reason in reason_map.items():
        if phrase in answer_lower:
            refund_details["refund_reason"] = reason
            break

    if "card" in answer_lower:
        refund_details["payment_method"] = "card"

    return refund_details


def recover_initial_request_without_glm(user_input: str) -> Dict[str, Any]:
    refund_details = {
        "order_id": "unknown",
        "customer_name": "unknown",
        "customer_contact": "unknown",
        "product_or_service": "unknown",
        "refund_reason": "unknown",
        "purchase_date": "unknown",
        "payment_method": "unknown",
        "refund_amount": "unknown",
        "currency": "unknown"
    }
    extract_refund_details_from_text(refund_details, user_input)

    recovered_result = {
        "intent": "request_refund",
        "confidence": 0.55,
        "workflow_stage": "checking_refund_eligibility",
        "missing_information": [],
        "next_action": "validate_refund_request",
        "structured_output": {
            "summary": "Recovered refund workflow from customer input after AI reasoning failed.",
            "refund_details": refund_details,
            "eligibility_assessment": {
                "is_likely_eligible": False,
                "reason": "Backend recovered enough details to run prototype verification.",
                "policy_flags": ["glm_recovery_mode"]
            },
            "tasks": ["Verify recovered refund details against the prototype order database."],
            "agent_tasks": [],
            "questions_to_user": [],
            "recommended_tools_or_apis": ["mock_order_db"],
            "final_message": "I recovered the refund details and will verify the order now."
        }
    }

    recovered_result = enforce_refund_validation_rules(recovered_result)
    if not recovered_result.get("missing_information"):
        recovered_result["workflow_stage"] = "checking_refund_eligibility"
        recovered_result["next_action"] = "validate_refund_request"

    return recovered_result


def recover_follow_up_without_glm(workflow: Dict[str, Any], answer: str) -> Dict[str, Any]:
    structured_output = json.loads(json.dumps(workflow.get("structured_output", {})))
    refund_details = structured_output.setdefault("refund_details", {})
    extract_refund_details_from_text(refund_details, answer)

    recovered_result = {
        "intent": workflow.get("intent", "request_refund"),
        "confidence": workflow.get("confidence", 0.55),
        "workflow_stage": workflow.get("current_stage", "collecting_information"),
        "missing_information": workflow.get("missing_information", []),
        "next_action": "validate_refund_request",
        "structured_output": structured_output
    }

    recovered_result = enforce_refund_validation_rules(recovered_result)
    if not recovered_result.get("missing_information"):
        recovered_result["workflow_stage"] = "checking_refund_eligibility"
        recovered_result["next_action"] = "validate_refund_request"
        recovered_result["structured_output"]["final_message"] = (
            "Thanks. I have updated the refund details and will verify the order now."
        )

    return recovered_result


def enforce_refund_validation_rules(glm_result: Dict[str, Any]) -> Dict[str, Any]:
    """Backend safety net so GLM cannot accidentally approve incomplete requests."""
    if glm_result.get("workflow_stage") == "failed":
        return glm_result

    protected_actions = {
        "create_refund_case",
        "validate_refund_request",
        "calculate_refund_estimate",
        "approve_refund_prototype",
        "reject_refund_request"
    }

    missing_fields = find_missing_required_fields(glm_result)
    glm_result["missing_information"] = missing_fields

    if missing_fields and glm_result.get("next_action") in protected_actions:
        request_missing_refund_info(glm_result, missing_fields)
        return glm_result

    if contains_risky_refund_signal(glm_result) and glm_result.get("next_action") == "approve_refund_prototype":
        glm_result["workflow_stage"] = "manual_review"
        glm_result["next_action"] = "manual_review_required"
        glm_result["missing_information"] = []
        glm_result.setdefault("structured_output", {}).setdefault("eligibility_assessment", {})[
            "policy_flags"
        ] = ["risk_signal_requires_human_review"]

    return glm_result


def determine_status(glm_result: Dict[str, Any]) -> str:
    if glm_result["workflow_stage"] == "failed":
        return "failed"

    if glm_result["next_action"] == "manual_review_required":
        return "manual_review"

    if len(glm_result["missing_information"]) > 0:
        return "waiting_for_customer"

    if glm_result["next_action"] in {
        "approve_refund_prototype",
        "reject_refund_request",
        "create_refund_case"
    }:
        return "ready"

    if glm_result["workflow_stage"] == "ready_to_process_refund":
        return "ready"

    return "active"


# -------------------------------
# REFUND ACTION MAPPING SECTION
# -------------------------------

def action_ask_follow_up_questions(workflow: Dict[str, Any]) -> Dict[str, Any]:
    questions = workflow["structured_output"].get("questions_to_user", [])

    return {
        "action_name": "ask_follow_up_questions",
        "executed": True,
        "visible_to_user": True,
        "message": workflow["structured_output"].get("final_message", "Please provide the missing refund details."),
        "questions": questions,
        "system_note": "Frontend should display these refund questions and wait for the customer reply."
    }


def action_generate_refund_checklist(workflow: Dict[str, Any]) -> Dict[str, Any]:
    tasks = workflow["structured_output"].get("tasks", [])

    if not tasks:
        tasks = [
            "Prepare your order ID or receipt number.",
            "Prepare the product or service name.",
            "Explain the reason for the refund.",
            "Provide your contact email or phone number.",
            "Wait for refund eligibility review."
        ]

    checklist = [
        {
            "id": index + 1,
            "task": task,
            "completed": False
        }
        for index, task in enumerate(tasks)
    ]

    workflow["refund_checklist"] = checklist

    return {
        "action_name": "generate_refund_checklist",
        "executed": True,
        "visible_to_user": True,
        "checklist": checklist,
        "system_note": "Refund checklist generated from GLM tasks."
    }


def action_create_refund_case(workflow: Dict[str, Any]) -> Dict[str, Any]:
    missing_fields = workflow.get("missing_information", [])
    if missing_fields:
        return action_ask_follow_up_questions(workflow)

    refund_case_id = "REFUND-" + str(uuid4())[:8].upper()
    refund_details = workflow["structured_output"].get("refund_details", {})
    eligibility = workflow["structured_output"].get("eligibility_assessment", {})

    refund_case = {
        "refund_case_id": refund_case_id,
        "workflow_id": workflow["workflow_id"],
        "intent": workflow["intent"],
        "summary": workflow["structured_output"].get("summary", ""),
        "refund_details": refund_details,
        "eligibility_assessment": eligibility,
        "status": "open",
        "priority": determine_refund_priority(workflow),
        "created_at": datetime.utcnow().isoformat()
    }

    refund_case_db[refund_case_id] = refund_case
    workflow["refund_case_id"] = refund_case_id

    return {
        "action_name": "create_refund_case",
        "executed": True,
        "visible_to_user": True,
        "refund_case": refund_case,
        "system_note": "Prototype refund case created in memory. No real refund has been issued."
    }


def action_validate_refund_request(workflow: Dict[str, Any]) -> Dict[str, Any]:
    refund_details = workflow["structured_output"].get("refund_details", {})
    eligibility = workflow["structured_output"].get("eligibility_assessment", {})
    missing_fields = [
        field for field in workflow.get("missing_information", [])
        if field in REQUIRED_REFUND_FIELDS
    ]
    workflow["missing_information"] = missing_fields
    policy_flags = eligibility.get("policy_flags", [])
    verification_result = get_workflow_verification(workflow)

    if missing_fields:
        validation_status = "blocked_missing_required_information"
        reason = (
            "Cannot validate this refund request yet because required information is missing: "
            + ", ".join(missing_fields)
            + "."
        )
        is_likely_eligible = False
    elif not verification_result["is_verified"]:
        validation_status = verification_result["verification_status"]
        reason = verification_result["reason"]
        is_likely_eligible = False
        policy_flags = list(set(policy_flags + verification_result["policy_flags"]))
    elif not verification_result["is_refund_eligible"]:
        validation_status = verification_result["verification_status"]
        reason = verification_result["reason"]
        is_likely_eligible = False
        policy_flags = list(set(policy_flags + verification_result["policy_flags"]))
    elif contains_risky_refund_signal({
        "structured_output": workflow.get("structured_output", {})
    }):
        validation_status = "requires_manual_review"
        reason = "Risk signals were detected, so this refund must be reviewed by a human before approval."
        is_likely_eligible = False
        if "risk_signal_requires_human_review" not in policy_flags:
            policy_flags.append("risk_signal_requires_human_review")
    else:
        validation_status = verification_result["verification_status"]
        reason = verification_result["reason"]
        is_likely_eligible = True

        order_record = verification_result.get("order_record") or {}
        refund_details["refund_amount"] = refund_details.get("refund_amount") or order_record.get("amount", "unknown")
        refund_details["currency"] = refund_details.get("currency") or order_record.get("currency", "unknown")

    case_action = None

    if validation_status == "verified_refund_eligible":
        workflow["status"] = "ready"
        workflow["current_stage"] = "ready_to_process_refund"
        workflow["next_action"] = "approve_refund_prototype"
        case_action = action_create_refund_case(workflow)
    elif validation_status == "verified_requires_manual_review":
        workflow["status"] = "manual_review"
        workflow["current_stage"] = "manual_review"
        workflow["next_action"] = "manual_review_required"
    elif validation_status in {
        "not_refund_eligible",
        "order_not_found",
        "customer_contact_mismatch",
        "payment_not_confirmed"
    }:
        workflow["status"] = "active"
        workflow["current_stage"] = "checking_refund_eligibility"
        workflow["next_action"] = "reject_refund_request"

    validation_result = {
        "validated_at": datetime.utcnow().isoformat(),
        "order_id": refund_details.get("order_id", "unknown"),
        "validation_status": validation_status,
        "is_likely_eligible": is_likely_eligible,
        "reason": reason,
        "missing_required_fields": missing_fields,
        "policy_flags": policy_flags,
        "verification_result": verification_result,
        "created_case": case_action.get("refund_case") if case_action else None,
        "next_recommended_action": workflow.get("next_action"),
        "prototype_note": "Verified against the in-memory prototype order database."
    }

    workflow["validation_result"] = validation_result

    return {
        "action_name": "validate_refund_request",
        "executed": True,
        "visible_to_user": True,
        "validation_result": validation_result,
        "system_note": "Refund request validation prepared in prototype mode."
    }


def action_calculate_refund_estimate(workflow: Dict[str, Any]) -> Dict[str, Any]:
    refund_details = workflow["structured_output"].get("refund_details", {})
    amount = refund_details.get("refund_amount", "unknown")
    currency = refund_details.get("currency", "unknown")

    refund_estimate = {
        "calculated_at": datetime.utcnow().isoformat(),
        "estimated_refund_amount": amount,
        "currency": currency,
        "deductions": [],
        "estimate_status": "estimate_only",
        "prototype_note": "Final refund amount must be verified using real payment and order records."
    }

    workflow["refund_estimate"] = refund_estimate

    return {
        "action_name": "calculate_refund_estimate",
        "executed": True,
        "visible_to_user": True,
        "refund_estimate": refund_estimate,
        "system_note": "Refund estimate calculated in prototype mode."
    }


def action_approve_refund_prototype(workflow: Dict[str, Any]) -> Dict[str, Any]:
    if workflow.get("missing_information"):
        return action_ask_follow_up_questions(workflow)

    verification_result = get_workflow_verification(workflow)
    if not verification_result["is_verified"]:
        workflow["validation_result"] = verification_result
        return action_validate_refund_request(workflow)

    if not verification_result["is_refund_eligible"]:
        if "manual_review" in verification_result["verification_status"]:
            return action_manual_review_required(workflow)
        return action_reject_refund_request(workflow)

    if contains_risky_refund_signal({
        "structured_output": workflow.get("structured_output", {})
    }):
        return action_manual_review_required(workflow)

    refund_case_id = workflow.get("refund_case_id")

    if not refund_case_id:
        case_action = action_create_refund_case(workflow)
        refund_case_id = case_action["refund_case"]["refund_case_id"]

    refund_case = refund_case_db[refund_case_id]
    refund_case["status"] = "approved_prototype"
    refund_case["approved_at"] = datetime.utcnow().isoformat()

    approval_result = {
        "refund_case_id": refund_case_id,
        "approved_at": refund_case["approved_at"],
        "status": "approved_prototype",
        "message": "Refund request approved in prototype mode. No real payment refund has been sent.",
        "next_steps": [
            "Verify order details in the real order database.",
            "Confirm payment method and refundable amount.",
            "Send refund confirmation to the customer.",
            "Process refund through payment gateway."
        ]
    }

    workflow["approval_result"] = approval_result

    return {
        "action_name": "approve_refund_prototype",
        "executed": True,
        "visible_to_user": True,
        "approval_result": approval_result,
        "system_note": "Prototype approval only. No real money movement occurred."
    }


def action_reject_refund_request(workflow: Dict[str, Any]) -> Dict[str, Any]:
    refund_case_id = workflow.get("refund_case_id")

    if not refund_case_id:
        case_action = action_create_refund_case(workflow)
        refund_case_id = case_action["refund_case"]["refund_case_id"]

    refund_case = refund_case_db[refund_case_id]
    refund_case["status"] = "rejected"
    refund_case["rejected_at"] = datetime.utcnow().isoformat()

    eligibility = workflow["structured_output"].get("eligibility_assessment", {})

    rejection_result = {
        "refund_case_id": refund_case_id,
        "rejected_at": refund_case["rejected_at"],
        "status": "rejected",
        "reason": eligibility.get("reason", workflow["structured_output"].get("summary", "Refund request is not eligible.")),
        "message": workflow["structured_output"].get(
            "final_message",
            "Based on the available information, this refund request cannot be approved."
        )
    }

    workflow["rejection_result"] = rejection_result

    return {
        "action_name": "reject_refund_request",
        "executed": True,
        "visible_to_user": True,
        "rejection_result": rejection_result,
        "system_note": "Refund request rejected in prototype mode."
    }


def action_manual_review_required(workflow: Dict[str, Any]) -> Dict[str, Any]:
    review_id = "REVIEW-" + str(uuid4())[:8].upper()

    manual_review = {
        "review_id": review_id,
        "workflow_id": workflow.get("workflow_id"),
        "intent": workflow.get("intent"),
        "reason": workflow["structured_output"].get("summary", "Refund request requires human review."),
        "status": "pending_human_review",
        "created_at": datetime.utcnow().isoformat()
    }

    workflow["manual_review"] = manual_review

    return {
        "action_name": "manual_review_required",
        "executed": True,
        "visible_to_user": True,
        "manual_review": manual_review,
        "message": "This refund request requires human review before any refund decision is made.",
        "system_note": "System stopped automatic refund approval to prevent unsafe or incorrect processing."
    }


def determine_refund_priority(workflow: Dict[str, Any]) -> str:
    text = json.dumps(workflow.get("structured_output", {})).lower()

    high_priority_words = [
        "fraud", "duplicate payment", "charged twice", "expensive", "high value",
        "legal", "bank", "urgent", "angry", "complaint"
    ]

    if any(word in text for word in high_priority_words):
        return "high"

    if workflow.get("next_action") == "manual_review_required":
        return "high"

    return "normal"


ACTION_MAP = {
    "ask_follow_up_questions": action_ask_follow_up_questions,
    "generate_refund_checklist": action_generate_refund_checklist,
    "create_refund_case": action_create_refund_case,
    "validate_refund_request": action_validate_refund_request,
    "calculate_refund_estimate": action_calculate_refund_estimate,
    "approve_refund_prototype": action_approve_refund_prototype,
    "reject_refund_request": action_reject_refund_request,
    "manual_review_required": action_manual_review_required
}


AGENT_REGISTRY = {
    "intake_agent": {
        "name": "Refund Intake Agent",
        "role": "Extracts customer intent, refund details, and missing information."
    },
    "customer_information_agent": {
        "name": "Customer Information Agent",
        "role": "Collects follow-up answers and keeps the customer-facing questions focused."
    },
    "checklist_agent": {
        "name": "Refund Checklist Agent",
        "role": "Turns workflow tasks into an actionable refund preparation checklist."
    },
    "case_creation_agent": {
        "name": "Refund Case Agent",
        "role": "Creates and stores prototype refund cases."
    },
    "eligibility_agent": {
        "name": "Eligibility Agent",
        "role": "Prepares order, policy, and payment validation work."
    },
    "refund_estimation_agent": {
        "name": "Refund Estimation Agent",
        "role": "Prepares prototype refund amount estimates."
    },
    "approval_agent": {
        "name": "Refund Approval Agent",
        "role": "Approves safe prototype refunds after required information is present."
    },
    "rejection_agent": {
        "name": "Refund Rejection Agent",
        "role": "Rejects clearly ineligible prototype refund requests."
    },
    "risk_review_agent": {
        "name": "Risk Review Agent",
        "role": "Escalates risky, unclear, or high-value cases to human review."
    },
    "communication_agent": {
        "name": "Customer Communication Agent",
        "role": "Prepares the final customer-facing message."
    }
}


ACTION_AGENT_MAP = {
    "ask_follow_up_questions": "customer_information_agent",
    "generate_refund_checklist": "checklist_agent",
    "create_refund_case": "case_creation_agent",
    "validate_refund_request": "eligibility_agent",
    "calculate_refund_estimate": "refund_estimation_agent",
    "approve_refund_prototype": "approval_agent",
    "reject_refund_request": "rejection_agent",
    "manual_review_required": "risk_review_agent"
}


def create_agent_task(
    agent_id: str,
    task: str,
    priority: str = "normal",
    status: str = "pending",
    result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    agent = AGENT_REGISTRY[agent_id]

    return {
        "task_id": "TASK-" + str(uuid4())[:8].upper(),
        "agent_id": agent_id,
        "agent_name": agent["name"],
        "agent_role": agent["role"],
        "task": task,
        "priority": priority,
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": datetime.utcnow().isoformat() if status == "completed" else None,
        "result": result or {}
    }


def get_structured_agent_tasks(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_tasks = workflow.get("structured_output", {}).get("agent_tasks", [])
    normalized_tasks = []

    if not isinstance(raw_tasks, list):
        return normalized_tasks

    for item in raw_tasks:
        if not isinstance(item, dict):
            continue

        agent_id = item.get("agent", "intake_agent")
        if agent_id not in AGENT_REGISTRY:
            agent_id = "intake_agent"

        normalized_tasks.append(create_agent_task(
            agent_id=agent_id,
            task=item.get("task", "Review refund workflow information."),
            priority=item.get("priority", "normal"),
            status=item.get("status", "pending")
        ))

    return normalized_tasks


def build_agent_task_plan(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured_output = workflow.get("structured_output", {})
    next_action = workflow.get("next_action", "manual_review_required")
    missing_information = workflow.get("missing_information", [])
    summary = structured_output.get("summary", "Review the customer refund request.")

    task_plan = [
        create_agent_task(
            "intake_agent",
            "Extract refund intent, customer details, and missing information.",
            status="completed",
            result={
                "intent": workflow.get("intent"),
                "missing_information": missing_information,
                "summary": summary
            }
        )
    ]

    task_plan.extend(get_structured_agent_tasks(workflow))

    if missing_information:
        task_plan.append(create_agent_task(
            "customer_information_agent",
            "Ask the customer only for the remaining missing refund information.",
            priority="high" if "order_id" in missing_information else "normal"
        ))

    specialist_agent_id = ACTION_AGENT_MAP.get(next_action, "risk_review_agent")
    task_plan.append(create_agent_task(
        specialist_agent_id,
        f"Run mapped workflow action: {next_action}.",
        priority="high" if specialist_agent_id in {"risk_review_agent", "approval_agent"} else "normal"
    ))

    policy_flags = structured_output.get("eligibility_assessment", {}).get("policy_flags", [])
    if next_action == "manual_review_required" or policy_flags:
        task_plan.append(create_agent_task(
            "risk_review_agent",
            "Review policy flags and decide whether human support must take over.",
            priority="high"
        ))

    task_plan.append(create_agent_task(
        "communication_agent",
        "Prepare the customer-facing response for the current refund workflow state."
    ))

    return task_plan


def complete_primary_agent_task(
    workflow: Dict[str, Any],
    action_name: str,
    action_result: Dict[str, Any]
) -> None:
    primary_agent_id = ACTION_AGENT_MAP.get(action_name, "risk_review_agent")
    action_task_prefix = f"Run mapped workflow action: {action_name}."

    for task in workflow.get("agent_tasks", []):
        if (
            task["agent_id"] == primary_agent_id
            and task["status"] == "pending"
            and task["task"] == action_task_prefix
        ):
            task["status"] = "completed"
            task["completed_at"] = datetime.utcnow().isoformat()
            task["result"] = {
                "action_name": action_result.get("action_name"),
                "executed": action_result.get("executed", False),
                "system_note": action_result.get("system_note", "")
            }
            break


def update_agent_summary(workflow: Dict[str, Any]) -> None:
    agent_tasks = workflow.get("agent_tasks", [])
    workflow["assigned_agents"] = sorted({task["agent_id"] for task in agent_tasks})
    workflow["agent_task_summary"] = {
        "total": len(agent_tasks),
        "completed": len([task for task in agent_tasks if task["status"] == "completed"]),
        "pending": len([task for task in agent_tasks if task["status"] == "pending"]),
        "agents": workflow["assigned_agents"]
    }


def run_mapped_action(workflow: Dict[str, Any]) -> Dict[str, Any]:
    action_name = workflow.get("next_action", "manual_review_required")

    if action_name not in ACTION_MAP:
        action_name = "manual_review_required"

    workflow["agent_tasks"] = build_agent_task_plan(workflow)
    update_agent_summary(workflow)

    action_function = ACTION_MAP[action_name]
    result = action_function(workflow)

    complete_primary_agent_task(workflow, action_name, result)
    update_agent_summary(workflow)

    workflow["mapped_action"] = result
    return result


def create_workflow_object(
    workflow_id: str,
    user_input: str,
    user_type: str,
    glm_result: Dict[str, Any],
    now: str
) -> Dict[str, Any]:

    workflow_data = {
        "workflow_id": workflow_id,
        "original_input": user_input,
        "user_type": user_type,
        "status": determine_status(glm_result),
        "intent": glm_result["intent"],
        "confidence": glm_result["confidence"],
        "current_stage": glm_result["workflow_stage"],
        "missing_information": glm_result["missing_information"],
        "next_action": glm_result["next_action"],
        "refund_case_id": None,
        "assigned_agents": [],
        "agent_tasks": [],
        "agent_task_summary": {},
        "mapped_action": {},
        "structured_output": glm_result["structured_output"],
        "history": [
            {
                "role": "customer",
                "content": user_input,
                "timestamp": now
            },
            {
                "role": "glm",
                "content": glm_result,
                "timestamp": now
            }
        ],
        "created_at": now,
        "updated_at": now
    }

    run_mapped_action(workflow_data)
    return workflow_data


@app.get("/")
def root():
    return {
        "message": "Multi-Agent Customer Refund System Backend is running",
        "model": ZAI_MODEL,
        "architecture": "multi_agent_task_dispatch",
        "available_agents": list(AGENT_REGISTRY.keys()),
        "status": "ok"
    }


@app.get("/agents")
def list_agents():
    return {
        "total": len(AGENT_REGISTRY),
        "architecture": "multi_agent_task_dispatch",
        "agents": AGENT_REGISTRY,
        "action_agent_map": ACTION_AGENT_MAP
    }


# Keep /workflow/start so your existing frontend pattern can still work.
# Add /refund/start so the API also looks like a refund system.
@app.post("/workflow/start", response_model=WorkflowResponse)
@app.post("/refund/start", response_model=WorkflowResponse)
def start_refund_workflow(request: WorkflowRequest):
    check_input_limit(request.user_input)

    workflow_id = str(uuid4())
    now = datetime.utcnow().isoformat()

    prompt = f"""
User type: {request.user_type}

Customer refund request:
{request.user_input}

Analyse this customer message and create a refund workflow plan.

Remember:
- Choose only one allowed next_action.
- If information is missing, ask only for missing refund information.
- Do not say a real refund has been completed.
- If this is a simple refund request with enough details, create a refund case.
- If eligibility or payment data must be checked, validate the refund request.
- If all required information is collected and it appears eligible, approve only in prototype mode.
"""

    glm_result = enforce_refund_validation_rules(call_glm(prompt))

    if glm_result.get("workflow_stage") == "failed":
        glm_result = recover_initial_request_without_glm(request.user_input)

    workflow_data = create_workflow_object(
        workflow_id=workflow_id,
        user_input=request.user_input,
        user_type=request.user_type,
        glm_result=glm_result,
        now=now
    )

    workflow_db[workflow_id] = workflow_data

    return WorkflowResponse(**workflow_data)


@app.post("/workflow/follow-up", response_model=WorkflowResponse)
@app.post("/refund/follow-up", response_model=WorkflowResponse)
def continue_refund_workflow(request: FollowUpRequest):
    if request.workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Refund workflow not found")

    workflow = workflow_db[request.workflow_id]
    check_input_limit(request.answer)

    now = datetime.utcnow().isoformat()

    previous_context = json.dumps(build_compact_workflow_context(workflow), indent=2)

    prompt = f"""
This is an existing customer refund workflow.

Previous refund workflow state:
{previous_context}

Customer follow-up answer:
{request.answer}

Update the refund workflow state.

Important:
- Do not repeat questions already answered.
- Only ask for remaining missing refund information.
- Do not say a real refund has been completed.
- If all required information is collected and the request appears eligible, use next_action = approve_refund_prototype and workflow_stage = ready_to_process_refund.
- If the next best step is a checklist, use next_action = generate_refund_checklist.
- If a case should be opened, use next_action = create_refund_case.
- If order or payment needs checking, use next_action = validate_refund_request.
- Choose only one allowed next_action.
"""

    glm_result = enforce_refund_validation_rules(call_glm(prompt))

    if glm_result.get("workflow_stage") == "failed":
        glm_result = recover_follow_up_without_glm(workflow, request.answer)

    workflow.update({
        "status": determine_status(glm_result),
        "intent": glm_result["intent"],
        "confidence": glm_result["confidence"],
        "current_stage": glm_result["workflow_stage"],
        "missing_information": glm_result["missing_information"],
        "next_action": glm_result["next_action"],
        "structured_output": glm_result["structured_output"],
        "updated_at": now
    })

    run_mapped_action(workflow)

    workflow["history"].append({
        "role": "customer",
        "content": request.answer,
        "timestamp": now
    })

    workflow["history"].append({
        "role": "glm",
        "content": glm_result,
        "timestamp": now
    })

    workflow["history"].append({
        "role": "system_action",
        "content": workflow["mapped_action"],
        "timestamp": now
    })

    workflow_db[request.workflow_id] = workflow

    return WorkflowResponse(**workflow)


@app.post("/workflow/execute/{workflow_id}")
@app.post("/refund/execute/{workflow_id}")
def execute_refund_workflow(workflow_id: str):
    if workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Refund workflow not found")

    workflow = workflow_db[workflow_id]

    guardrail_result = enforce_refund_validation_rules({
        "intent": workflow.get("intent", "request_refund"),
        "confidence": workflow.get("confidence", 0.0),
        "workflow_stage": workflow.get("current_stage", "intake"),
        "missing_information": workflow.get("missing_information", []),
        "next_action": workflow.get("next_action", "manual_review_required"),
        "structured_output": workflow.get("structured_output", {})
    })

    if guardrail_result.get("missing_information"):
        workflow.update({
            "status": determine_status(guardrail_result),
            "current_stage": guardrail_result["workflow_stage"],
            "missing_information": guardrail_result["missing_information"],
            "next_action": guardrail_result["next_action"],
            "structured_output": guardrail_result["structured_output"],
            "updated_at": datetime.utcnow().isoformat()
        })
        run_mapped_action(workflow)
        workflow_db[workflow_id] = workflow

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Refund workflow cannot be executed because required refund information is missing.",
                "missing_information": workflow["missing_information"],
                "mapped_action": workflow["mapped_action"]
            }
        )

    if workflow["status"] not in ["ready", "active"]:
        raise HTTPException(
            status_code=400,
            detail="Refund workflow is not ready to execute. Missing information may still exist."
        )

    if len(workflow.get("missing_information", [])) > 0:
        raise HTTPException(
            status_code=400,
            detail="Refund workflow cannot be executed because required refund information is missing."
        )

    now = datetime.utcnow().isoformat()

    workflow["next_action"] = "approve_refund_prototype"
    execution_action = run_mapped_action(workflow)

    workflow["status"] = "completed_prototype"
    workflow["current_stage"] = "approved_prototype"
    workflow["updated_at"] = now

    workflow["history"].append({
        "role": "system_action",
        "content": execution_action,
        "timestamp": now
    })

    workflow_db[workflow_id] = workflow

    return {
        **workflow,
        "executed_at": now,
        "mapped_action": execution_action,
        "message": "Refund workflow approved in prototype mode. No real refund payment was processed."
    }


@app.get("/workflow/{workflow_id}")
@app.get("/refund/{workflow_id}")
def get_refund_workflow(workflow_id: str):
    if workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Refund workflow not found")

    return workflow_db[workflow_id]


@app.get("/workflows")
@app.get("/refunds")
def list_refund_workflows():
    return {
        "total": len(workflow_db),
        "workflows": list(workflow_db.values())
    }


@app.get("/refund-cases")
def list_refund_cases():
    return {
        "total": len(refund_case_db),
        "refund_cases": list(refund_case_db.values())
    }


@app.get("/mock-orders")
def list_mock_orders():
    return {
        "total": len(mock_order_db),
        "orders": list(mock_order_db.values()),
        "system_note": "Prototype in-memory order database used for refund verification."
    }


@app.delete("/workflow/{workflow_id}")
@app.delete("/refund/{workflow_id}")
def delete_refund_workflow(workflow_id: str):
    if workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Refund workflow not found")

    del workflow_db[workflow_id]

    return {
        "message": "Refund workflow deleted successfully",
        "workflow_id": workflow_id
    }


@app.post("/test/glm")
@app.post("/test/refund-glm")
def test_refund_glm(request: WorkflowRequest):
    check_input_limit(request.user_input)

    prompt = f"""
Test the GLM refund workflow reasoning using this customer input:

{request.user_input}

Choose one allowed next_action:
- ask_follow_up_questions
- generate_refund_checklist
- create_refund_case
- validate_refund_request
- calculate_refund_estimate
- approve_refund_prototype
- reject_refund_request
- manual_review_required
"""

    result = enforce_refund_validation_rules(call_glm(prompt))

    fake_workflow = {
        "workflow_id": "test-only",
        "intent": result["intent"],
        "next_action": result["next_action"],
        "structured_output": result["structured_output"]
    }

    mapped_action = run_mapped_action(fake_workflow)

    return {
        "input": request.user_input,
        "glm_result": result,
        "mapped_action": mapped_action
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "customer_refund_system",
        "glm_model": ZAI_MODEL,
        "api_key_loaded": bool(ZAI_API_KEY),
        "workflow_count": len(workflow_db),
        "refund_case_count": len(refund_case_db),
        "available_actions": list(ACTION_MAP.keys()),
        "available_agents": AGENT_REGISTRY
    }
