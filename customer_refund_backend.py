from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import os
import json
import requests

load_dotenv()

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_API_URL = os.getenv("ZAI_API_URL", "https://api.z.ai/api/paas/v4/chat/completions")
ZAI_MODEL = os.getenv("ZAI_MODEL", "glm-4.5")

app = FastAPI(
    title="Customer Refund System Backend",
    description="Hackathon backend using Z.AI GLM as a customer refund workflow reasoning engine",
    version="1.0.0"
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
    mapped_action: Dict[str, Any]
    structured_output: Dict[str, Any]
    created_at: str
    updated_at: str


def build_system_prompt() -> str:
    return """
You are the central reasoning engine for a CUSTOMER REFUND SYSTEM.

Your job:
1. Understand messy customer refund messages.
2. Identify the refund-related intent.
3. Decide the refund workflow stage.
4. Detect missing information needed to process a refund.
5. Choose ONE allowed next_action.
6. Generate structured actionable output for the frontend/backend.

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
            "summary": "The refund system could not process the request correctly.",
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
            "final_message": "Sorry, I could not understand the refund request properly. Please try again with your order details.",
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

    validation_result = {
        "validated_at": datetime.utcnow().isoformat(),
        "order_id": refund_details.get("order_id", "unknown"),
        "validation_status": "needs_policy_or_order_check",
        "is_likely_eligible": eligibility.get("is_likely_eligible", False),
        "reason": eligibility.get("reason", "Refund eligibility needs to be checked."),
        "policy_flags": eligibility.get("policy_flags", []),
        "prototype_note": "This backend does not connect to a real order database yet."
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


def run_mapped_action(workflow: Dict[str, Any]) -> Dict[str, Any]:
    action_name = workflow.get("next_action", "manual_review_required")

    if action_name not in ACTION_MAP:
        action_name = "manual_review_required"

    action_function = ACTION_MAP[action_name]
    result = action_function(workflow)

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
        "message": "Customer Refund System Backend is running",
        "model": ZAI_MODEL,
        "status": "ok"
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

    glm_result = call_glm(prompt)

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

    previous_context = json.dumps(workflow, indent=2)

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

    glm_result = call_glm(prompt)

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
        "workflow_id": workflow_id,
        "executed_at": now,
        "status": "completed_prototype",
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

    result = call_glm(prompt)

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
        "available_actions": list(ACTION_MAP.keys())
    }
