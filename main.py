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

load_dotenv()

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_API_URL = os.getenv("ZAI_API_URL", "https://api.z.ai/api/paas/v4/chat/completions")
ZAI_MODEL = os.getenv("ZAI_MODEL", "glm-4.5")

app = FastAPI(
    title="Customer Refund Workflow System",
    description="AI refund automation backend using Z.AI GLM as the reasoning engine",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    assigned_agents: List[str] = Field(default_factory=list)
    agent_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    mapped_action: Dict[str, Any]
    structured_output: Dict[str, Any]
    created_at: str
    updated_at: str


def build_system_prompt() -> str:
    return """
You are the coordinator for a MULTI-AGENT CUSTOMER REFUND WORKFLOW AUTOMATION SYSTEM.

Your job:
1. Understand messy customer refund requests.
2. Identify the refund reason.
3. Decide whether the refund has enough information.
4. Detect missing information.
5. Choose ONE allowed next_action.
6. Generate structured actionable output that can be assigned to specialist backend agents.

You MUST return valid JSON only.

Allowed next_action values:
- ask_follow_up_questions
- check_refund_eligibility
- create_refund_case
- generate_refund_checklist
- escalate_manual_review
- execute_refund
- manual_review_required

The JSON format must be:

{
  "intent": "string",
  "confidence": 0.0,
  "workflow_stage": "string",
  "missing_information": ["item1", "item2"],
  "next_action": "one_allowed_action_only",
  "structured_output": {
    "summary": "string",
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

Refund rules:
- Do not invent order details.
- Required refund information normally includes order_id, refund_reason, purchase_date, item_condition, payment_method, and evidence if needed.
- If order_id or refund_reason is missing, use next_action = ask_follow_up_questions.
- If the customer only asks for steps, use next_action = generate_refund_checklist.
- If enough basic information is provided but eligibility is not confirmed, use next_action = check_refund_eligibility.
- If the refund looks valid and enough information is provided, use next_action = create_refund_case.
- If the request involves fraud, high-value refund, missing evidence, no receipt, no order ID, or unclear dispute, use next_action = escalate_manual_review.
- If all required information is collected and refund is approved, use next_action = execute_refund and workflow_stage = ready_to_execute.
- If the input is unsafe, impossible, or too unclear, use next_action = manual_review_required.
- Do not repeat the same question if the user already answered it.
- Return JSON only. No markdown.
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
            "check_refund_eligibility",
            "create_refund_case",
            "generate_refund_checklist",
            "escalate_manual_review",
            "execute_refund",
            "manual_review_required"
        }

        if parsed["next_action"] not in allowed_actions:
            parsed["next_action"] = "manual_review_required"

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
            "summary": "The AI system could not process the refund request correctly.",
            "tasks": [],
            "questions_to_user": [
                "Can you rephrase your refund request with more details?"
            ],
            "recommended_tools_or_apis": [],
            "final_message": "Sorry, I could not process your refund request properly. Please try again.",
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
            detail="Input is too short. Please provide more details."
        )


def determine_status(glm_result: Dict[str, Any]) -> str:
    if glm_result["workflow_stage"] == "failed":
        return "failed"

    if glm_result["next_action"] in ["manual_review_required", "escalate_manual_review"]:
        return "manual_review"

    if len(glm_result["missing_information"]) > 0:
        return "waiting_for_user"

    if glm_result["next_action"] == "execute_refund":
        return "ready"

    if glm_result["workflow_stage"] == "ready_to_execute":
        return "ready"

    return "active"


def action_ask_follow_up_questions(workflow: Dict[str, Any]) -> Dict[str, Any]:
    questions = workflow["structured_output"].get("questions_to_user", [])

    return {
        "action_name": "ask_follow_up_questions",
        "executed": True,
        "visible_to_user": True,
        "message": workflow["structured_output"].get("final_message", ""),
        "questions": questions,
        "system_note": "Frontend should display these refund questions and wait for customer reply."
    }


def action_check_refund_eligibility(workflow: Dict[str, Any]) -> Dict[str, Any]:
    summary = workflow["structured_output"].get("summary", "")
    tasks = workflow["structured_output"].get("tasks", [])

    eligibility_result = {
        "checked_at": datetime.utcnow().isoformat(),
        "eligibility_status": "pending_policy_validation",
        "reasoning_summary": summary,
        "recommended_checks": tasks,
        "prototype_note": "Real system would verify order database, refund policy, payment status, delivery status, and evidence."
    }

    workflow["eligibility_result"] = eligibility_result

    return {
        "action_name": "check_refund_eligibility",
        "executed": True,
        "visible_to_user": True,
        "eligibility_result": eligibility_result,
        "message": "Your refund request has been checked in prototype mode. Additional policy validation may be required.",
        "system_note": "Refund eligibility checked in prototype mode."
    }


def action_create_refund_case(workflow: Dict[str, Any]) -> Dict[str, Any]:
    case_id = "REFUND-" + str(uuid4())[:8].upper()

    refund_case = {
        "case_id": case_id,
        "workflow_id": workflow["workflow_id"],
        "intent": workflow["intent"],
        "summary": workflow["structured_output"].get("summary", ""),
        "status": "open",
        "priority": "normal",
        "created_at": datetime.utcnow().isoformat()
    }

    refund_case_db[case_id] = refund_case
    workflow["refund_case_id"] = case_id

    return {
        "action_name": "create_refund_case",
        "executed": True,
        "visible_to_user": True,
        "refund_case": refund_case,
        "message": f"Your refund case has been created. Case ID: {case_id}",
        "system_note": "Refund case created in memory."
    }


def action_generate_refund_checklist(workflow: Dict[str, Any]) -> Dict[str, Any]:
    tasks = workflow["structured_output"].get("tasks", [])

    if not tasks:
        tasks = [
            "Prepare your order ID or receipt.",
            "State the refund reason clearly.",
            "Prepare evidence such as photos, screenshots, or delivery proof if applicable.",
            "Confirm your payment method.",
            "Submit the refund request for eligibility checking."
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
        "message": "A refund preparation checklist has been generated.",
        "system_note": "Refund checklist generated from GLM tasks."
    }


def action_escalate_manual_review(workflow: Dict[str, Any]) -> Dict[str, Any]:
    review_id = "REVIEW-" + str(uuid4())[:8].upper()

    review = {
        "review_id": review_id,
        "workflow_id": workflow["workflow_id"],
        "reason": workflow["structured_output"].get("summary", ""),
        "status": "pending_human_review",
        "created_at": datetime.utcnow().isoformat()
    }

    workflow["manual_review"] = review

    return {
        "action_name": "escalate_manual_review",
        "executed": True,
        "visible_to_user": True,
        "review": review,
        "message": "Your refund request has been escalated for manual review.",
        "system_note": "Automation stopped because human review is required."
    }


def action_execute_refund(workflow: Dict[str, Any]) -> Dict[str, Any]:
    refund_id = "PAYREF-" + str(uuid4())[:8].upper()

    refund_result = {
        "refund_id": refund_id,
        "executed_at": datetime.utcnow().isoformat(),
        "status": "refund_processed_prototype",
        "result": "Refund processed in prototype mode. No real money was transferred."
    }

    workflow["refund_result"] = refund_result

    return {
        "action_name": "execute_refund",
        "executed": True,
        "visible_to_user": True,
        "refund_result": refund_result,
        "message": f"Refund processed in prototype mode. Refund ID: {refund_id}",
        "system_note": "Refund executed safely in prototype mode."
    }


def action_manual_review_required(workflow: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_name": "manual_review_required",
        "executed": True,
        "visible_to_user": True,
        "message": "This refund request requires human review before execution.",
        "reason": workflow["structured_output"].get("summary", ""),
        "system_note": "System stopped automation to prevent unsafe or incorrect refund execution."
    }


ACTION_MAP = {
    "ask_follow_up_questions": action_ask_follow_up_questions,
    "check_refund_eligibility": action_check_refund_eligibility,
    "create_refund_case": action_create_refund_case,
    "generate_refund_checklist": action_generate_refund_checklist,
    "escalate_manual_review": action_escalate_manual_review,
    "execute_refund": action_execute_refund,
    "manual_review_required": action_manual_review_required
}


AGENT_REGISTRY = {
    "intake_agent": {
        "name": "Refund Intake Agent",
        "role": "Extracts customer intent, refund context, and missing information."
    },
    "customer_information_agent": {
        "name": "Customer Information Agent",
        "role": "Collects follow-up answers and keeps customer questions focused."
    },
    "checklist_agent": {
        "name": "Refund Checklist Agent",
        "role": "Turns workflow tasks into a customer preparation checklist."
    },
    "eligibility_agent": {
        "name": "Eligibility Agent",
        "role": "Prepares order, policy, payment, delivery, and evidence checks."
    },
    "case_creation_agent": {
        "name": "Refund Case Agent",
        "role": "Creates and stores prototype refund cases."
    },
    "risk_review_agent": {
        "name": "Risk Review Agent",
        "role": "Escalates risky, unclear, or high-value cases to human review."
    },
    "execution_agent": {
        "name": "Prototype Refund Execution Agent",
        "role": "Runs safe prototype execution without moving real money."
    },
    "communication_agent": {
        "name": "Customer Communication Agent",
        "role": "Prepares the customer-facing response for the current workflow state."
    }
}


ACTION_AGENT_MAP = {
    "ask_follow_up_questions": "customer_information_agent",
    "check_refund_eligibility": "eligibility_agent",
    "create_refund_case": "case_creation_agent",
    "generate_refund_checklist": "checklist_agent",
    "escalate_manual_review": "risk_review_agent",
    "execute_refund": "execution_agent",
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

    task_plan = [
        create_agent_task(
            "intake_agent",
            "Extract refund intent, workflow stage, and missing information.",
            status="completed",
            result={
                "intent": workflow.get("intent"),
                "missing_information": missing_information,
                "summary": structured_output.get("summary", "")
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
        priority="high" if specialist_agent_id in {"risk_review_agent", "execution_agent"} else "normal"
    ))

    task_plan.append(create_agent_task(
        "communication_agent",
        "Prepare the customer-facing response for this refund workflow state."
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
        "message": "Customer Refund Workflow Backend is running",
        "model": ZAI_MODEL,
        "status": "ok"
    }


@app.post("/workflow/start", response_model=WorkflowResponse)
def start_workflow(request: WorkflowRequest):
    check_input_limit(request.user_input)

    workflow_id = str(uuid4())
    now = datetime.utcnow().isoformat()

    prompt = f"""
Customer type: {request.user_type}

Customer refund request:
{request.user_input}

Analyse this as a customer refund workflow.

Remember:
- Choose only one allowed next_action.
- If order ID or refund reason is missing, ask follow-up questions.
- If basic refund details are available, check refund eligibility.
- If the refund appears valid, create a refund case.
- If the customer asks for steps, generate a refund checklist.
- If request is high-risk, unclear, or lacks proof for a serious claim, escalate manual review.
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
def continue_workflow(request: FollowUpRequest):
    if request.workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = workflow_db[request.workflow_id]
    check_input_limit(request.answer)

    now = datetime.utcnow().isoformat()
    previous_context = json.dumps(workflow, indent=2)

    prompt = f"""
This is an existing customer refund workflow.

Previous workflow state:
{previous_context}

Customer follow-up answer:
{request.answer}

Update the refund workflow state.

Important:
- Do not repeat questions already answered.
- Only ask for remaining missing refund information.
- If order ID and refund reason are now available, use check_refund_eligibility or create_refund_case.
- If all required information is collected and refund is approved, use next_action = execute_refund and workflow_stage = ready_to_execute.
- If the request is risky or lacks evidence, use escalate_manual_review.
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
def execute_workflow(workflow_id: str):
    if workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = workflow_db[workflow_id]

    if workflow["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail="Refund workflow is not ready to execute. Missing information or approval still exists."
        )

    now = datetime.utcnow().isoformat()

    workflow["next_action"] = "execute_refund"
    execution_action = run_mapped_action(workflow)

    workflow["status"] = "completed"
    workflow["current_stage"] = "completed"
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
        "status": "completed",
        "mapped_action": execution_action,
        "message": "Refund workflow executed successfully in prototype mode."
    }


@app.get("/workflow/{workflow_id}")
def get_workflow(workflow_id: str):
    if workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return workflow_db[workflow_id]


@app.get("/workflows")
def list_workflows():
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
def delete_workflow(workflow_id: str):
    if workflow_id not in workflow_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    del workflow_db[workflow_id]

    return {
        "message": "Workflow deleted successfully",
        "workflow_id": workflow_id
    }


@app.post("/test/glm")
def test_glm(request: WorkflowRequest):
    check_input_limit(request.user_input)

    prompt = f"""
Test the GLM refund workflow reasoning using this input:

{request.user_input}

Choose one allowed next_action:
- ask_follow_up_questions
- check_refund_eligibility
- create_refund_case
- generate_refund_checklist
- escalate_manual_review
- execute_refund
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
        "system": "Customer Refund Workflow System",
        "glm_model": ZAI_MODEL,
        "api_key_loaded": bool(ZAI_API_KEY),
        "workflow_count": len(workflow_db),
        "refund_case_count": len(refund_case_db),
        "available_actions": list(ACTION_MAP.keys()),
        "available_agents": AGENT_REGISTRY
    }
