import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def print_result(title, response):
    print("\n" + "=" * 60)
    print(title)
    print("Status code:", response.status_code)

    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)


# 1. Health check
response = requests.get(f"{BASE_URL}/health")
print_result("1. HEALTH CHECK", response)


# 2. Test GLM directly
response = requests.post(
    f"{BASE_URL}/test/glm",
    json={
        "user_input": "I want to apply for hostel but I don't know the steps",
        "user_type": "student"
    }
)
print_result("2. TEST GLM", response)


# 3. Start workflow
response = requests.post(
    f"{BASE_URL}/workflow/start",
    json={
        "user_input": "I want to apply for hostel but I don't know what documents I need",
        "user_type": "student"
    }
)
print_result("3. START WORKFLOW", response)

data = response.json()
workflow_id = data.get("workflow_id")

if not workflow_id:
    print("\nNo workflow_id returned. Stop testing.")
    exit()


# 4. Follow-up workflow
response = requests.post(
    f"{BASE_URL}/workflow/follow-up",
    json={
        "workflow_id": workflow_id,
        "answer": "I am an international student and my semester starts next month."
    }
)
print_result("4. FOLLOW UP WORKFLOW", response)


# 5. Get workflow
response = requests.get(f"{BASE_URL}/workflow/{workflow_id}")
print_result("5. GET WORKFLOW", response)


# 6. Try execute workflow
response = requests.post(f"{BASE_URL}/workflow/execute/{workflow_id}")
print_result("6. EXECUTE WORKFLOW", response)


# 7. List all workflows
response = requests.get(f"{BASE_URL}/workflows")
print_result("7. LIST WORKFLOWS", response)


# 8. Edge case: short input
response = requests.post(
    f"{BASE_URL}/workflow/start",
    json={
        "user_input": "help",
        "user_type": "user"
    }
)
print_result("8. EDGE CASE SHORT INPUT", response)


# 9. Edge case: nonsense input
response = requests.post(
    f"{BASE_URL}/workflow/start",
    json={
        "user_input": "asdfghjkl random unclear words no meaning",
        "user_type": "user"
    }
)
print_result("9. EDGE CASE NONSENSE INPUT", response)