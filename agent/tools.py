import requests
from langchain_core.tools import tool

API_BASE = "https://sqfajbpym7.execute-api.us-east-1.amazonaws.com/prod"


@tool
def calculate_treatment_invoice(
    services: list,
    insurance_covered: bool,
    discount_percent: float
):
    """
    Calculate treatment invoice for hospital services.
    """

    response = requests.post(
        f"{API_BASE}/calculate-treatment-invoice",
        json={
            "services": services,
            "insurance_covered": insurance_covered,
            "discount_percent": discount_percent
        }
    )

    return response.json()


@tool
def get_staff_schedule(department: str):
    """
    Get staff schedule for a hospital department.
    """

    response = requests.post(
        f"{API_BASE}/get-staff-schedule",
        json={
            "department": department
        }
    )

    return response.json()