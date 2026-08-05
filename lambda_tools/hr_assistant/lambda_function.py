import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Mock HR staff scheduling database
DUTY_SCHEDULE = {
    "cardiology": [
        {"name": "Dr. Sarah Patel", "role": "Senior Cardiologist", "shift": "Morning (08:00 - 16:00)", "status": "On Duty"},
        {"name": "Nurse James Wilson", "role": "ICU Specialist Nurse", "shift": "Morning (08:00 - 16:00)", "status": "On Duty"}
    ],
    "orthopedics": [
        {"name": "Dr. Michael Chang", "role": "Orthopedic Surgeon", "shift": "Evening (16:00 - 00:00)", "status": "On Call"},
        {"name": "Nurse Elena Rostova", "role": "Staff Nurse", "shift": "Full Day", "status": "On Duty"}
    ],
    "emergency": [
        {"name": "Dr. Robert Vance", "role": "ER Attending Physician", "shift": "Night (00:00 - 08:00)", "status": "On Duty"},
        {"name": "Nurse Priya Sharma", "role": "Triage Nurse", "shift": "Night (00:00 - 08:00)", "status": "On Duty"}
    ],
    "pediatrics": [
        {"name": "Dr. Emily Taylor", "role": "Pediatrician", "shift": "Morning (08:00 - 16:00)", "status": "On Duty"}
    ]
}

def lambda_handler(event, context):
    """
    AWS Lambda handler to query HR shift schedules for hospital departments.
    
    Expected event payload:
    {
        "department": "cardiology"
    }
    """
    logger.info(f"Received HR schedule request: {json.dumps(event)}")
    
    try:
        department = event.get("department", "").strip().lower()
        
        if not department:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Department name is required.",
                    "available_departments": list(DUTY_SCHEDULE.keys())
                })
            }
            
        staff_on_duty = DUTY_SCHEDULE.get(department)
        
        if staff_on_duty is None:
            return {
                "statusCode": 404,
                "body": json.dumps({
                    "message": f"No staff records found for department '{department}'.",
                    "available_departments": list(DUTY_SCHEDULE.keys())
                })
            }
            
        response_payload = {
            "department": department.capitalize(),
            "active_staff_count": len(staff_on_duty),
            "staff_list": staff_on_duty
        }
        
        logger.info(f"Retrieved {len(staff_on_duty)} staff members for {department}")
        
        return {
            "statusCode": 200,
            "body": json.dumps(response_payload)
        }
        
    except Exception as e:
        logger.error(f"Error querying HR schedule: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error fetching HR schedule.", "details": str(e)})
        }