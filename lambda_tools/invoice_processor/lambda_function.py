import json
import logging

# Configure production-grade structured logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Standard pricing catalog for hospital services (in USD)
SERVICE_PRICING = {
    "general consultation": 100.0,
    "specialist consultation": 200.0,
    "blood test": 50.0,
    "x-ray": 150.0,
    "mri scan": 800.0,
    "ct scan": 600.0,
    "physiotherapy": 120.0,
    "emergency room visit": 500.0,
}

TAX_RATE = 0.08  # 8% medical services tax

def lambda_handler(event, context):
    """
    AWS Lambda handler for calculating hospital treatment invoices.
    
    Expected event payload from Agent / Gateway:
    {
        "services": ["general consultation", "blood test", "x-ray"],
        "insurance_covered": True,  # Optional: Default False
        "discount_percent": 10.0    # Optional: Default 0.0
    }
    """
    logger.info(f"Received invoice calculation request: {json.dumps(event)}")
    
    try:
        # Extract inputs with defensive default handling
        services = event.get("services", [])
        insurance_covered = event.get("insurance_covered", False)
        discount_percent = float(event.get("discount_percent", 0.0))
        
        if not services:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No services provided for invoice calculation."})
            }
        
        # Calculate base line items
        line_items = []
        subtotal = 0.0
        
        for service in services:
            service_clean = service.strip().lower()
            cost = SERVICE_PRICING.get(service_clean, 150.0)  # Default fallback price
            line_items.append({"service": service_clean, "price": cost})
            subtotal += cost
            
        # Calculate discount
        discount_amount = subtotal * (discount_percent / 100.0)
        discounted_subtotal = subtotal - discount_amount
        
        # Calculate tax
        tax_amount = discounted_subtotal * TAX_RATE
        
        # Calculate insurance contribution (e.g., insurance covers 80% of total after discount/tax if covered)
        total_before_insurance = discounted_subtotal + tax_amount
        insurance_paid = total_before_insurance * 0.80 if insurance_covered else 0.0
        patient_out_of_pocket = total_before_insurance - insurance_paid
        
        invoice_summary = {
            "status": "SUCCESS",
            "itemized_bill": line_items,
            "subtotal": round(subtotal, 2),
            "discount_applied": f"{discount_percent}% (${round(discount_amount, 2)})",
            "tax": round(tax_amount, 2),
            "total_gross": round(total_before_insurance, 2),
            "insurance_paid": round(insurance_paid, 2),
            "patient_due": round(patient_out_of_pocket, 2)
        }
        
        logger.info(f"Successfully calculated invoice. Patient due: ${invoice_summary['patient_due']}")
        
        return {
            "statusCode": 200,
            "body": json.dumps(invoice_summary)
        }
        
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error calculating invoice.", "details": str(e)})
        }