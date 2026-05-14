# services/parameter_service.py


# =========================================================
# PARAMETERS
# =========================================================

def get_parameters(call_category=None, call_type=None):

    return [

        {
            "key": "communication",
            "label": "Communication Skills",
            "icon": "💬",

            "parameters": [

                {
                    "key": "clarity",
                    "name": "Clarity",
                    "weight": 10,
                    "evaluation_guide": "Check whether agent communication is clear and understandable."
                },

                {
                    "key": "confidence",
                    "name": "Confidence",
                    "weight": 10,
                    "evaluation_guide": "Check whether agent speaks confidently."
                }
            ]
        },

        {
            "key": "resolution",
            "label": "Problem Resolution",
            "icon": "✅",

            "parameters": [

                {
                    "key": "issue_resolution",
                    "name": "Issue Resolution",
                    "weight": 15,
                    "evaluation_guide": "Check whether customer issue was resolved."
                },

                {
                    "key": "customer_satisfaction",
                    "name": "Customer Satisfaction",
                    "weight": 15,
                    "evaluation_guide": "Check whether customer appeared satisfied."
                }
            ]
        }
    ]


# =========================================================
# AGENT MAPPINGS
# =========================================================

def get_agent_mappings(call_category=None, call_type=None):

    return [

        {
            "agent_category": "Communication",
            "parameter_keys": [
                "clarity",
                "confidence"
            ],
            "weightage": 50
        },

        {
            "agent_category": "Resolution",
            "parameter_keys": [
                "issue_resolution",
                "customer_satisfaction"
            ],
            "weightage": 50
        }
    ]


# =========================================================
# AUTO WEIGHTAGES
# =========================================================

def get_auto_weightages(call_category=None, call_type=None):

    return {

        "communication": 50,
        "resolution": 50
    }