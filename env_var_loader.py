import os


def validate_email_env_vars():
    """
    Validate that required email environment variables are set
    
    Returns:
        tuple: (is_valid, missing_vars, warnings)
    """
    required_vars = ['SENDER_EMAIL', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET', 'GOOGLE_REFRESH_TOKEN']
    optional_vars = ['SENDER_NAME', 'SMTP_SERVER', 'SMTP_PORT', 'SMTP_USE_TLS', 'CC_EMAILS', 'BCC_EMAILS']
    
    missing_vars = []
    warnings = []
    
    # Check required variables
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    # Check optional variables and provide warnings
    if not os.getenv('SENDER_NAME'):
        warnings.append('SENDER_NAME not set, will use default display name')
    
    if not os.getenv('SMTP_SERVER'):
        warnings.append('SMTP_SERVER not set, will try to detect from email provider')
    
    is_valid = len(missing_vars) == 0
    
    return is_valid, missing_vars, warnings