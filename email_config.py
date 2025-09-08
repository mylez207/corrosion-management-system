import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
import logging
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import streamlit as st

# Configure logging for email system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailConfig:
    """Email configuration for different providers"""
    
    PROVIDERS = {
        'gmail': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True
        },
        'outlook': {
            'smtp_server': 'smtp-mail.outlook.com',
            'smtp_port': 587,
            'use_tls': True
        },
        'yahoo': {
            'smtp_server': 'smtp.mail.yahoo.com',
            'smtp_port': 587,
            'use_tls': True
        },
        'corporate': {
            'smtp_server': os.getenv('SMTP_SERVER', 'mail.company.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
        }
    }
    
    def __init__(self, provider='corporate'):
        self.provider = provider
        self.config = self.PROVIDERS.get(provider, self.PROVIDERS['corporate'])
        
        # Email credentials from environment variables
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_name = os.getenv('SENDER_NAME', 'Corrosion Monitoring System')
        self.cc_emails = os.getenv('CC_EMAILS', '').split(',') if os.getenv('CC_EMAILS') else []
        self.bcc_emails = os.getenv('BCC_EMAILS', '').split(',') if os.getenv('BCC_EMAILS') else []
        
        # OAuth 2.0 credentials for Gmail
        if provider == 'gmail':
            self.client_id = os.getenv('GOOGLE_CLIENT_ID')
            self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
            self.refresh_token = os.getenv('GOOGLE_REFRESH_TOKEN')
    
    def validate_config(self):
        """Validate email configuration"""
        if not self.sender_email:
            raise ValueError("SENDER_EMAIL environment variable must be set")
        if self.provider == 'gmail' and not (self.client_id and self.client_secret and self.refresh_token):
            raise ValueError("GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN must be set for Gmail")
        return True

    def get_access_token(self):
        """Obtain OAuth 2.0 access token for Gmail"""
        if self.provider != 'gmail':
            return None
        creds = Credentials(
            None,
            refresh_token=self.refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=['https://www.googleapis.com/auth/gmail.send']
        )
        if creds.expired or not creds.valid:
            creds.refresh(Request())
        return creds.token

def send_email_notification(to_email, severity, additional_info=None):
    """
    Send email notification for corrosion alerts
    
    Args:
        to_email (str): Recipient email address
        severity (str): Severity level (Low, Moderate, High, Severe)
        additional_info (dict): Additional information to include in email
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Initialize email configuration
        email_config = EmailConfig(provider='gmail' if 'gmail.com' in os.getenv('SENDER_EMAIL', '').lower() else 'corporate')
        email_config.validate_config()
        
        # Validate recipient email
        if not to_email or "@" not in to_email or "." not in to_email:
            logger.error(f"Invalid email address: {to_email}")
            st.error("❌ Invalid email address format")
            return False
        
        # Create message
        message = MIMEMultipart("alternative")
        
        priority_map = {
            "Severe": "🔴 CRITICAL",
            "High": "🟠 HIGH PRIORITY", 
            "Moderate": "🟡 MEDIUM PRIORITY",
            "Low": "🟢 LOW PRIORITY"
        }
        
        priority = priority_map.get(severity, "")
        message["Subject"] = f"{priority}: Corrosion Alert - {severity} Severity Detected"
        message["From"] = f"{email_config.sender_name} <{email_config.sender_email}>"
        message["To"] = to_email
        
        if email_config.cc_emails:
            message["Cc"] = ", ".join(email_config.cc_emails)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_body = create_html_email_body(severity, current_time, additional_info)
        text_body = create_text_email_body(severity, current_time, additional_info)
        
        part1 = MIMEText(text_body, "plain")
        part2 = MIMEText(html_body, "html")
        message.attach(part1)
        message.attach(part2)
        
        # Create secure connection
        context = ssl.create_default_context()
        with smtplib.SMTP(email_config.config['smtp_server'], email_config.config['smtp_port']) as server:
            server.starttls(context=context)
            
            if email_config.provider == 'gmail':
                access_token = email_config.get_access_token()
                server.auth('XOAUTH2', lambda: f"user={email_config.sender_email}\1auth=Bearer {access_token}\1\1")
            else:
                server.login(email_config.sender_email, os.getenv('SENDER_PASSWORD'))
            
            recipients = [to_email] + email_config.cc_emails + email_config.bcc_emails
            recipients = [email for email in recipients if email.strip()]
            server.send_message(message, to_addrs=recipients)
            
        logger.info(f"Email sent successfully to {to_email} for {severity} severity alert")
        st.success("✅ Email notification sent successfully!")
        
        with st.expander("📧 Email Sent - Details"):
            st.write(f"**To:** {to_email}")
            if email_config.cc_emails:
                st.write(f"**CC:** {', '.join(email_config.cc_emails)}")
            st.write(f"**Subject:** {message['Subject']}")
            st.write(f"**Sent at:** {current_time}")
            
        return True
        
    except Exception as e:
        error_msg = f"❌ Failed to send email: {str(e)}"
        logger.error(f"Email sending failed: {str(e)}")
        st.error(error_msg)
        return False