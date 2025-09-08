#!/bin/bash

# Email Configuration Setup Script
# Run this script to set up email environment variables

echo "🔧 Corrosion Monitoring System - Email Setup"
echo "=============================================="
echo ""

# Function to prompt for input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local varname="$3"
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " input
        export $varname="${input:-$default}"
    else
        read -p "$prompt: " input
        export $varname="$input"
    fi
}

echo "1. Email Provider Selection"
echo "   a) Gmail"
echo "   b) Outlook/Hotmail"
echo "   c) Yahoo"
echo "   d) Corporate Email Server"
echo ""

read -p "Select your email provider (a/b/c/d): " provider_choice

case $provider_choice in
    a|A)
        echo "📧 Configuring for Gmail..."
        echo "Note: Gmail now requires OAuth 2.0. You need a Client ID, Client Secret, and Refresh Token."
        echo "See: https://developers.google.com/identity/protocols/oauth2 for setup instructions."
        prompt_with_default "Gmail address" "" "SENDER_EMAIL"
        prompt_with_default "Google Client ID" "" "GOOGLE_CLIENT_ID"
        prompt_with_default "Google Client Secret" "" "GOOGLE_CLIENT_SECRET"
        prompt_with_default "Google Refresh Token" "" "GOOGLE_REFRESH_TOKEN"
        export SMTP_SERVER="smtp.gmail.com"
        export SMTP_PORT="587"
        export SMTP_USE_TLS="true"
        ;;
    b|B)
        echo "📧 Configuring for Outlook..."
        prompt_with_default "Outlook email address" "" "SENDER_EMAIL"
        prompt_password "Outlook password" "SENDER_PASSWORD"
        export SMTP_SERVER="smtp-mail.outlook.com"
        export SMTP_PORT="587"
        export SMTP_USE_TLS="true"
        ;;
    c|C)
        echo "📧 Configuring for Yahoo..."
        prompt_with_default "Yahoo email address" "" "SENDER_EMAIL"
        prompt_password "Yahoo App Password" "SENDER_PASSWORD"
        export SMTP_SERVER="smtp.mail.yahoo.com"
        export SMTP_PORT="587"
        export SMTP_USE_TLS="true"
        ;;
    d|D)
        echo "🏢 Configuring for Corporate Email..."
        prompt_with_default "Email address" "" "SENDER_EMAIL"
        prompt_password "Email password" "SENDER_PASSWORD"
        prompt_with_default "SMTP Server" "mail.company.com" "SMTP_SERVER"
        prompt_with_default "SMTP Port" "587" "SMTP_PORT"
        prompt_with_default "Use TLS (true/false)" "true" "SMTP_USE_TLS"
        ;;
    *)
        echo "❌ Invalid selection. Exiting."
        exit 1
        ;;
esac

echo ""
echo "2. Additional Configuration"

prompt_with_default "Display name for emails" "Corrosion Monitoring System" "SENDER_NAME"
prompt_with_default "CC recipients (comma-separated, optional)" "" "CC_EMAILS"
prompt_with_default "BCC recipients (comma-separated, optional)" "" "BCC_EMAILS"

echo ""
echo "3. Creating Environment File"

# Create .env file
cat > .env << EOF
# Corrosion Monitoring System - Email Configuration
# Generated on $(date)

# Email credentials (required)
SENDER_EMAIL=$SENDER_EMAIL
$(if [ "$provider_choice" = "a" ] || [ "$provider_choice" = "A" ]; then
    echo "GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID"
    echo "GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET"
    echo "GOOGLE_REFRESH_TOKEN=$GOOGLE_REFRESH_TOKEN"
else
    echo "SENDER_PASSWORD=$SENDER_PASSWORD"
fi)
SENDER_NAME=$SENDER_NAME

# SMTP Configuration
SMTP_SERVER=$SMTP_SERVER
SMTP_PORT=$SMTP_PORT
SMTP_USE_TLS=$SMTP_USE_TLS

# Additional recipients (optional)
CC_EMAILS=$CC_EMAILS
BCC_EMAILS=$BCC_EMAILS
EOF

echo "✅ Environment file (.env) created successfully!"
echo ""

# Create systemd service file template (for Linux production)
cat > corrosion-monitor.service << EOF
[Unit]
Description=Corrosion Monitoring System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/corrosion-monitor
EnvironmentFile=/opt/corrosion-monitor/.env
ExecStart=/usr/local/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Systemd service file (corrosion-monitor.service) created!"
echo ""

# Create Docker environment file
cat > docker.env << EOF
# Docker Environment Variables for Corrosion Monitoring System
SENDER_EMAIL=$SENDER_EMAIL
$(if [ "$provider_choice" = "a" ] || [ "$provider_choice" = "A" ]; then
    echo "GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID"
    echo "GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET"
    echo "GOOGLE_REFRESH_TOKEN=$GOOGLE_REFRESH_TOKEN"
else
    echo "SENDER_PASSWORD=$SENDER_PASSWORD"
fi)
SENDER_NAME=$SENDER_NAME
SMTP_SERVER=$SMTP_SERVER
SMTP_PORT=$SMTP_PORT
SMTP_USE_TLS=$SMTP_USE_TLS
CC_EMAILS=$CC_EMAILS
BCC_EMAILS=$BCC_EMAILS
EOF

echo "✅ Docker environment file (docker.env) created!"
echo ""

echo "4. Security Recommendations"
echo "🔒 IMPORTANT SECURITY NOTES:"
echo "   - The .env file contains sensitive credentials"
echo "   - Make sure to add .env to your .gitignore file"
echo "   - Set proper file permissions: chmod 600 .env"
echo "   - Consider using a secrets management system in production"
echo ""

# Set proper permissions
chmod 600 .env docker.env 2>/dev/null || true

# Create .gitignore entry
if [ -f .gitignore ]; then
    if ! grep -q "\.env" .gitignore; then
        echo ".env" >> .gitignore
        echo "docker.env" >> .gitignore
        echo "✅ Added .env files to .gitignore"
    fi
else
    echo ".env" > .gitignore
    echo "docker.env" >> .gitignore
    echo "✅ Created .gitignore with .env files"
fi

echo "5. Testing Configuration"
echo "To test your email configuration, run:"
echo "   python3 -c \"from email_config import test_email_configuration; test_email_configuration()\""
echo ""

echo "6. Loading Environment Variables"
echo "To load the environment variables in your current session:"
echo "   source .env"
echo "   # or"
echo "   export \$(cat .env | xargs)"
echo ""

echo "7. Production Deployment"
echo "For production deployment:"
echo "   - Copy .env to your production server"
echo "   - Use the systemd service file for Linux systems"
echo "   - Use docker.env for Docker deployments"
echo "   - Consider using Azure Key Vault, AWS Secrets Manager, or similar"
echo ""

echo "🎉 Email setup complete!"
echo "📧 You can now send production email alerts."
echo ""

# Test connection (optional)
read -p "Would you like to test the email configuration now? (y/n): " test_choice
if [[ $test_choice =~ ^[Yy]$ ]]; then
    echo "Testing email configuration..."
    python3 -c "
import os
import sys
sys.path.append('.')

# Load environment variables from .env file
with open('.env', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

from email_config import test_email_configuration
test_email_configuration()
" 2>/dev/null || echo "❌ Unable to test automatically. Make sure Python is installed and email_config.py is available."
fi