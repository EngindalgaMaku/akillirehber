"""Email service for sending notifications to users."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


def send_email(
    to_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None
) -> bool:
    """
    Send an email using SMTP.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body_text: Plain text email body
        body_html: HTML email body (optional)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    settings = get_settings()
    
    # Check if email is configured
    if not settings.smtp_host or not settings.smtp_from_email:
        logger.warning(
            "Email not configured. SMTP_HOST and SMTP_FROM_EMAIL "
            "must be set in environment variables."
        )
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = settings.smtp_from_email
        msg['To'] = to_email
        
        # Attach plain text version
        part1 = MIMEText(body_text, 'plain')
        msg.attach(part1)
        
        # Attach HTML version if provided
        if body_html:
            part2 = MIMEText(body_html, 'html')
            msg.attach(part2)
        
        # Connect to SMTP server and send
        with smtplib.SMTP(
            settings.smtp_host,
            settings.smtp_port
        ) as server:
            # Enable TLS if configured
            if settings.smtp_use_tls:
                server.starttls()
            
            # Login if credentials provided
            if settings.smtp_username and settings.smtp_password:
                server.login(
                    settings.smtp_username,
                    settings.smtp_password
                )
            
            # Send email
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False


def send_password_reset_email(
    to_email: str,
    user_name: str,
    temporary_password: str,
    expires_hours: int = 24
) -> bool:
    """
    Send password reset email with temporary password.
    
    Args:
        to_email: User's email address
        user_name: User's full name
        temporary_password: Temporary password to send
        expires_hours: Hours until password expires (default: 24)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    subject = "Şifre Sıfırlama - RAG Chatbot"
    
    # Plain text version
    body_text = f"""
Merhaba {user_name},

Yönetici tarafından şifreniz sıfırlandı.

Geçici Şifreniz: {temporary_password}

Bu geçici şifre {expires_hours} saat boyunca geçerlidir.

Lütfen bu geçici şifre ile giriş yapın ve ardından şifrenizi değiştirin.

Güvenlik nedeniyle, bu e-postayı kimseyle paylaşmayın.

Saygılarımızla,
RAG Chatbot Ekibi
"""
    
    # HTML version
    body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #4F46E5;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px 5px 0 0;
        }}
        .content {{
            background-color: #f9f9f9;
            padding: 30px;
            border: 1px solid #ddd;
            border-radius: 0 0 5px 5px;
        }}
        .password-box {{
            background-color: #fff;
            border: 2px solid #4F46E5;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            letter-spacing: 2px;
            border-radius: 5px;
        }}
        .warning {{
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 10px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Şifre Sıfırlama</h1>
        </div>
        <div class="content">
            <p>Merhaba <strong>{user_name}</strong>,</p>
            
            <p>Yönetici tarafından şifreniz sıfırlandı.</p>
            
            <div class="password-box">
                {temporary_password}
            </div>
            
            <p>Bu geçici şifre <strong>{expires_hours} saat</strong> boyunca geçerlidir.</p>
            
            <p>Lütfen bu geçici şifre ile giriş yapın ve ardından şifrenizi değiştirin.</p>
            
            <div class="warning">
                <strong>⚠️ Güvenlik Uyarısı:</strong> Bu e-postayı kimseyle paylaşmayın.
            </div>
            
            <div class="footer">
                <p>Saygılarımızla,<br>RAG Chatbot Ekibi</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return send_email(to_email, subject, body_text, body_html)
