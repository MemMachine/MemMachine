"""
Email utility for sending verification emails
Supports multiple email providers: SMTP, SendGrid, Mailgun, AWS SES
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class EmailService:
    """
    Email service for sending verification emails
    
    Supports multiple providers:
    - SMTP (Gmail, Outlook, custom SMTP server)
    - SendGrid (API key required)
    - Mailgun (API key required)
    - AWS SES (credentials required)
    """
    
    def __init__(self):
        """Initialize email service with configuration from environment"""
        self.provider = os.getenv("EMAIL_PROVIDER", "smtp").lower()
        
        # SMTP Configuration
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        
        # Sender configuration
        self.from_email = os.getenv("FROM_EMAIL", "noreply@HippoSync.com")
        self.from_name = os.getenv("FROM_NAME", "HippoSync")
        
        # App configuration
        self.app_url = os.getenv("APP_URL", "http://localhost:5173")
        self.app_name = os.getenv("APP_NAME", "HippoSync")
        
        # SendGrid
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY", "")
        
        # Mailgun
        self.mailgun_api_key = os.getenv("MAILGUN_API_KEY", "")
        self.mailgun_domain = os.getenv("MAILGUN_DOMAIN", "")
    
    def send_verification_email(
        self,
        to_email: str,
        verification_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Send verification email to user
        
        Args:
            to_email: Recipient email address
            verification_token: Unique verification token
            user_name: Optional user name for personalization
        
        Returns:
            True if email sent successfully, False otherwise
        """
        verification_url = f"{settings.APP_URL}:8001/auth/verify-email?token={token}"
        
        # Email subject
        subject = f"Verify your {self.app_name} account"
        
        # Email body (HTML)
        html_body = self._create_verification_email_html(
            verification_link=verification_link,
            user_name=user_name
        )
        
        # Email body (Plain text fallback)
        text_body = self._create_verification_email_text(
            verification_link=verification_link,
            user_name=user_name
        )
        
        try:
            if self.provider == "smtp":
                return self._send_via_smtp(to_email, subject, html_body, text_body)
            elif self.provider == "sendgrid":
                return self._send_via_sendgrid(to_email, subject, html_body, text_body)
            elif self.provider == "mailgun":
                return self._send_via_mailgun(to_email, subject, html_body, text_body)
            else:
                logger.error(f"Unsupported email provider: {self.provider}")
                return False
        except Exception as e:
            logger.error(f"Failed to send verification email: {e}")
            return False
    
    def _send_via_smtp(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str
    ) -> bool:
        """Send email via SMTP"""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email
            
            # Attach text and HTML parts
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()
                
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"✅ Verification email sent to {to_email} via SMTP")
            return True
        
        except Exception as e:
            logger.error(f"SMTP send failed: {e}")
            return False
    
    def _send_via_sendgrid(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str
    ) -> bool:
        """Send email via SendGrid API"""
        try:
            import sendgrid
            from sendgrid.helpers.mail import Mail, Email, To, Content
            
            sg = sendgrid.SendGridAPIClient(api_key=self.sendgrid_api_key)
            
            from_email = Email(self.from_email, self.from_name)
            to_email = To(to_email)
            content = Content("text/html", html_body)
            
            mail = Mail(from_email, to_email, subject, content)
            
            response = sg.client.mail.send.post(request_body=mail.get())
            
            logger.info(f"✅ Verification email sent to {to_email} via SendGrid")
            return response.status_code in [200, 201, 202]
        
        except Exception as e:
            logger.error(f"SendGrid send failed: {e}")
            return False
    
    def _send_via_mailgun(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str
    ) -> bool:
        """Send email via Mailgun API"""
        try:
            import requests
            
            response = requests.post(
                f"https://api.mailgun.net/v3/{self.mailgun_domain}/messages",
                auth=("api", self.mailgun_api_key),
                data={
                    "from": f"{self.from_name} <{self.from_email}>",
                    "to": to_email,
                    "subject": subject,
                    "text": text_body,
                    "html": html_body
                }
            )
            
            logger.info(f"✅ Verification email sent to {to_email} via Mailgun")
            return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Mailgun send failed: {e}")
            return False
    
    def _create_verification_email_html(
        self,
        verification_link: str,
        user_name: Optional[str] = None
    ) -> str:
        """Create HTML email body"""
        greeting = f"Hi {user_name}" if user_name else "Hi there"
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 600px; width: 100%; background: #ffffff; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 20px; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 16px 0 0;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: bold;">
                                🔐 {self.app_name}
                            </h1>
                        </td>
                    </tr>
                    
                    <!-- Body -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 20px; color: #1a202c; font-size: 24px; font-weight: 600;">
                                Verify Your Email Address
                            </h2>
                            
                            <p style="margin: 0 0 20px; color: #4a5568; font-size: 16px; line-height: 1.6;">
                                {greeting},
                            </p>
                            
                            <p style="margin: 0 0 20px; color: #4a5568; font-size: 16px; line-height: 1.6;">
                                Thank you for signing up for {self.app_name}! We're excited to have you on board.
                            </p>
                            
                            <p style="margin: 0 0 30px; color: #4a5568; font-size: 16px; line-height: 1.6;">
                                To complete your registration and start using your account, please verify your email address by clicking the button below:
                            </p>
                            
                            <!-- Button -->
                            <table role="presentation" style="margin: 0 auto;">
                                <tr>
                                    <td style="border-radius: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                        <a href="{verification_link}" target="_blank" style="display: inline-block; padding: 16px 48px; color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; border-radius: 12px;">
                                            ✓ Verify Email Address
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 30px 0 20px; color: #718096; font-size: 14px; line-height: 1.6;">
                                Or copy and paste this link into your browser:
                            </p>
                            
                            <p style="margin: 0 0 30px; padding: 12px; background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; color: #667eea; font-size: 13px; word-break: break-all;">
                                {verification_link}
                            </p>
                            
                            <p style="margin: 0 0 10px; color: #718096; font-size: 14px; line-height: 1.6;">
                                <strong>⏰ This link will expire in 24 hours.</strong>
                            </p>
                            
                            <p style="margin: 0; color: #718096; font-size: 14px; line-height: 1.6;">
                                If you didn't create an account with {self.app_name}, you can safely ignore this email.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 30px 40px; text-align: center; background: #f7fafc; border-radius: 0 0 16px 16px; border-top: 1px solid #e2e8f0;">
                            <p style="margin: 0 0 10px; color: #a0aec0; font-size: 12px;">
                                © 2024 {self.app_name}. All rights reserved.
                            </p>
                            <p style="margin: 0; color: #a0aec0; font-size: 12px;">
                                Powered by MemMachine
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    def _create_verification_email_text(
        self,
        verification_link: str,
        user_name: Optional[str] = None
    ) -> str:
        """Create plain text email body"""
        greeting = f"Hi {user_name}" if user_name else "Hi there"
        
        return f"""
{greeting},

Thank you for signing up for {self.app_name}! We're excited to have you on board.

To complete your registration and start using your account, please verify your email address by clicking the link below:

{verification_link}

This link will expire in 24 hours.

If you didn't create an account with {self.app_name}, you can safely ignore this email.

---
© 2024 {self.app_name}. All rights reserved.
Powered by MemMachine
"""


# Singleton instance
email_service = EmailService()