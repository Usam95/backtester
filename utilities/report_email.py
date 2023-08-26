import smtplib
from report_email.message import EmailMessage
from utilities.credentials import *
import time

MAX_RETRIES = 5


class Email:
    def __init__(self):
        self.msg = None
        self.smtp = None
        self.init_msg()
        self.init_sender()

    def init_sender(self):
        self.smtp = smtplib.SMTP('smtp.gmail.com:587')
        self.smtp.ehlo()
        self.smtp.starttls()
        self.smtp.login(email_user, email_password)

    def close_sender(self):
        if self.smtp:
            self.smtp.quit()

    def init_msg(self):
        self.msg = EmailMessage()
        self.msg['Subject'] = 'INFO: binance bot.'
        self.msg['From'] = email_user
        self.msg['To'] = email_user

    def set_email_text(self, text):
        self.msg.set_content(text)

    def send_email(self, text):
        for _ in range(MAX_RETRIES):
            try:
                self.msg.set_content(text)
                self.smtp.send_message(self.msg)
                return  # Successful send, exit the function
            except smtplib.SMTPException as e:
                print(f"Failed to send email. Error: {e}. Retrying...")
                self.close_sender()
                self.init_sender()
            time.sleep(5)
        print(f"Failed to send email after {MAX_RETRIES} attempts.")