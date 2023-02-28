
import os
import smtplib
import imghdr
from email.message import EmailMessage

import sys
sys.path.append('')
from utilities.credentials import *

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

    def init_msg(self):
        self.msg = EmailMessage()
        self.msg['Subject'] = 'INFO: binance bot.'
        self.msg['From'] = email_user
        self.msg['To'] = email_user

    def set_email_text(self, text):
        self.msg.set_content(text)

    def send_email(self, text):
        try:
            self.msg.set_content(text)
            self.smtp.send_message(self.msg)
        except Exception: # replace this with the appropriate SMTPLib exception

            # Overwrite the stale connection object with a new one
            # Then, re-attempt the smtp_operations() method (now that you have a fresh connection object instantiated).
            self.init_sender()
            self.send_email(text)


if __name__ == "__main__":
    email = Email()
    email.send_email("Hallo, das bin ich!")