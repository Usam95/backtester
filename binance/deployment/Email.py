
import os
import smtplib
import imghdr
from email.message import EmailMessage

import sys
sys.path.append('../utilities')
from credentials import *

class Email:
    def __init__(self):
        self.msg = None
        self.smtp = None
        self.init_msg()
        self.init_sender()
    def init_sender(self):
        self.smtp = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        self.smtp.login(email_user, email_password)

    def init_msg(self):
        self.msg = EmailMessage()
        self.msg['Subject'] = 'INFO: binance bot.'
        self.msg['From'] = email_user
        self.msg['To'] = email_user

    def set_email_text(self, text):
        self.msg.set_content(text)

    def send_email(self, text):
        self.msg.set_content(text)
        self.smtp.send_message(self.msg)


if __name__ == "__main__":
    email = Email()
    email.send_email("Hallo, das bin ich!")