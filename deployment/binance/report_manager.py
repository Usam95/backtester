# Standard library imports
import sys
import os
from datetime import datetime

# Update sys.path for local application imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '', '..', '..')))

# Local application/library specific imports
from utilities.report_email import Email
from utilities.logger import Logger


class ReportManager:
    def __init__(self):
       #self.email = Email()
        self.logger = Logger().get_logger()

    def log_trade_data(self, trading_data):
        # Construct the trade report string
        trade_report = "\n" + 100 * "-" + "\n"
        trade_report += "{} | {}\n".format(trading_data["time"], trading_data["trade_action"])
        trade_report += "Strategies: {}\n".format(trading_data["strategy_names"])
        trade_report += "Ticker: {}\n".format(trading_data["ticker"])
        trade_report += "{} | Base_Units = {} | Quote_Units = {} | Price = {} \n".format(trading_data["time"],
                                                                                         trading_data["base_units"],
                                                                                         trading_data["quote_units"],
                                                                                         trading_data["price"])
        trade_report += "{} | Profit = {} | CumProfits = {} \n".format(trading_data["time"],
                                                                       trading_data["real_profit"],
                                                                       trading_data["cumulative_profits"])
        trade_report += 100 * "-" + "\n"

        # Log the trade report
        self.logger.info(trade_report)

    def send_email_report(self, trading_data):
        now = datetime.now()
        # If it's the beginning of an hour divisible by 4, send the email
        if now.hour % 4 == 0 and now.minute == 0:
            # Enhanced email content for clarity and better presentation
            text = f"""Trading Report:
--------------------------------
Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
Action: {trading_data["trade_action"]}
Number of Trades: {trading_data["total_trades"]}
Profit: {trading_data["real_profit"]}
Cumulative Profits: {trading_data["cumulative_profits"]}
--------------------------------
Thank you for using our trading system.
"""
            self.email.send_email(text)



