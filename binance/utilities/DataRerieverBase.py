import time
import pandas as pd
from typing import Optional
from datetime import datetime
import requests


class DataRetrieverBase:
    def __init__(self):
        self._base_url = "https://api.binance.com"

    def _make_request(self, endpoint: str, query_parameters: dict):
        response = requests.get(self._base_url + endpoint, params=query_parameters)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error while making request to %s: %s (status code = %s)",
                  endpoint, response.json(), response.status_code)
            return None

    def _get_symbols(self) -> list[str]:
        params = dict()
        endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        data = self._make_request(endpoint, params)
        symbols = [x["symbol"] for x in data["symbols"]]
        return symbols

    def get_historical_data(self, symbol: str, interval: Optional[str] = "1m", start_time: Optional[int] = None,
                            end_time: Optional[int] = None, limit: Optional[int] = 1500):

        params = dict()
        params["symbol"] = symbol
        params["interval"] = interval
        params["limit"] = limit

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"

        try:
            # timeout should also be given as a parameter to the function
            raw_candles = self._make_request(endpoint, params)
        except requests.exceptions.ConnectionError:
            print('Connection error, Cooling down for 3 mins...')
            time.sleep(3 * 60)
            raw_candles = self._make_request(endpoint, params)

        except requests.exceptions.Timeout:
            print('Timeout, Cooling down for 3 mins...')
            time.sleep(3 * 60)
            raw_candles = self._make_request(endpoint, params)

        except requests.exceptions.ConnectionResetError:
            print('Connection reset by peer, Cooling down for 3 mins...')
            time.sleep(3 * 60)
            raw_candles = self._make_request(endpoint, params)

        candles = []

        if raw_candles is not None:
            for c in raw_candles:
                candles.append((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]),))
            return candles
        else:
            return None
        print(f"Loaded data of length: {len(candles)}")

    def ms_to_dt_utc(self, ms: int) -> datetime:
        return datetime.utcfromtimestamp(ms / 1000)

    def ms_to_dt_local(self, ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000)

    def get_historical_data_as_df(self, symbol, start_time, end_time, limit=1500):
        data = self.get_ticker_hist_data_for_period(symbol, start_time, end_time)
        df = pd.DataFrame(data, columns=['Timestamp', "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = df["Timestamp"].apply(lambda x: self.ms_to_dt_local(x))
        column_names = ["Open", "High", "Low", "Close", "Volume"]
        df = df.set_index('Date')
        df = df.reindex(columns=column_names)
        return df

    def get_ticker_hist_data_for_period(self, symbol, start_time, end_time, limit=1500):
        collection = []

        while start_time < end_time:
            print(f"In get_ticker_hist_data_for_period: {start_time=}, {end_time=}:")
            print(f"Calling: get_historical_data() function")
            data = self.get_historical_data(symbol, start_time=start_time, end_time=end_time, limit=limit)
            try:
                print("BINANCE" + " " + symbol + " : Collected " + str(len(data)) + " initial data from " + str(
                    self.ms_to_dt_local(data[0][0])) + " to " + str(self.ms_to_dt_local(data[-1][0])))
                start_time = int(data[-1][0] + 1000)
                collection += data
                time.sleep(0.01)
            except Exception as e:
                print(f"BINANCE no data was received for time period from {start_time} to {end_time}| {e}..")
                collection += data
                break
        return collection
