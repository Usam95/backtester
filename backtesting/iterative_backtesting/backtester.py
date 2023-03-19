
import os
import sys

import inspect
import iterative_backtester_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))
import backtesting.set_project_path
#from iterative_backtester_base import VectorBacktesterBase
print(inspect.getfile(iterative_backtester_base))

from backtester_base import VectorBacktesterBase
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "indicators")))


#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "indicators")))

#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from bb_backtester import BBBacktester
#from backtester_base import VectorBacktesterBase

filepath = os.path.abspath(os.path.join("..", "..", "hist_data", "XRPUSDT", "XRPUSDT.parquet.gzip"))
print(filepath)
symbol = "XRPUSDT"
start = "2020-08-20"
end = "2020-11-20"
ptc = 0.00007
bb = BBBacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
bb.prepare_data(freq=15, window=20, dev=2)
print(bb.results.head())
print(bb.results.position.value_counts())
print("ok")