
import os
import sys

import inspect
import backtester_base

from backtester_base import VectorBacktesterBase
print(inspect.getfile(backtester_base))


#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "indicators")))


#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "indicators")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..")))
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import backtesting.set_project_path

#from bb_backtester import BBBacktester
#from backtester_base import VectorBacktesterBase
filepath = "../../../../hist_data/XRPUSDT/train/XRPUSDT.csv"
symbol = "XRPUSDT"
start = "2020-08-20"
end = "2020-11-20"
ptc = 0.00007
#bb = BBBacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)