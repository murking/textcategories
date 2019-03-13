import pandas as pd
import numpy as np
import jieba
import re
data_read = pd.read_csv("data/3ci.csv")
print(data_read.info())
