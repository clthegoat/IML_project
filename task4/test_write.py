import numpy as np  
from utils import write_preds

a=np.arange(10)
filepath = 'pred.txt'
write_preds(a, filepath)
