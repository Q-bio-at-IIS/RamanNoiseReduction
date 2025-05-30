import numpy as np
import pandas as pd

def returnValues(DATA):
   OUT = DATA.copy()
   picked = np.array([str(data).isdigit() for data in DATA.columns])
   OUT = OUT.iloc[:, np.where(picked == True)[0]]
   return OUT.values.astype(float)


def argsPrint(p, bar=30):
   print('-' * bar)
   args = [(i, getattr(p, i)) for i in dir(p) if not '_' in i[0]]
   for i, j in args:
      if isinstance(j, list):
         print('{0}[{1}]:'.format(i, len(j)))
         [print('\t{}'.format(k)) for k in j]
      else:
         print('{0}:\t{1}'.format(i, j))

   print('-' * bar)
   