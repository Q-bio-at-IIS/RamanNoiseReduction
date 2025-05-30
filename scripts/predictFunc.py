import numpy as np
import pandas as pd
import random
import math
import itertools

from sklearn.cross_decomposition import PLSRegression

def predict_plsr(INPUT, TARGET, testIdx=0, n_components=7, max_iter=500):
   X = INPUT[INPUT["label"] != testIdx].values[:, :-1]
   Y = TARGET[TARGET["label"] != testIdx].values[:, :-1]

   if n_components == 0:
      n_components = np.min([X.shape[0] - 2, X.shape[1] - 1]) 

   testX = INPUT[INPUT["label"] == testIdx].values[:, :-1]
   testY = TARGET[TARGET["label"] == testIdx].values[:, :-1]
   
   plsr = PLSRegression(n_components=n_components, scale=False, max_iter=max_iter).fit(X, Y)
   predict = (np.dot(testY, np.linalg.pinv(plsr.coef_).T) * plsr._x_std) + plsr._x_mean
   
   return predict, np.sum((predict - testX)**2)


def calcPrediction(INPUT, TARGET, n_components=7, max_iter=500):
   PREDICT = pd.DataFrame([])
   for i in range(INPUT["label"].values.max() + 1):
      predict, diff = predict_plsr(INPUT=INPUT, TARGET=TARGET, testIdx=i, n_components=n_components, max_iter=max_iter)
      PREDICT_i = pd.DataFrame(predict)
      PREDICT_i["diff"] = diff
      PREDICT_i["label"] = i
      PREDICT = pd.concat([PREDICT, PREDICT_i], ignore_index=True)
   return PREDICT


def permutation_test(INPUT, TARGET, loopNum=10000, n_components=7, isLog=True):
   shuffled = []
   clusterNum = len(np.unique(INPUT["label"]))
   
   if clusterNum <= 5:
      if loopNum > math.factorial(clusterNum):
         isAllComb = True
         loopNum = math.factorial(clusterNum)
         permList = list(itertools.permutations(np.arange(clusterNum)))
      else:
         isAllComb = False
   else:
      isAllComb = False

   random.seed(0)
   for i in range(loopNum):
      if isAllComb:
         nList = list(permList[i])
      else:
         nList = random.sample(list(range(clusterNum)), clusterNum)

      INPUT_i = INPUT.iloc[nList].copy()
      INPUT_i["label"] = np.arange(INPUT_i.shape[0])
      
      PREDICT = calcPrediction(INPUT_i, TARGET, n_components=n_components)
      shuffled.append(PREDICT["diff"].sum())

      if loopNum > 10:
         if (i % int(loopNum/10) == 0) & (isLog):
            print(f"DONE {i}")
         
   return shuffled


def calcPermutationTest(INPUT, TARGET, loopNum=1000, default=None, mode="train", shuffleList=[], isLog=True, n_components=7):
   PREDICT = calcPrediction(INPUT, TARGET, n_components=n_components)

   shuffled = permutation_test(INPUT=INPUT, TARGET=TARGET, loopNum=loopNum, isLog=isLog, n_components=n_components)
   if default:
      ratio = PREDICT["diff"].sum() / default
   else:
      ratio = None

   return PREDICT, shuffled, ratio, PREDICT["diff"].sum()
