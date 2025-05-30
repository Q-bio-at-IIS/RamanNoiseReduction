import numpy as np
import pandas as pd
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score


class LDA_model():
   def __init__(self):
      pass

   def fit(self, DATA):
      lda = LDA(store_covariance=True).fit(DATA.values[:, :-1], DATA["label"].values)
      scalings = lda.scalings_
      scalings = scalings / np.sqrt(np.sum(scalings ** 2, axis=0))

      self.scalings = scalings
      self.lda = lda

   def transform(self, DATA):
      OUT = pd.DataFrame(np.dot(DATA.values[:, :-1], self.scalings))
      OUT["label"] = DATA["label"]
      return OUT

   def fit_transform(self, DATA):
      self.fit(DATA)
      OUT = self.transform(DATA)
      return OUT

   def makeDataSetIdx(self, DATA, seed=0):
      random.seed(seed)
      isTest = np.zeros(DATA.shape[0])
      idxList = [np.where(DATA["label"] == label)[0] for label in np.unique(DATA["label"])]
      for idx in idxList:
         idx_test = np.array(random.sample(list(idx), int(len(idx)/6)))
         isTest[idx_test] = 1
      self.test = np.where(isTest == 1)[0]
      self.train = np.where(isTest == 0)[0]

   def makeDataSet(self, DATA, seed=0):
      self.makeDataSetIdx(DATA, seed)
      return DATA.iloc[self.train, :].reset_index(drop=True), DATA.iloc[self.test, :].reset_index(drop=True)

   def prediction_LDA(self, DATA, seed=0, isMatrix=False):
      TRAIN, TEST = self.makeDataSet(DATA, seed)
      TRAIN_OUT = self.fit_transform(TRAIN)
      TEST_OUT = self.transform(TEST)

      lda_predict = np.argmax(self.lda.decision_function(TEST.values[:, :-1]), axis=1)
      TEST_OUT["predict"] = lda_predict
      
      accuracy = accuracy_score(TEST["label"].values, TEST_OUT["predict"].values) * 100

      if isMatrix:
         labelMax = TEST["label"].max()
         matrix= np.vstack([[(TEST_OUT[TEST_OUT["label"] == i]["predict"] == j).sum() for j in range(labelMax + 1)]
                           for i in range(labelMax + 1)])
         matrix = matrix / np.sum(matrix, axis=1)
         return TRAIN_OUT, TEST_OUT, accuracy, matrix
      else:
         return TRAIN_OUT, TEST_OUT, accuracy

   def clustering_LDA(self, ALLDATA, itr=100, isPrint=True, isMatrix=False):
      accList = []
      for i in range(itr):
         lda_model = LDA_model()
         if isMatrix:
            TRAIN_OUT, TEST_OUT, acc, matrix = lda_model.prediction_LDA(ALLDATA, seed=i, isMatrix=True)
            if i == 0:
               matrix_out = matrix / itr
            else:
               matrix_out += matrix / itr
         else:
            TRAIN_OUT, TEST_OUT, acc = lda_model.prediction_LDA(ALLDATA, seed=i, isMatrix=False)
         accList.append(acc)

         if isPrint:
            if i % (itr // 10) == 0:
               print(f".", end="")

      mean_acc = np.mean(accList)
      sd_acc = np.std(accList)

      if isPrint:
         print("")
         print(f"accuracy: {mean_acc:.2f} ± {sd_acc:.2f} % ({np.min(accList):.2f} ~ {np.max(accList):.2f} %)")
      
      if isMatrix:
         return accList, matrix_out
      else:
         return accList
