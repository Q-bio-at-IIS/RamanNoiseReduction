import numpy as np
import pandas as pd
from scipy import signal, linalg

def raman_smoothing(ALLDATA):
   OUT = pd.DataFrame(signal.savgol_filter(ALLDATA.values[:, :-1], polyorder=3, window_length=5, axis=1))
   OUT["label"] = ALLDATA["label"].values
   return OUT

def zTransform(ALLDATA):
   OUT = pd.DataFrame((ALLDATA.values[:, :-1] - np.mean(ALLDATA.values[:, :-1], axis=1).reshape(-1, 1)) / np.std(ALLDATA.values[:, :-1], axis=1).reshape(-1, 1))
   OUT["label"] = ALLDATA["label"].values
   return OUT

def preprocessing(ALLDATA):
   OUT = raman_smoothing(ALLDATA)
   OUT = zTransform(OUT)
   return OUT

class Raman_model():
   def __init__(self, RAMAN, cutRange=98, cutMode="percent"):
      self.RAMAN = RAMAN
      self.cutRange = cutRange
      self.cutMode = cutMode
      self.N = RAMAN.shape[0]
      self.d = RAMAN.shape[1] - 1
      self.data = self.RAMAN.values[:, :-1].T.astype(float)
      self.mean = np.mean(self.data, axis=1).reshape(-1, 1)

   def calcH(self, lam):
      lambda_ = np.zeros(len(lam))
      lambda_[lam > 0] = ((self.N - 1) * lam[lam > 0]) ** (-1 / 2)
      return np.dot(np.dot(self.data - self.mean, self.U), np.diag(lambda_))

   def calcX(self, lam, k):
      return np.sqrt(self.N - 1) * np.dot(np.diag(lam[:k]) ** (1 / 2), self.U[:, :k].T)

   def calcTransformation(self):
      data = self.RAMAN.values[:, :-1].T.astype(float)
      mean = np.mean(data, axis=1).reshape(-1, 1)
      S_D = np.dot((data - mean).T, (data - mean)) / (data.shape[1] - 1)

      lambda_hat, U = linalg.eigh(S_D)
      lambda_hat = np.abs(lambda_hat.real)
      U = U[:, np.argsort(lambda_hat)[::-1]]
      lambda_hat = lambda_hat[np.argsort(lambda_hat)[::-1]]

      lambda_tilde = np.array([lambda_hat[i] - (np.trace(S_D) - np.sum(lambda_hat[:i + 1])) / (data.shape[1] - 1 - (i + 1))
                              for i in range(np.min([data.shape[0], data.shape[1] - 2]))])

      lambda_tilde = np.hstack([lambda_tilde, np.zeros(len(lambda_hat) - len(lambda_tilde))])

      self.U = U
      self.lambda_hat = lambda_hat
      self.lambda_tilde = lambda_tilde
      self.explained_variance_ratio_hat = lambda_hat / lambda_hat.sum()
      self.explained_variance_ratio_tilde = lambda_tilde / lambda_tilde.sum()

      if self.cutMode == "dim":
         self.k_hat = self.cutRange
         self.k_tilde = self.cutRange
         self.percent_hat = self.explained_variance_ratio_hat[:self.k_hat].sum() * 100
         self.percent_tilde = self.explained_variance_ratio_tilde[:self.k_tilde].sum() * 100
      elif self.cutMode == "percent":
         self.percent_hat = self.cutRange
         self.percent_tilde = self.cutRange
         if self.cutRange == 100:
            self.k_hat = self.N
            self.k_tilde = self.N
         else:
            self.k_hat = np.where(np.cumsum(self.explained_variance_ratio_hat) * 100 > self.cutRange)[0][0] + 1
            self.k_tilde = np.where(np.cumsum(self.explained_variance_ratio_tilde) * 100 > self.cutRange)[0][0] + 1
      elif self.cutMode == "percent_fixedDim":
         self.percent_hat = self.cutRange
         if self.cutRange == 100:
            self.k_hat = self.N
            self.k_tilde = self.N
            self.percent_tilde = self.cutRange
         else:
            self.k_hat = np.where(np.cumsum(self.explained_variance_ratio_hat) * 100 > self.cutRange)[0][0] + 1
            self.k_tilde = self.k_hat
            self.percent_tilde = self.explained_variance_ratio_tilde[:self.k_tilde].sum() * 100
      else:
         raise ValueError("Please set possible mode as `cutMode`")
         
      self.H_hat = self.calcH(lambda_hat)[:, :self.k_hat]
      self.H_tilde = self.calcH(lambda_tilde)[:, :self.k_tilde]

      self.X_hat = self.calcX(self.lambda_hat, self.k_hat)
      OUT = pd.DataFrame(self.X_hat.T)
      OUT["label"] = self.RAMAN["label"]
      self.RAMAN_PCA = OUT

      self.X_tilde = self.calcX(self.lambda_tilde, self.k_tilde)
      OUT = pd.DataFrame(self.X_tilde.T)
      OUT["label"] = self.RAMAN["label"]
      self.RAMAN_NRM = OUT
