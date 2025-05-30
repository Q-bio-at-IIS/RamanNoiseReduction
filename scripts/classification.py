import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool
import argparse, os, pathlib, random
from datetime import datetime

import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append(".")
from util import argsPrint
from noiseReductionMethodology import preprocessing, Raman_model

def makeDataSet(DATA, seed=0):
   random.seed(seed)
   isTest = np.zeros(DATA.shape[0])
   idxList = [np.where(DATA["label"] == label)[0] for label in np.unique(DATA["label"])]
   for idx in idxList:
      idx_test = np.array(random.sample(list(idx), int(len(idx)/6)))
      isTest[idx_test] = 1
   test = np.where(isTest == 1)[0]
   train = np.where(isTest == 0)[0]

   return DATA.iloc[train, :].reset_index(drop=True), DATA.iloc[test, :].reset_index(drop=True)


def classification_method(DATA, method="MLP", itr=100, isPrint=True):
   accList = []
   for i in range(itr):
      TRAIN_DATA, TEST_DATA = makeDataSet(DATA, seed=i)

      if method == "LDA":
         model = LDA(store_covariance=True)
      elif method == "MLP":
         model = MLPClassifier(random_state=0)
      elif method == "SVMlinear":
         model = SVC(C=1.0, kernel="linear")
      elif method == "SVMrbf":
         model = SVC(C=1.0, kernel="rbf")
      else:
         raise ValueError("Please set available methods")

      model.fit(TRAIN_DATA.values[:, :-1], TRAIN_DATA["label"])
      acc = model.score(TEST_DATA.values[:, :-1], TEST_DATA["label"]) * 100
      accList.append(acc)

      if isPrint:
         if i % (itr // 10) == 0:
               print(f".", end="")

   mean_acc = np.mean(accList)
   sd_acc = np.std(accList)

   if isPrint:
      print("")
      print(f"accuracy: {mean_acc:.2f} ± {sd_acc:.2f} % ({np.min(accList):.2f} ~ {np.max(accList):.2f} %)")

   return accList


def classification_wSelectedData(values):
   GROUP, outputDir, numList, thr, classificationMethod, mode, cutMode, seed, shuffleNum, itr, suffix = values

   accList_numEffects = []
   cutModeStr = cutMode[0].upper() + cutMode[1:]

   if thr % 1 == 0:
      thrStr = int(thr)
   else:
      thrStr = f"{thr:.1f}".replace(".", "")
   if suffix:
      path = f"{outputDir}/ACCLIST_NUMEFFECTS_{mode}_cut{cutModeStr}{thrStr}_{suffix}.csv"
   else:   
      path = f"{outputDir}/ACCLIST_NUMEFFECTS_{mode}_cut{cutModeStr}{thrStr}.csv"

   if os.path.exists(path):
      print(f"{thr}*", end="", flush=True)
   else:
      print(f"{thr}>", end="", flush=True)
      for n in numList:
         print(".", end="", flush=True)
         accList_n = []
         np.random.seed(seed)
         for j in range(shuffleNum):
               INPUTDATA = pd.DataFrame([])
               for OUT in GROUP:
                  idxList = list(np.arange(OUT.shape[0]))
                  np.random.shuffle(idxList)
                  INPUTDATA = pd.concat([INPUTDATA, OUT.iloc[idxList[:n], :]], axis=0)
                  
               INPUTDATA = INPUTDATA.reset_index(drop=True)
               
               raman_model_ = Raman_model(INPUTDATA, cutRange=thr, cutMode=cutMode)
               raman_model_.calcTransformation()

               if mode == "PCA":
                  DATA = raman_model_.RAMAN_PCA
               elif mode == "NRM":
                  DATA = raman_model_.RAMAN_NRM

               accList_n.append(classification_method(DATA, method=classificationMethod, itr=itr, isPrint=False))

         accList_numEffects.append(accList_n)
         
      ACCLIST_NUMEFFECTS = pd.DataFrame(np.mean(accList_numEffects, axis=2).T, 
                                       columns=[f"n={n}" for n in numList])
      ACCLIST_NUMEFFECTS.to_csv(path, index=None)


def main():
   parser = argparse.ArgumentParser(description="Comparison")
   parser.add_argument("--outdir", "-o", type=str, required=True)
   parser.add_argument("--seed", "-s", type=int, default=0)
   parser.add_argument("--itr", "-itr", type=int, default=100)
   parser.add_argument("--shuffleNum", "-s_n", type=int, default=100)
   parser.add_argument("--pNum", "-p", type=int, default=4)
   parser.add_argument("--cutMode", "-c_mode", type=str, default="dim")
   parser.add_argument("--cutRange_start", "-start", type=float, default=5)
   parser.add_argument("--cutRange_end", "-end", type=float, default=120)
   parser.add_argument("--cutRange_step", "-step", type=float, default=5)
   parser.add_argument("--classificationMethod", "-method", type=str, default="LDA")   
   parser.add_argument("--mode", "-mode", type=str, default="PCA")   
   parser.add_argument("--pathRaman", "-p_r", type=str, default="../data/RAMAN_FINGERPRINT.csv")
   parser.add_argument("--suffix", "-suffix", type=str, default="")

   args = parser.parse_args()

   logs = sys.argv
   print(" ".join(logs))

   argsPrint(args)

   now = datetime.now()
   now_str = now.strftime("%Y/%m/%d %H:%M")
   print(f">>>>>>>>>> START {now_str}")

   RAMAN = pd.read_csv(pathlib.Path(args.pathRaman).resolve())
   RAMAN_PROCESSED = preprocessing(RAMAN)
   GROUP = [RAMAN_PROCESSED[RAMAN_PROCESSED["label"] == i] for i in range(len(RAMAN_PROCESSED["label"].unique()))]

   numList = np.arange(6, RAMAN_PROCESSED.groupby("label").count().values[:, 0].min() + 1, 6)
   thrList = np.arange(args.cutRange_start, args.cutRange_end + args.cutRange_step / 2, args.cutRange_step)[::-1]

   os.makedirs(args.outdir, exist_ok=True)
   valueList = [(GROUP, args.outdir, numList, thr, args.classificationMethod, args.mode, args.cutMode, args.seed, args.shuffleNum, args.itr, args.suffix)
                for thr in thrList]
             
   p = Pool(args.pNum)
   results = p.map(classification_wSelectedData, valueList)

   now = datetime.now()
   now_str = now.strftime("%Y/%m/%d %H:%M")
   print("")
   print(f">>>>>>>>>> END {now_str}")


if __name__ == "__main__":
   main()
