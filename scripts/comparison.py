import numpy as np
import pandas as pd
from multiprocessing import Pool
import argparse, pathlib
from datetime import datetime

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

import sys
sys.path.append("")
from util import argsPrint
from analysis_pclda import LDA_model
from noiseReductionMethodology import Raman_model, preprocessing
from predictFunc import calcPermutationTest


def loadData(outdir, seed=0, dataNum=10, shuffleNum=100,
             path_raman="../data/RAMAN_FINGERPRINT.csv",
             path_trans="../data/TRANS_SPOMBE.csv",
             loopNum=1000, pNum=10, n_components=7,
             cutMode="percent", cutRange=98):

   values = []

   np.random.rand(seed)

   print("Loading seqInfo data ... ", end="", flush=True)
   RAMAN = pd.read_csv(path_raman)
   RAMAN = preprocessing(RAMAN)

   TRANSCRIPTOME = pd.read_csv(path_trans)
   print(f"done (Raman: {RAMAN.shape[1] - 1}, Transcriptome: {TRANSCRIPTOME.shape[1] - 1})")


   for shuffle_i in range(shuffleNum):
      RAMAN_PICKED = pd.DataFrame([])
      for label in range(RAMAN["label"].max() + 1):
         RAMAN_label = RAMAN[RAMAN["label"] == label].copy().reset_index(drop=True)
         idxList = np.arange(RAMAN_label.shape[0])
         np.random.shuffle(idxList)
         RAMAN_i = RAMAN_label.iloc[idxList[:np.min([dataNum, len(idxList)])], :]
         RAMAN_PICKED = pd.concat([RAMAN_PICKED, RAMAN_i], axis=0, ignore_index=True)


      raman_model = Raman_model(RAMAN_PICKED, cutRange=cutRange, cutMode=cutMode)
      raman_model.calcTransformation()

      ## PCA
      lda_model = LDA_model()
      PCA_LDA = lda_model.fit_transform(raman_model.RAMAN_PCA)
      MEAN_PCA_LDA = PCA_LDA.groupby("label").mean()
      MEAN_PCA_LDA["label"] = np.arange(MEAN_PCA_LDA.shape[0])

      ## NRM
      lda_model = LDA_model()
      NRM_LDA = lda_model.fit_transform(raman_model.RAMAN_NRM)
      MEAN_NRM_LDA = NRM_LDA.groupby("label").mean()
      MEAN_NRM_LDA["label"] = np.arange(MEAN_NRM_LDA.shape[0])

      for MEAN_DATA, name in zip([MEAN_PCA_LDA,  MEAN_NRM_LDA],
                                 ["PCA", "NRM"]):
         values.append((TRANSCRIPTOME, MEAN_DATA, loopNum, name, shuffle_i, n_components))
      
   print("")
   print(f"All tasks: n={len(values)}", flush=True)

   print(TRANSCRIPTOME.shape, MEAN_DATA.shape)

   results = []
   for value in values:
      result = calcPermutationTest_parallel(value)
      results.append(result)
   results = np.vstack([result for result in results])
   OUT = pd.DataFrame(results, columns=["diff", "p", "mode", "shuffle_ID"])
   OUT.to_csv(outdir, index=None)


def calcPermutationTest_parallel(values):
   print("-", end="", flush=True)
   TRANSCRIPTOME, MEAN_DATA, loopNum, name, shuffle_i, n_components = values
   _, shuffled, _, diff = calcPermutationTest(TRANSCRIPTOME, MEAN_DATA, loopNum=loopNum, default=None,
                                              isLog=False, n_components=n_components)
   p = ((np.array(shuffled) < diff).sum() + 1) / (len(shuffled) + 1)
   print(f"*", end="", flush=True)
   return diff, p, name, shuffle_i


def main():
   parser = argparse.ArgumentParser(description="Comparison")
   parser.add_argument("--outdir", "-o", type=str, required=True)
   parser.add_argument("--seed", "-s", type=int, default=0)
   parser.add_argument("--dataNum", "-n", type=int, default=10)
   parser.add_argument("--shuffleNum", "-s_n", type=int, default=100)
   parser.add_argument("--loopNum", "-l_n", type=int, default=1000)
   parser.add_argument("--pNum", "-p", type=int, default=4)
   parser.add_argument("--n_components", "-n_c", type=int, default=0)
   parser.add_argument("--cutMode", "-c_mode", type=str, default="percent")
   parser.add_argument("--cutRange", "-c_range", type=float, default=98)
   parser.add_argument("--pathRaman", "-p_r", type=str, default="../data/CancerCellLine/PICKED_RAMAN_DATA.csv")
   parser.add_argument("--pathTrans", "-p_t", type=str, default="../data/CancerCellLine/TRANS_TPM_MEAN.csv")

   args = parser.parse_args()

   logs = sys.argv
   print(" ".join(logs))

   argsPrint(args)

   now = datetime.now()
   now_str = now.strftime("%Y/%m/%d %H:%M")
   print(f">>>>>>>>>> START {now_str}")

   pathRaman = pathlib.Path(args.pathRaman).resolve()
   pathTrans = pathlib.Path(args.pathTrans).resolve()

   loadData(args.outdir, seed=args.seed, dataNum=args.dataNum, shuffleNum=args.shuffleNum,
            path_raman=pathRaman, path_trans=pathTrans,
            loopNum=args.loopNum, pNum=args.pNum, n_components=args.n_components,
            cutMode=args.cutMode, cutRange=args.cutRange)

   now = datetime.now()
   now_str = now.strftime("%Y/%m/%d %H:%M")
   print("")
   print(f">>>>>>>>>> END {now_str}")


if __name__ == "__main__":
   main()
