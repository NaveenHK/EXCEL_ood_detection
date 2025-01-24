from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
import openood.utils.comm as comm
from .base_postprocessor import BasePostprocessor
from sklearn.decomposition import PCA
import numpy as np
from torchsummary import summary
import sys

class ExcelPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.a = self.args.a
        self.upper = self.args.upper
        self.reward = self.args.reward
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.postprocessorname = "excel"
        self.topk = 200  # No.of classes

    @torch.no_grad()
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  pca_obj: None,
                  pca_fit: int,
                  progress: bool = True):
        pred_list, conf_list, label_list, logits_list = [], [], [], []
        for batch in tqdm(data_loader,disable=False):
                          #disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf, logits = self.postprocess(net, data, pca_obj)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())
            logits_list.append(logits.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        logits_list = torch.cat(logits_list).numpy()



        if pca_fit == 1:
            ####### pca_fit == 1 computes the original probabilities of the class likelihood matrics.
            ####### pca_fit == 2 smoothes the class likelihood matrices. This is used for hyperparamter tuning.

            ###### Logits list is a numpy tensor with shape (No. of samples, No.of classes) #######
            # logits_list = np.load("imagenet_logits.npy")
            # print("Numpy file loading completed...", flush=True)
            number_classes = logits_list.shape[1]
            train_y = label_list
            pred_class = np.argmax(logits_list, axis=1)

            pred_pdf = {}
            for i in range(number_classes):
                top_sequence = {}
                ref_seq = []

                correct_ids = np.where(pred_class == i)[0] #ids[np.where(pred_class == i)[0]]  ### Correctly predicted sample IDs of class i.
                # print(correct_ids)

                correct_logits = logits_list[correct_ids]
                logits_id = np.argsort(-correct_logits, axis=1)[:, :self.topk]

                # print(logits)
                # print(logits_id)
                pos_count = 0
                for j in range(self.topk):
                    posterior = {}
                    ij = logits_id[:, j]
                    # print(ij)
                    unique_values, counts = np.unique(ij, return_counts=True)
                    counts = counts / np.sum(counts)

                    if True:
                        for z in range(number_classes):
                            if z in unique_values:
                                index = np.where(unique_values == z)[0][0]
                                posterior[unique_values[index]] = counts[index]
                            else:
                                posterior[z] = 0 


                    top_sequence[j] = posterior


                pred_pdf[i] = top_sequence


            return pred_list, conf_list, label_list, pred_pdf

        elif pca_fit==2:
            number_classes = logits_list.shape[1]
            pred_pdf_2 = pca_obj
            for i in pred_pdf_2:
                for j in pred_pdf_2[i]:
                    for k in pred_pdf_2[i][j]:
                        if pred_pdf_2[i][j][k] > self.upper / number_classes:
                            pred_pdf_2[i][j][k]  = self.reward/number_classes

                        elif pred_pdf_2[i][j][k] > 1 / number_classes:
                            pred_pdf_2[i][j][k] = 1/number_classes

                        elif pred_pdf_2[i][j][k] > 0 :
                            pred_pdf_2[i][j][k] =  -1/number_classes
                        else:
                            pred_pdf_2[i][j][k] = -self.reward/number_classes

            return pred_list, conf_list, label_list, pred_pdf_2



        else:
            return pred_list, conf_list, label_list, None



    def postprocess(self, net: nn.Module, data: Any, pca_obj):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)


####### Custom OOD Metric "conf" variable is computed within this block #########
        if pca_obj is not None:
            predic_pdf = pca_obj
            original_logits = output.cpu().numpy()

            pred_score_id = self.prediction_strength(original_logits, predic_pdf , self.topk)

            conf = torch.tensor(pred_score_id).cuda()

        return pred, conf, output

    def prediction_strength(self,orig_logits, pred_pdf, k):
        softmax_score = np.exp(orig_logits) / np.sum(np.exp(orig_logits), axis=1, keepdims=True)
        y_pred = np.argsort(-orig_logits, axis=1)[:, :k]
        max_logits = np.max(orig_logits, axis=1)
        pred_strength = np.zeros((y_pred.shape[0]))
        for i in range(y_pred.shape[0]):
            max_lgt = max_logits[i]
            # print(max_lgt)
            seq = y_pred[i, :]
            class_id = seq[0]
            score = 0
            for position in range(1, k):
                score += pred_pdf[class_id][position][seq[position]]
            pred_strength[i] = (1-self.a)*max_lgt + self.a*score
        return pred_strength



    def set_hyperparam(self,  hyperparam:list):
        self.a = hyperparam[0]
        self.upper = hyperparam[1]
        self.reward = hyperparam[2]

    def get_hyperparam(self):
        return [self.a, self.upper, self.reward]
