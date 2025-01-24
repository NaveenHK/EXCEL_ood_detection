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


class SPCAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        # self.pcacomponents = self.args.pcacomponents
        self.a = self.args.a
        self.upper = self.args.upper
        self.reward = self.args.reward
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.postprocessorname = "SPCA"
        self.topk = 10

    @torch.no_grad()
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  pca_obj: None,
                  pca_fit: bool = False,
                  progress: bool = True):
        pred_list, conf_list, label_list, logits_list = [], [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
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

        if pca_fit:
            ######## Reference Prediction Sequence for each ID class is found here ########
            ###### Logits list is a numpy tensor with shape (No. of samples, No.of classes) #######

            number_classes = logits_list.shape[1]
            train_y = label_list
            pred_class = np.argmax(logits_list, axis=1)

            pred_pdf = {}
            pdf_entropy = {}
            for i in range(number_classes):
                pos_entropy = {}
                top_sequence = {}
                ref_seq = []
                # ids = np.where(train_y == i)[0]
                # print(ids)
                # pred_i = pred_class[ids]
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

                    entropy = self.compute_entropy(counts, number_classes)
                    # print("Class "+str(i)+" Position "+str(j)+" Posterior :",unique_values, counts)
                    if True: #entropy < 0.99:
                        # pos_count +=1
                        for z in range(number_classes):
                            if z in unique_values:
                                index = np.where(unique_values == z)[0][0]
                                # if counts[index] > 30/number_classes:   # 5 best
                                #     posterior[unique_values[index]] = counts[index] #20/number_classes
                                if counts[index] > self.upper/number_classes:   # 5 best
                                    posterior[unique_values[index]] = self.reward/number_classes # 10 best
                                elif counts[index] > 1/number_classes:
                                    posterior[unique_values[index]] = 1/number_classes #counts[index]
                                else:
                                    posterior[unique_values[index]] =  -1/number_classes
                            else:
                                posterior[z] = -self.reward/number_classes
                    # else:
                    #     for z in range(number_classes):
                    #         if z in unique_values:
                    #             index = np.where(unique_values == z)[0][0]
                    #             if counts[index] > 1 / (number_classes):
                    #                 posterior[unique_values[index]] = 1/number_classes
                    #             else:
                    #                 posterior[unique_values[index]] = -1/number_classes
                    #
                    #         else:
                    #             posterior[z] =  -10/number_classes


                    top_sequence[j] = posterior
                    pos_entropy[j] = entropy
                # print("Class - ",i, pos_count)

                pred_pdf[i] = top_sequence
                pdf_entropy[i] = pos_entropy

                file_path = 'places_pdf_new.pickle'

                # Open the file in binary write mode
                with open(file_path, 'wb') as file:
                    # Use pickle.dump() to save the dictionary to the file
                    pickle.dump(pred_pdf, file)

            return pred_list, conf_list, label_list, pred_pdf

        else:
            return pred_list, conf_list, label_list, None

    # def hook(self, net: nn.Module, input, output):
    #     outputs.append(output)

    def postprocess(self, net: nn.Module, data: Any, pca_obj):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)


####### Custom OOD Metric "conf" variable is computed within this block #########
        if pca_obj is not None:
            predic_pdf = pca_obj
            original_logits = output.cpu().numpy()
            # topk_test = np.argsort(-original_logits, axis=1)[:, :self.topk]

            pred_score_id = self.prediction_strength(original_logits, predic_pdf , self.topk)

            conf = torch.tensor(pred_score_id).cuda()

        return pred, conf, output

    def prediction_strength(self,orig_logits, pred_pdf, k):
        # a = 0.2
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
                score += pred_pdf[class_id][position][seq[position]] #* (1 / pdf_entropy[class_id][position])
            pred_strength[i] = (1-self.a)*max_lgt + self.a*score
        return pred_strength

    def compute_entropy(self,probabilities,base):
        small_constant = 1e-10  # A small constant to avoid zero probabilities
        modified_probs = probabilities + small_constant
        entropy = -np.round(np.sum(modified_probs * np.log(modified_probs) / np.log(base)), 5)
        # entropy = -np.round(np.sum(modified_probs * np.log10(modified_probs)), 5)
        return entropy

    def set_hyperparam(self,  hyperparam:list):
        self.a = hyperparam[0]
        self.upper = hyperparam[1]
        self.reward = hyperparam[2]

    def get_hyperparam(self):
        return [self.a, self.upper, self.reward]
