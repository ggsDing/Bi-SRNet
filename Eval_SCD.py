# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import math
import os
import glob
from tqdm import tqdm
from scipy import stats
from utils.utils import AverageMeter

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def get_hist(image, label):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def Eval():
    IMAGE_FORMAT = '.png'
    
    INFER_DIR1 = '/PATH_TO_PRED/im1/'  # Inference path1
    INFER_DIR2 = '/PATH_TO_PRED/im2/'  # Inference path2
    LABEL_DIR1 = '/PATH_TO_GT/label1/'  # GroundTruth path
    LABEL_DIR2 = '/PATH_TO_GT/label2/'  # GroundTruth path

    infer_list1 = glob.glob(INFER_DIR1 + "*{}".format(IMAGE_FORMAT))
    infer_list2 = glob.glob(INFER_DIR2 + "*{}".format(IMAGE_FORMAT))

    infer_list1.sort()
    infer_list2.sort()

    infer_list = infer_list1 + infer_list2

    label_list1 = glob.glob(LABEL_DIR1 + "*{}".format(IMAGE_FORMAT))
    label_list2 = glob.glob(LABEL_DIR2 + "*{}".format(IMAGE_FORMAT))

    label_list1.sort()
    label_list2.sort()

    label_list = label_list1 + label_list2
    assert len(label_list) == len(infer_list), "Predictions do not match targets length"
    assert set([os.path.basename(label) for label in label_list1]) == set([os.path.basename(infer) for infer in infer_list1]), "Predictions do not match targets name"
    assert set([os.path.basename(label) for label in label_list2]) == set([os.path.basename(infer) for infer in infer_list2]), "Predictions do not match targets name"
    acc_meter = AverageMeter()

    hist = np.zeros((num_class, num_class))
    for infer, gt in tqdm(zip(infer_list, label_list)):
        try:
            infer = Image.open(infer)
        except:
            print("File open error")
            sys.exit(0)
        try:
            label = Image.open(gt)
        except:
            print("File open error")
            sys.exit(0)
        infer_array = np.array(infer)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        acc = accuracy(infer_array, label_array)
        acc_meter.update(acc)

        hist += get_hist(infer_array, label_array)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0] #TN
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0] #FP. hist.sum(1): pred_hist
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0] #FN. hist.sum(0): label_hist
    c2hist[1][1] = hist_fg.sum() #TP
    print('bn_hist: TP %d, FN %d, FP %d, TN %d'%(c2hist[1][1], c2hist[1][0], c2hist[0][1], c2hist[0][0]))
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    print('Mean IoU = %.3f' % (IoU_mean*100))
    print('Sek = %.3f' % (Sek*100))
    print('Score = %.3f' % (Score*100))
    
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    F1 = stats.hmean([SC_Precision, SC_Recall])
    print(acc_meter.avg*100)
    print('change_ratio = %.4f, SC_Precision = %.4f, SC_Recall = %.4f, F_scd = %.4f' % (change_ratio*100, SC_Precision*100, SC_Recall*100, F1*100))


if __name__ == '__main__':
    num_class = 7
    Eval()
