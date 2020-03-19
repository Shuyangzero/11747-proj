import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from arguments import get_eval_arguments


def get_y_true(name):

    assert name in ["sentihood_NLI_M", "sentihood_NLI_B", "sentihood_single",
                    "sentihood_QA_B", "sentihood_QA_M"], 'this is an unimplemented task'
    fname = "data/sentihood/bert-pair/test_NLI_M.tsv"
    ylabel = []
    df = pd.read_csv(fname, sep='\t')
    for i in range(len(df)):
        assert df['label'][i] in ['Negative',
                                  'Positive', 'None'], 'unknow label'
        if df['label'][i] == 'Negative':
            ylabel.append(2)
        elif df['label'][i] == 'Positive':
            ylabel.append(1)
        else:
            ylabel.append(0)

    return ylabel


def get_y_pred(name, pred_data_dir):

    assert name in ["sentihood_NLI_M", "sentihood_NLI_B", "sentihood_single",
                    "sentihood_QA_B", "sentihood_QA_M"], 'this is an unimplemented task'
    pred = []
    score = []

    if name == "sentihood_single":
        count = 0
        with open(pred_data_dir + "loc1_general.txt", "r", encoding="utf-8") as f1_general, \
                open(pred_data_dir + "loc1_price.txt", "r", encoding="utf-8") as f1_price, \
                open(pred_data_dir + "loc1_safety.txt", "r", encoding="utf-8") as f1_safety, \
                open(pred_data_dir + "loc1_transit.txt", "r", encoding="utf-8") as f1_transit:
            s = f1_general.readline().strip().split()
            while s:
                count += 1
                p = int(s[0])
                s1, s2, s3 = float(s[1]), float(s[2]), float(s[3])
                score.append([s1, s2, s3])
                pred.append(p)
                ftype = None
                if count % 4 == 0:
                    ftype = f1_general
                if count % 4 == 1:
                    ftype = f1_price
                if count % 4 == 2:
                    ftype = f1_safety
                if count % 4 == 3:
                    ftype = f1_transit
                s = ftype.readline().strip().split()
        count = 0
        with open(pred_data_dir + "loc2_general.txt", "r", encoding="utf-8") as f2_general, \
                open(pred_data_dir + "loc2_price.txt", "r", encoding="utf-8") as f2_price, \
                open(pred_data_dir + "loc2_safety.txt", "r", encoding="utf-8") as f2_safety, \
                open(pred_data_dir + "loc2_transit.txt", "r", encoding="utf-8") as f2_transit:
            s = f2_general.readline().strip().split()
            while s:
                count += 1
                p = int(s[0])
                s1, s2, s3 = float(s[1]), float(s[2]), float(s[3])
                score.append([s1, s2, s3])
                pred.append(p)
                ftype = None
                if count % 4 == 0:
                    ftype = f2_general
                if count % 4 == 1:
                    ftype = f2_price
                if count % 4 == 2:
                    ftype = f2_safety
                if count % 4 == 3:
                    ftype = f2_transit
                s = ftype.readline().strip().split()
    elif name in ["sentihood_NLI_M", "sentihood_QA_M"]:
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                p = int(s[0])
                s1, s2, s3 = float(s[1]), float(s[2]), float(s[3])
                score.append([s1, s2, s3])
                pred.append(p)
                s = f.readline().strip().split()
    elif name in ["sentihood_NLI_B", "sentihood_QA_B"]:
        #NEED FURTHER UPDATES#####################################
        count = 0
        tmp = []
        num = 3
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                count += 1
                tmp.append([float(s[2])])
                if count % num == 0:
                    tmp_sum = np.sum(tmp)
                    t = []
                    for i in range(num):
                        avg = tmp[i] / tmp_sum
                        t.append(avg)
                        score.append(t)
                    if t[0] >= t[1] and t[0] >= t[2]:
                        pred.append(0)
                    elif t[1] >= t[0] and t[1] >= t[2]:
                        pred.append(1)
                    else:
                        pred.append(2)
                        tmp = []
                s = f.readline().strip().split()
    return pred, score


def sentihood_macro_F1(y_true, y_pred):
    p_all, r_all, count = 0, 0, 0
    num = 4
    for i in range(len(y_pred) // num):
        a = set()
        b = set()
        for j in range(num):
            ind = i * num + j
            if y_true[ind] != 0:
                b.add(j)
            if y_pred[ind] != 0:
                a.add(j)

        if len(b) == 0:
            continue
        ab_intersect = b.intersection(a)
        if len(ab_intersect) > 0:
            r, p = len(ab_intersect) / len(b), len(ab_intersect) / len(a)
        else:
            r, p = 0, 0
        count += 1
        p_all += p
        r_all += r
    Ma_r, Ma_p = r_all / count,  p_all / count
    return Ma_p * Ma_r * 2 / (Ma_p + Ma_r)


def sentihood_strict_acc(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    num = 4
    true_cases = 0
    for i in range(int(len(y_true) / num)):
        ind = num * i
        if y_true[ind] != y_pred[ind] or y_true[ind + 1] != y_pred[ind + 1] or y_true[ind + 2] != y_pred[ind + 2] or y_true[ind + 3] != y_pred[ind + 3]:
            continue
        true_cases += 1
    return true_cases / int(len(y_true) / num)


def sentihood_AUC_Acc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # aspect-Macro-AUC
    aspect_score, aspect_true = [], []
    aspect_num = 4
    aspect_trues = [[] for i in range(aspect_num)]
    aspect_scores = [[] for i in range(aspect_num)]
    for i in range(len(y_true)):
        val = 1 if y_true[i] <= 0 else 0
        aspect_true.append(val)
        aspect_score.append(score[i][0])
        ind = i % aspect_num
        aspect_trues[ind].append(aspect_true[-1])
        aspect_scores[ind].append(aspect_score[-1])

    aspect_auc = []
    for i in range(aspect_num):
        roc = metrics.roc_auc_score(aspect_trues[i], aspect_scores[i])
        aspect_auc.append(roc)

        aspect_Macro_AUC = np.mean(aspect_auc)

    # sentimentsen-Macro-AUC
    sent_score, sent_pred, sent_true = [], [], []
    sent_num = 4
    sent_scores = [[] for i in range(sent_num)]
    sent_trues = [[] for i in range(sent_num)]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            sent_true.append(y_true[i] - 1)  # "Postive":0, "Negative":1
            tmp_score = score[i][2] / (score[i][1] + score[i][2])
            sent_score.append(tmp_score)
            val = 0 if tmp_score <= 0.5 else 1
            sent_pred.append(val)
            ind = i % aspect_num
            sent_trues[ind].append(sent_true[-1])
            sent_scores[ind].append(sent_score[-1])

    sentiment_auc = []
    for i in range(aspect_num):
        roc = metrics.roc_auc_score(sent_trues[i], sent_scores[i])
        sentiment_auc.append(roc)
    sentiment_Macro_AUC = np.mean(sentiment_auc)
    sentiment_Acc = metrics.accuracy_score(
        np.array(sent_true), np.array(sent_pred))

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    sall, gall, sg = 0, 0, 0
    num = 4
    names = 5
    for i in range(len(y_pred) // names):
        s, g = set(), set()
        for j in range(names):
            ind = i * names + j
            if y_pred[ind] != num:
                s.add(j)
            if y_true[ind] != num:
                g.add(j)
        if len(g) == 0:
            continue
        s_g_int = s.intersection(g)
        gall += len(g)
        sall += len(s)
        sg += len(s_g_int)

    p, r = sg / sall, sg / gall

    return p, r, 2 * p * r / (p + r)


def cp(s, i, j, k):
    return s[i][j] >= s[i, k]


def semeval_4(y_true, y_pred, s, classes=4):

    sum = 0
    sum_right = 0
    for i in range(len(y_true)):
        if y_true[i] == 4:
            continue
        tmp = y_pred[i]
        sum += 1
        if tmp == 4:
            if cp(s, i, 0, 1) and cp(s, i, 0, 2) and cp(s, i, 0, 3):
                tmp = 0
            elif cp(s, i, 1, 0) and cp(s, i, 1, 2) and cp(s, i, 1, 3):
                tmp = 1
            elif cp(s, i, 2, 0) and cp(s, i, 2, 1) and cp(s, i, 2, 3):
                tmp = 2
            else:
                tmp = 3
        if y_true[i] == tmp:
            sum_right += 1

    return sum_right / sum


def semeval_3(y_true, y_pred, s, classes=4):
    sum, sum_right = 0, 0
    for i in range(len(y_true)):
        if y_true[i] >= 3:
            continue
        sum += 1
        tmp = y_pred[i]
        if tmp >= 3:
            if cp(s, i, 0, 1) and cp(s, i, 0, 2):
                tmp = 0
            elif cp(s, i, 1, 0) and cp(s, i, 1, 2):
                tmp = 1
            else:
                tmp = 2
        if y_true[i] == tmp:
            sum_right += 1
    return sum_right / sum


def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    sentiment_Acc = 0
    if classes == 4:
        sentiment_Acc = semeval_4(y_true, y_pred, score, classes)
    elif classes == 3:
        sentiment_Acc = semeval_3(y_true, y_pred, score, classes)
    else:
        sum = 0
        sum_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3 or y_true[i] == 1:
                continue
            sum += 1
            tmp = y_pred[i]
            if tmp >= 3 or tmp == 1:
                if score[i][0] < score[i][2]:
                    tmp = 2
                else:
                    tmp = 0
            if y_true[i] == tmp:
                sum_right += 1
        sentiment_Acc = sum_right / sum

    return sum_right / sum


def main():

    args = get_eval_arguments()

    result = collections.OrderedDict()
    if args.task_name in ["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", "sentihood_NLI_B", "sentihood_QA_B"]:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(
            y_true, score)
    measures = ['aspect_strict_Acc', 'aspect_Macro_F1', 'aspect_Macro_AUC', 'sentiment_Acc',
                'sentiment_Macro_AUC']
    vals = [aspect_strict_Acc, aspect_Macro_F1, aspect_Macro_AUC,
            sentiment_Acc, sentiment_Macro_AUC]
    for i in range(len(measures)):
        print("{}:{}\t".format(measures[i], vals[i]))


if __name__ == "__main__":
    main()
