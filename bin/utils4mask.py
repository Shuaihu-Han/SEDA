import math
import random
import pytorch_lightning as pl
sample_count = 5
chunk_size = 28158
# file_name = 'max_prob.txt'
file_name = 'max_prob_sample5.txt'

test_num = chunk_size * sample_count

context_event_len = 8

def count_one(n):
    count = 0
    while n:
        count += 1
        n &= (n - 1)
    return count

def cal_weight(n, s):
    combination = math.comb(n - 1, s)
    return 1 / (n * combination)

def my_sort_key(arr):
    return arr[1]

def sum_important_array_axis_0(important_array):
    ave_array = [.0] * context_event_len
    for i in range(context_event_len):
        for j in range(chunk_size):
            ave_array[i] += important_array[j][i][1]
        ave_array[i] /= chunk_size
    return ave_array

def laplace_correction(p):
    return (p*chunk_size + 1)/(chunk_size + 5)

def odds(lap_c_p):
    return lap_c_p / (1-lap_c_p)

def WoE(odds1, odds2):
    woe = math.log(odds1, 2) - math.log(odds2, 2)
    return woe if woe >= 0 else 0

def WoE_with_p(p1, p2):
    lp1 = laplace_correction(p1)
    lp2 = laplace_correction(p2)
    odds1 = odds(lp1)
    odds2 = odds(lp2)
    return WoE(odds1, odds2)

def diff_p(p1, p2):
    diff = p1 - p2
    return diff if diff >= 0 else 0

def shuffle_2darray(arr):
    pl.seed_everything(10000)
    for item in arr:
        random.shuffle(item)

def get_ablation_from_importance(imp, chunks):
    ablation_array = [[[i, .0] for i in range(context_event_len)] for _ in range(chunk_size)]
    for i in range(chunk_size):
        sum = 0
        for j in range(context_event_len):

            sum += 2**(7-imp[i][j][0])
            ablation_array[i][j][0] = imp[i][j][0]
            ablation_array[i][j][1] = 1 if chunks[sum][i] > 0.5 else 0
    return ablation_array

def get_ablations_ave(imps, chunks):
    ablations_ave = []
    for item in imps:
        ablation = get_ablation_from_importance(item, chunks)
        ave_ab = sum_important_array_axis_0(ablation)
        ablations_ave.append(ave_ab)
    return ablations_ave

def get_loss_from_importance(imp, chunks):
    loss_array = [[[i, .0] for i in range(context_event_len)] for _ in range(chunk_size)]
    for i in range(chunk_size):
        sum = 0
        for j in range(context_event_len):

            sum += 2**(7-imp[i][j][0])
            loss_array[i][j][0] = imp[i][j][0]
            loss_array[i][j][1] = -math.log(chunks[sum][i] + 1e-5)
    return loss_array

def get_loss_ave(imps, chunks):
    loss_ave = []
    loss = []
    for item in imps:
        loss_item = get_loss_from_importance(item, chunks)
        ave_loss = sum_important_array_axis_0(loss_item)
        loss_ave.append(ave_loss)
        loss.append(loss_item)
    return loss_ave, loss

def get_importances_ave(imps):
    importances_ave = []
    for item in imps:
        ave_imp = sum_important_array_axis_0(item)
        importances_ave.append(ave_imp)
    return importances_ave

def get_importances(imp_array):
    importance1 = []
    importance2 = []
    for i in range(chunk_size):
        importance1.append(sorted(imp_array[i], key=my_sort_key, reverse=True))
        importance2.append(sorted(imp_array[i], key=my_sort_key, reverse=False))
    shuffle_2darray(imp_array)
    importances = []
    importances.extend([importance1, importance2, imp_array])
    return importances
