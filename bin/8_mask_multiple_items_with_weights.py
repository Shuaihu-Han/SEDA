import utils4mask as ut
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import random

weights = [.0] * ut.context_event_len
for i in range(ut.context_event_len):
    weights[i] = ut.cal_weight(ut.context_event_len, i)

def __main__():
    with open(ut.file_name, 'r') as f:
        lines = tqdm(f.readlines())
        lines.set_description('loading numbers from file')
        numbers = []
        for line in lines:
            line_numbers = line.split()
            for number in line_numbers:
                numbers.append(float(number))

        chunks = [numbers[i:i+ut.test_num] for i in range(0, len(numbers), ut.test_num)]
        chunks = np.array(chunks)
        chunks = np.split(chunks, ut.sample_count, axis=1)
        chunks = np.array(chunks)
        chunks = chunks.mean(axis=0)
        # print(chunks.shape)
        important_array = [[[i, .0] for i in range(ut.context_event_len)] for _ in range(ut.chunk_size)]

        for i in trange(ut.chunk_size, desc='computing importance array:'):
            for j in range(ut.context_event_len):
                bit_mask = 1 << (7 - j)
                for k in range(2**8):
                    bit_value = k & bit_mask
                    S_width = ut.count_one(k) - 1
                    if bit_value != 0:
                        important_array[i][j][1] += (ut.WoE_with_p(chunks[k][i], chunks[k - 2**(7-j)][i])) * weights[S_width]

        print(important_array[0])

        importances = ut.get_importances(important_array)
        for i in range(20):
            print(importances[0][ut.chunk_size - 1 - i])

        ablations_ave = ut.get_ablations_ave(importances, chunks)
        imps_ave = ut.get_importances_ave(importances)
        loss_ave, loss = ut.get_loss_ave(importances, chunks)

        colors = ['orangered', 'palegreen', 'dodgerblue']
        labels = ['important', 'not important', 'random']

        plt.figure(1)
        for i in range(len(ablations_ave)):
            plt.plot(range(1, ut.context_event_len + 1), ablations_ave[i], c=colors[i], label=labels[i])
        plt.ylabel('Accuracy')
        ax = plt.gca()
        ax.spines['top'].set_color('darkgrey')
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['left'].set_color('darkgrey')
        ax.spines['right'].set_color('darkgrey')
        ax.tick_params(axis='x', color='darkgrey', labelcolor='dimgrey')
        ax.tick_params(axis='y', color='darkgrey', labelcolor='dimgrey')
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        plt.plot()
        plt.savefig('acc.pdf')

        plt.figure(2)
        for i in range(len(imps_ave)):
            plt.plot(range(1, ut.context_event_len + 1), imps_ave[i], c=colors[i], label=labels[i])
        plt.ylabel('$\phi$')
        ax = plt.gca()
        ax.spines['top'].set_color('darkgrey')
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['left'].set_color('darkgrey')
        ax.spines['right'].set_color('darkgrey')
        ax.tick_params(axis='x', color='darkgrey', labelcolor='dimgrey')
        ax.tick_params(axis='y', color='darkgrey', labelcolor='dimgrey')
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        plt.plot()
        plt.savefig('diff.pdf')

        plt.figure(3)
        N = 2000
        x = []
        y = []
        for k in range(ut.context_event_len):
            for i in range(N):
                x.append(k + random.random())
                y.append(importances[0][i][k][1])
        plt.ylabel('$\phi$')
        ax = plt.gca()
        ax.spines['top'].set_color('darkgrey')
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['left'].set_color('darkgrey')
        ax.spines['right'].set_color('darkgrey')
        ax.tick_params(axis='x', color='darkgrey', labelcolor='dimgrey')
        ax.tick_params(axis='y', color='darkgrey', labelcolor='dimgrey')
        plt.scatter(x, y, s=5)
        plt.savefig('scatter.pdf')

        plt.figure(4)
        for i in range(len(loss_ave)):
            plt.plot(range(1, ut.context_event_len + 1), loss_ave[i], c=colors[i], label=labels[i])
        plt.ylabel('Loss')
        ax = plt.gca()
        ax.spines['top'].set_color('darkgrey')
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['left'].set_color('darkgrey')
        ax.spines['right'].set_color('darkgrey')
        ax.tick_params(axis='x', color='darkgrey', labelcolor='dimgrey')
        ax.tick_params(axis='y', color='darkgrey', labelcolor='dimgrey')
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        plt.plot()
        plt.savefig('Loss.pdf')

        plt.show()
if __name__ == '__main__':
    __main__()