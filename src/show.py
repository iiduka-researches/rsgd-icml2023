import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


if __name__ == '__main__':
    s = 'sfo'
    threshold: float = 1 / 4
    b_list: list[int] = [n for n in range(2 ** 5, 2 ** 9 + 1)]
    lr_list: list[str] = ['constant', 'diminishing1', 'diminishing2']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for lr in lr_list:
        y = []
        for b in b_list:
            df = pd.read_csv(f'results/{lr}/rsgd{b}.csv', header=None)
            loss_list = []
            for loss in df[0].to_list():
                if loss <= threshold:
                    break
                loss_list.append(loss)
            step: int = len(loss_list)

            if s == 'steps':
                y.append(step)
            elif s == 'sfo':
                y.append(step * b)
    
        print(f'{lr}: {b_list[y.index(min(y))]}')
        ax.plot(b_list, y, label=lr)

    ax.set_xticks([2 ** 5, 2 ** 6, 2 **7, 2 ** 8, 2 ** 9])
    b_labels = ['$2^5$', '$2^6$', '$2^7$', '$2^8$', '$2^9$']
    ax.set_xticklabels(b_labels, fontsize=8)
    
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xlabel('Batch Size')
    if s == 'steps':
        plt.ylabel('Steps')
    elif s == 'sfo':
        plt.ylabel('SFO Complexity')
    plt.grid(which='major')
    # plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()
