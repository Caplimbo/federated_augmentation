import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
markers = ['.', ',', 'o', '*', '_', '1', 'v', '^', '<', '>', '1', '2']


def get_data(path, centralized=False):
    train_accs = []
    test_accs = []
    with open(path, "r") as f:
        for line in f.readlines():
            if centralized:
                test_acc = line.strip()[-6:]
            else:
                train_acc, test_acc = line.strip().split("\t")
            # train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
    return train_accs, test_accs


def plot_figure(root, name, limit=60, scale=500, smooth_flag=False, range_y=[0.4, 0.65], zero_acc=0.0):


    all_logs = []
    for root, dirs, files in os.walk(root):
        for file in files:
            label = root.split("/")[-1]
            if file[-5] != 'g':
                continue
            path = os.path.join(root, file)

            print(path)
            centralized = (root.split("/")[-1] == 'Centralized')
            _, test_acc = get_data(path, centralized)

            mark = root.split("/")[-1]
            print(label, test_acc)
            print(len(test_acc))
            test_acc = test_acc[:limit]
            test_acc = [zero_acc] + test_acc
            if smooth_flag:
                test_acc = smooth(test_acc)

            label = ", ".join(mark.split("_"))
            all_logs.append((test_acc, label))
    all_logs.sort(key=lambda x: x[0][-1], reverse=True)
    # plot_small_figure(name, all_logs, 5, scale)
    #exit()
    x = np.arange(0, (limit+1) * scale, scale)
    plt.figure(figsize=(12, 9))
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy")
    index = 0
    plt.ylim(range_y)
    plt.xticks(np.arange(0, limit+10, limit//10) * scale, np.arange(0, limit + 10, limit//10) * scale)

    for acc, label in all_logs:
        # test_acc = smooth(test_acc, 0.8)
        print(label)
        plt.plot(x, acc, label=f"{label}", marker=markers[index % len(markers)])
        index += 1
    plt.legend(prop={'size': 10})
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close()

def plot_small_figure(name, all_accs, final_count, scale):
    plt.figure(figsize=(3, 2))
    x = np.arange((len(all_accs[0][0]) - final_count) * scale, (len(all_accs[0][0])) * scale, scale)
    print("x: ", x)
    x_label = np.arange((len(all_accs[0][0]) - final_count), len(all_accs[0][0]))
    print("label: ", x_label)
    x_label *= scale
    print("label: ", x_label)
    print()
    plt.xticks(size=8)
    plt.yticks(size=7)
    index = 0
    for acc, label in all_accs:
        print(acc[-final_count:],label)
        plt.plot(x, acc[-final_count:], label=f"{label}", marker=markers[index % len(markers)])
        index += 1
    plt.savefig(f"{name}_small.png")
    plt.show()

    plt.close()


def plot_data_hist(path):
    with open(path, "r") as f:
        data = list(map(int, f.read().strip().split(" ")))
    res = pd.Series(data)
    print(res.describe())
    #plt.hist(data, bins=50)
    #plt.show()

def plot_acc_hist(full_client_acc):
    """

    :param test_acc: list of (test_acc, number_of_samples)
    :return: plots a histogram of all accs
    """
    accs = [item[0] for item in full_client_acc]
    plt.hist(accs, bins=8)
    plt.show()
    plt.close()

def compare_client_accs(client_accs_new, client_accs_original, name, lower_bound=-0.75, upper_bound=0.75, step=0.1):
    delta_accs = []
    for key in client_accs_new.keys():
        delta_accs.append(client_accs_new[key] - client_accs_original[key])

    plt.hist(delta_accs, bins=np.arange(lower_bound, upper_bound, step), align='mid', edgecolor="white", rwidth=0.8)
    y_pos = np.arange(0, 1.0, 0.1) * len(delta_accs)
    y_str = np.arange(0, 1.0, 0.1)
    y_str = map(lambda x: f"{x:.1f}", y_str)
    plt.yticks(y_pos, y_str)
    plt.ylabel("Proportion of Clients")
    plt.xlabel("Test Accuracy Gain")
    plt.savefig(f"{name}_hist.png")
    plt.show()
    plt.close()

def scatter_plot_acc_compare(client_accs_new, client_accs_original, name):
    x, y = [], []
    for key, value in client_accs_original.items():
        x.append(value)
        y.append(client_accs_new[key] - value)
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x=x, y=y, c=z, s=20, cmap='Spectral')

    cbl = plt.colorbar(label="Number of Clients")
    tick_locater = ticker.MaxNLocator(nbins=9)
    cbl.locater = tick_locater
    cbl.set_ticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    cbl.update_ticks()
    plt.ylabel("Test Accuracy Gain by Data Augmentation")
    plt.xlabel("Test Accuracy of FedAvg + Fine-tuning")
    plt.savefig(f"{name}_scatter.png")
    plt.show()
    plt.close()


def read_client_acc_data(path):
    all_rounds_result = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            one_round = []
            for one_client in line.strip().split('\t'):
                acc, weight, id = one_client.split()
                acc = float(acc)
                weight = int(weight)
                one_round.append((acc, weight, id))
            all_rounds_result.append(one_round)
            line = f.readline()
    print(len(all_rounds_result))
    return all_rounds_result


def plot_difference(path_new, path_original, name, lower_bound=-0.6, upper_bound=0.75, step=0.1, index=-1):
    client_data_new = read_client_acc_data(path_new)[index]
    client_data_old = read_client_acc_data(path_original)[index]
    client_new_dct, client_old_dct = {}, {}
    for i in range(len(client_data_old)):
        client_new_dct[client_data_new[i][2]] = client_data_new[i][0]
        client_old_dct[client_data_old[i][2]] = client_data_old[i][0]
    # compare_client_accs(client_new_dct, client_old_dct, name, lower_bound, upper_bound, step)
    scatter_plot_acc_compare(client_new_dct, client_old_dct, name)

def read_loss_data(path):
    train_loss = []
    with open(path, 'r') as f:
        for line in f.readlines():
            train_loss.append(float(line.strip()))
    return train_loss

def plot_loss_figure(root, name, smooth_factor=0.6):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Rounds")
    plt.ylabel("Train Loss")
    # plt.title("Train Loss over Communication Rounds", fontdict={'fontsize': 23})
    index = 0
    # plt.ylim([0.85, 0.96])

    for root, dirs, files in os.walk(root):
        for file in files:

            if file[-5] != 'd':
                continue
            path = os.path.join(root, file)
            print(path)
            train_loss = read_loss_data(path)
            print(train_loss)
            train_loss = smooth(train_loss, smooth_factor)
            mark = root.split("/")[-1]
            if root.split('/')[-1][0] == 'f' and root.split('/')[-1][7] == 'w':
                continue
            elif len(train_loss) == 200:
                continue
            x = np.arange(0, len(train_loss)*50, 50)
            label = ", ".join(mark.split("_"))
            plt.plot(x, train_loss, label=f"{label}")  # , marker=markers[index % len(markers)])
            index += 1
    plt.legend(prop={'size': 5})
    plt.savefig(f"{name}_loss.png")
    plt.show()
    plt.close()


def smooth(data_list, weight=0.85): #weight是平滑度，tensorboard 默认0.6    last = scalar[0]
    smoothed = []
    last = data_list[0]
    for point in data_list:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


if __name__ == "__main__":
    # plot_figure("text_result/amazon_books/client10", "amazon_books_0", smooth_flag=True, range_y=[0.1, 0.65], zero_acc=0.1557)
    # plot_figure("result/only_digits/new", "femnist_digits_0", limit=60, scale=50, range_y=[0.1, 1.03], zero_acc=0.1081)
    plot_figure("result/with_letters/new", "femnist_full_0", limit=20, scale=50, range_y=[0, 0.88], zero_acc=0.0183)

    # plot_loss_figure("text_result/amazon_books/client10", "books_5class_0.01", 0.95)

    # plot_difference("result/with_letters/new/Augment_WithTune/all_clients.txt", "result/with_letters/new/FedAvg_WithTune/all_clients.txt", "femnist_full", -0.3, 0.5, 0.05, index=20)
    # plot_difference("result/only_digits/new/Augment_WithTune/all_clients.txt", "result/only_digits/new/FedAvg_WithTune/all_clients.txt", "femnist_digits", -0.15, 0.15, 0.02, index=40)
    # plot_difference("text_result/amazon_books/client10/Augment_WithTune/all_clients.txt", "text_result/amazon_books/client10/FedAvg_WithTune/all_clients.txt", "amazon_books", index=60)


"""
avg notune: 0.8504
avg withtune: 0.8537
augment notune: 0.8567
augment withtune: 0.8644
"""
