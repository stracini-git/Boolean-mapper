import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def lz_complexity(s):
    i, k, l = 0, 1, 1
    k_max = 1
    n = len(s) - 1
    c = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k = k + 1
            if l + k >= n - 1:
                c = c + 1
                break
        else:
            if k > k_max:
                k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c


def k_lempel_ziv(sequence):
    if (np.sum(sequence == 0) == len(sequence)) or (np.sum(sequence == 1) == len(sequence)):

        out = np.log2(len(sequence))
    else:
        forward = sequence
        backward = sequence[::-1]

        out = np.log2(len(sequence)) * (lz_complexity(forward) + lz_complexity(backward)) / 2

    # if out == 14.0:
    #     return 7.0
    return out


def array_with_n_ones(size, n):
    y = [1] * n
    y.extend([0] * (size - n))
    np.random.shuffle(y)

    return np.asarray(y)


def hypothesis_1(y_train, size, **kwargs):
    """
    hypothesis_1:
        - the model learns the 0/1 statistics from the train set
        - for the test set it only predicts correctly the frequency of 0/1, not the correlation between
            inputs and labels
    """

    # do the train statistics
    n1 = np.count_nonzero(y_train)
    n0 = len(y_train) - n1

    # set the probability of choosing the test statistics according to the train statistics
    p1 = n1 / len(y_train)
    p0 = n0 / len(y_train)

    # if the train data had all zeros or all ones it is unlikely that the prediction will be all 0s or 1s,
    # so we change the probability that at least one or two prediction will be different - this line of
    # reasoning matches the low LZ datapoints pretty well
    if p1 == 0:
        p1 = np.random.choice([1, 2]) / len(y_train)
        p0 = 1 - p1

    if p0 == 0:
        p0 = np.random.choice([1, 2]) / len(y_train)
        p1 = 1 - p0

    # do a "prediction" according to hypothesis_1: random prediction with probabilities for 0/1 classes
    prediction = np.random.choice([0, 1], p=(p0, p1), size=size)

    return prediction, p0, p1


def hypothesis_2(y_train, size, **kwargs):
    """
    hypothesis_2:
        - the model learns the 0/1 statistics from the train set
        - for the class with more examples it increases the probability of the class
    """

    # do the train statistics
    n1 = np.count_nonzero(y_train)
    n0 = len(y_train) - n1

    p1 = n1 / len(y_train)
    p0 = n0 / len(y_train)

    sigmoid = lambda x: 2 / (1 + np.exp(-x))
    if 0 < min(p1, p0) < np.random.choice(np.arange(10, 32)) / 64:  # parameters found by hand
        dp = np.abs(p0 - p1)
        f = 2.5 * sigmoid(2.5 * dp)  # parameters found by hand
        if p1 > p0:
            p0 /= f
            p1 = 1 - p0
        elif p0 > p1:
            p1 /= f
            p0 = 1 - p1

    # if the train data had all zeros or all ones it is unlikely that the prediction will be all 0s or 1s,
    # so we change the probability that at least one or two prediction will be different - this line of
    # reasoning matches the low LZ datapoints pretty well
    if p1 == 0:
        p1 = np.random.choice([1, 2]) / len(y_train)
        p0 = 1 - p1

    if p0 == 0:
        p0 = np.random.choice([1, 2]) / len(y_train)
        p1 = 1 - p0

    # do a "prediction" according to hypothesis_1: random prediction with probabilities for 0/1 classes
    prediction = np.random.choice([0, 1], p=(p0, p1), size=size)

    return prediction, p0, p1


def hypothesis_3(y_train, size, **kwargs):
    """
    hypothesis_3:
        always bet on the class with the highest probability
    """

    # do the train statistics
    n1 = np.count_nonzero(y_train)
    n0 = len(y_train) - n1

    p1 = n1 / len(y_train)
    p0 = n0 / len(y_train)

    if p1 > p0:
        p1 = 1
        p0 = 0

    elif p0 > p1:
        p0 = 1
        p1 = 0

    # do a "prediction" according to hypothesis_1: random prediction with probabilities for 0/1 classes
    prediction = np.random.choice([0, 1], p=(p0, p1), size=size)

    np.random.shuffle(prediction)
    return prediction, p0, p1


def prepare_random_boolean_dataset(n1s, train_test_ratio):
    target_function = array_with_n_ones(128, n=n1s)

    # calculate LZ complexity
    lz_complexity = k_lempel_ziv(target_function)

    # find the index where to split in train/test
    split_index = int(len(target_function) * train_test_ratio)

    # split full data into train and test samples
    y_train = target_function[:split_index]
    y_test = target_function[split_index:]

    return y_train, y_test, lz_complexity


def calculate_error(true, pred):
    # count the number of matches between true labels and the prediction
    matches = np.count_nonzero(np.equal(pred, true))

    # calculate accuracy
    acc = matches / len(true)

    # calculate error
    error = 1 - acc

    return error


def statistical_learners():
    # parameters for the figure
    scale = 8
    n_rows = 2
    n_cols = 2
    aspect_ratio = 1.5
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(aspect_ratio * scale, scale), dpi=90, )
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # experiment parameters
    n_experiments = 1000
    train_test_ratio = 64 / 128

    hypothesis_classes = {0: {"name": "class statistics",
                              "func": hypothesis_1,
                              "color": 'red'
                              },
                          1: {"name": "class statistics with emphasis on largest probability",
                              "func": hypothesis_2,
                              "color": 'green'
                              },
                          2: {"name": "always bet on highest probability",
                              "func": hypothesis_3,
                              "color": 'gray'
                              },
                          }

    sup_title = f'{n_experiments}'
    fig.suptitle(sup_title, fontsize=9)

    # dictionary which will hold the results for different hypotheses
    results_dict = {}

    # do the experiments
    for _ in range(n_experiments):

        # choose the number of 1's in the dataset
        for n1s in range(1, 128, 1):
            # prepare dataset
            y_train, y_test, k_lz = prepare_random_boolean_dataset(n1s, train_test_ratio)

            # do a "prediction" according to hypothesis id:
            for h_id in list(hypothesis_classes.keys()):
                prediction, _, _ = hypothesis_classes[h_id]['func'](y_train, len(y_test))

                # calculate prediction error
                error = calculate_error(y_test, prediction)

                # add the results in the dictionary
                if h_id not in results_dict.keys():
                    results_dict[h_id] = {}

                    # gather results about the experiments for later plots
                    if k_lz not in results_dict[h_id].keys():
                        results_dict[h_id][k_lz] = [error]
                    else:
                        results_dict[h_id][k_lz].append(error)
                else:
                    # gather results about the experiments for later plots
                    if k_lz not in results_dict[h_id].keys():
                        results_dict[h_id][k_lz] = [error]
                    else:
                        results_dict[h_id][k_lz].append(error)

    # plot the means over several experiments
    for h_id, klz_vs_err in results_dict.items():
        for complexity, errors in klz_vs_err.items():
            for a in axes:
                a.scatter(complexity, np.mean(errors), c=hypothesis_classes[h_id]['color'], marker='x')
                a.scatter(complexity, np.mean(errors) + np.std(errors), c=hypothesis_classes[h_id]['color'], marker='_')
                a.scatter(complexity, np.mean(errors) - np.std(errors), c=hypothesis_classes[h_id]['color'], marker='_')

    for a in axes:
        a.grid()
        a.set_ylabel("generalization error")
        a.set_yticks(np.linspace(0, 0.6, 7))
        a.set_xticks(np.linspace(0, 160, 5))

    # overlay experimental curves from the paper - extracted and aligned by hand
    current_file_path = Path(sys.argv[0]).parents[0].as_posix()
    root_dir = current_file_path

    axes[0].imshow(np.asarray(Image.open(Path(root_dir).as_posix() + '/Fig1c.png')),
                   extent=[1, 160, 0., 0.60], aspect='auto')
    axes[0].set_xlabel("LZ complexity, target function - Fig1c, m=64, ce, tanh")

    axes[1].imshow(np.asarray(Image.open(Path(root_dir).as_posix() + '/FigS10e.png')),
                   extent=[4.5, 162.5, 0., 0.60], aspect='auto')
    axes[1].set_xlabel("LZ complexity, target function - FigS10e, m=64, ce, tanh")
    axes[1].set_xlim((0, 158))

    axes[2].imshow(np.asarray(Image.open(Path(root_dir).as_posix() + '/FigS10f.png')),
                   extent=[4, 163, -0.005, 0.597], aspect='auto')
    axes[2].set_xlabel("LZ complexity, target function - FigS10f, m=64, ce, relu")
    axes[2].set_xlim((0, 158))

    axes[3].imshow(np.asarray(Image.open(Path(root_dir).as_posix() + '/FigS10g.png')),
                   extent=[4, 162.5, -0.005, 0.607], aspect='auto')
    axes[3].set_xlabel("LZ complexity, target function - FigS10g, m=64, mse, tanh")
    axes[3].set_xlim((0, 158))

    fig.tight_layout()
    plt.savefig(Path(root_dir).as_posix() + "/Overlays.png")
    plt.show()


def main():
    statistical_learners()
    return


if __name__ == '__main__':
    main()
