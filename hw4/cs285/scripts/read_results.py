import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []

    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    logdir = 'data/hw4_q4_reacher_ensemble1_reacher-cs285-v0_03-11-2021_13-47-09/events*'
    eventfile = glob.glob(logdir)[0]
    X1, Y1 = get_section_results(eventfile)
    X1 = np.arange(0, 15, 1)

    logdir = 'data/hw4_q4_reacher_ensemble3_reacher-cs285-v0_03-11-2021_14-48-30/events*'
    eventfile = glob.glob(logdir)[0]
    X2, Y2 = get_section_results(eventfile)

    logdir = 'data/hw4_q4_reacher_ensemble5_reacher-cs285-v0_03-11-2021_14-48-57/events*'
    eventfile = glob.glob(logdir)[0]
    X3, Y3 = get_section_results(eventfile)

    # data = np.array([Y1, Y2, Y3])
    # Y_ave = np.average(data, axis=0)

    # logdir = 'data/q3_batch_size_128_LunarLander-v3_19-10-2021_00-38-12/events*'
    # eventfile = glob.glob(logdir)[0]
    # X4, Y4 = get_section_results(eventfile)

    # logdir = 'data/q2_dqn_2_LunarLander-v3_18-10-2021_20-10-21/events*'
    # eventfile = glob.glob(logdir)[0]
    # X5, Y5 = get_section_results(eventfile)
    #
    # logdir = 'data/q2_dqn_3_LunarLander-v3_18-10-2021_21-02-00/events*'
    # eventfile = glob.glob(logdir)[0]
    # X6, Y6 = get_section_results(eventfile)
    #
    # data = np.array([Y4, Y5, Y6])
    # Y_ave_d = np.average(data, axis=0)

    # ax1.plot(x, y_hopper_train, 'g', label="train_mean")
    # ax1.plot(x, y_hopper_eval, 'b', label="eval_mean")

    # plt.plot(X1, Y1, 'b')
    plt.plot(X1, Y1, 'r', label="ensemble1")
    plt.plot(X1, Y2, 'g', label="ensemble3")
    plt.plot(X1, Y3, 'b', label="ensemble5")

    plt.legend(loc="lower right")

    plt.xlabel('iter_num')
    plt.ylabel('return')
    plt.title("q4 different ensemble")
    plt.show()
    # for i, (x, y) in enumerate(zip(X, Y)):
    #     print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))