import matplotlib.pyplot as plt
import pandas


def plot_learning_curve(iterations, train_scores, test_scores, title):
#     _, _, test_scores_base = base_curve

    plt.figure()
    plt.title(title)
    plt.ylim((.3, 1.01))
    
    # if datasetNum == 1:
    #     plt.ylim((.55, 1.01))

    plt.xlabel("# Iterations")
    plt.ylabel("Score")
    
    plt.grid()
    
    plt.plot(iterations, train_scores, 'o-', color="r",
             label="Training score")
#     plt.plot(train_sizes, test_scores_base_mean, 'o-', color="b",
#              label="Test Score without CV")
    plt.plot(iterations, test_scores, 'o-', color="g",
             label="Test Score")

    plt.legend(loc="best")
    return plt

def plot_timing_curve(iterations, timeDuration, title):
#     _, _, test_scores_base = base_curve

    plt.figure()
    plt.title(title)
    # plt.ylim((.3, 1.01))
    
    # if datasetNum == 1:
    #     plt.ylim((.55, 1.01))

    plt.xlabel("# Iterations")
    plt.ylabel("Training Time (s)")
    
    plt.grid()
    
    plt.plot(iterations, timeDuration, 'o-', color="r",
             label="Duration")

    plt.legend(loc="best")
    return plt


def plot(dataset_path, title):
    df = pandas.read_csv('./jtay-code/' + dataset_path)
    plot = plot_learning_curve(df['iteration'], df['acc_tst'], df['acc_trg'], title)
    plot.savefig('analysis/plots/' + title.replace(' ', '_').lower() + '.jpg')
    plot = plot_timing_curve(df['iteration'], df['elapsed'], title + ' Training Time')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').lower() + '_time.jpg')
    # plot.show()


plot('NN_OUTPUT/BACKPROP_LOG.txt', 'Backprop NN')
plot('NN_OUTPUT/RHC_LOG.txt', 'Randomized Hill Climbing NN')
plot('NN_OUTPUT/SA0.15_LOG.txt', 'Simulated Annealing Cooling .15 NN')
plot('NN_OUTPUT/SA0.35_LOG.txt', 'Simulated Annealing Cooling .35 NN')
plot('NN_OUTPUT/SA0.55_LOG.txt', 'Simulated Annealing Cooling .55 NN')
plot('NN_OUTPUT/SA0.7_LOG.txt', 'Simulated Annealing Cooling .7 NN')
plot('NN_OUTPUT/SA0.95_LOG.txt', 'Simulated Annealing Cooling .95 NN')


# iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,elapsed
# 0,0.293247932778,0.292920659286,0.300509176221,0.413502881713,0.414196242171,0.398981670061,0.588158795
# 10,0.068612203824,0.0699373695198,0.0645621181263,0.862775592352,0.86012526096,0.870875763747,7.978658689
# 20,0.0686569920033,0.0699354731105,0.0641716006192,0.862684109414,0.86012526096,0.871690427699,15.104390501
