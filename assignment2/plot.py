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

def plot_many_curves(iterations, valLabels, valIndex, title, yLabel):
    plt.figure()
    plt.title(title)

    plt.xlabel("# Iterations")
    plt.ylabel(yLabel)
    
    plt.grid()

    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']
    for i in range(len(valLabels)):
        val = valLabels[i][valIndex]
        label = valLabels[i][-1]
        color = colors[i]

        plt.plot(iterations, val, 'o-', color=color,
             label=label)

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
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '.jpg')
    plot.close()

    plot = plot_timing_curve(df['iteration'], df['elapsed'], title + ' Training Time')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.jpg')
    plot.close()
    # plot.show()

def get_df(path):
    return pandas.read_csv('./jtay-code/' + path)

def plot_peaks(title):
    valLabels = []

    df1 = get_df('CONTPEAKS/CONTPEAKS_RHC_4_LOG.txt')
    valLabels.append((df1['fitness'], df1['time'], df1['fevals'], 'Randomized Hill Climbing'))

    df2 = get_df('CONTPEAKS/CONTPEAKS_SA0.15_1_LOG.txt')
    valLabels.append((df2['fitness'], df2['time'], df2['fevals'], 'Simulated Annealing'))

    df3 = get_df('CONTPEAKS/CONTPEAKS_GA100_30_30_1_LOG.txt')
    valLabels.append((df3['fitness'], df3['time'], df3['fevals'], 'Genetic Algorithm'))

    df4 = get_df('CONTPEAKS/CONTPEAKS_MIMIC100_50_0.5_1_LOG.txt')
    valLabels.append((df4['fitness'], df4['time'], df4['fevals'], 'MIMIC'))

    plot = plot_many_curves(df1['iterations'], valLabels, 0, title + ' Fitness', 'Fitness Function')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_fitness.jpg')

    plot = plot_many_curves(df1['iterations'], valLabels, 1, title + ' Time', 'Training Time (s)')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.jpg')

    plot = plot_many_curves(df1['iterations'], valLabels, 2, title + ' Evals', 'Function Evals')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_evals.jpg')

    plot.close()

def plot_tsp(title):
    valLabels = []

    df1 = get_df('TSP/TSP_RHC_1_LOG.txt')
    valLabels.append((df1['fitness'], df1['time'], df1['fevals'], 'Randomized Hill Climbing'))

    df2 = get_df('TSP/TSP_SA0.55_1_LOG.txt')
    valLabels.append((df2['fitness'], df2['time'], df2['fevals'], 'Simulated Annealing'))

    df3 = get_df('TSP/TSP_GA100_30_30_1_LOG.txt')
    valLabels.append((df3['fitness'], df3['time'], df3['fevals'], 'Genetic Algorithm'))

    df4 = get_df('TSP/TSP_MIMIC100_50_0.5_1_LOG.txt')
    valLabels.append((df4['fitness'], df4['time'], df4['fevals'], 'MIMIC'))

    plot = plot_many_curves(df1['iterations'], valLabels, 0, title + ' Fitness', 'Fitness Function')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_fitness.jpg')

    plot = plot_many_curves(df1['iterations'], valLabels, 1, title + ' Time', 'Training Time (s)')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.jpg')

    plot = plot_many_curves(df1['iterations'], valLabels, 2, title + ' Evals', 'Function Evals')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_evals.jpg')

    plot.close()

def plot_flipflop(title):
    valLabels = []

    df1 = get_df('FLIPFLOP/FLIPFLOP_RHC_1_LOG.txt')
    valLabels.append((df1['fitness'], df1['time'], df1['fevals'], 'Randomized Hill Climbing'))

    df2 = get_df('FLIPFLOP/FLIPFLOP_SA0.15_1_LOG.txt')
    valLabels.append((df2['fitness'], df2['time'], df2['fevals'], 'Simulated Annealing'))

    df3 = get_df('FLIPFLOP/FLIPFLOP_GA100_30_30_1_LOG.txt')
    valLabels.append((df3['fitness'], df3['time'], df3['fevals'], 'Genetic Algorithm'))

    df4 = get_df('FLIPFLOP/FLIPFLOP_MIMIC100_50_0.5_1_LOG.txt')
    valLabels.append((df4['fitness'], df4['time'], df4['fevals'], 'MIMIC'))

    plot = plot_many_curves(df1['iterations'], valLabels, 0, title + ' Fitness', 'Fitness Function')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_fitness.jpg')

    plot = plot_many_curves(df1['iterations'], valLabels, 1, title + ' Time', 'Training Time (s)')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.jpg')

    plot = plot_many_curves(df1['iterations'], valLabels, 2, title + ' Evals', 'Function Evals')
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_evals.jpg')

    plot.close()

plot('NN_OUTPUT/BACKPROP_LOG.txt', 'Backprop NN')

plot('NN_OUTPUT/RHC_LOG.txt', 'Randomized Hill Climbing NN')

plot('NN_OUTPUT/SA0.15_LOG.txt', 'Simulated Annealing Cooling .15 NN')
plot('NN_OUTPUT/SA0.35_LOG.txt', 'Simulated Annealing Cooling .35 NN')
plot('NN_OUTPUT/SA0.55_LOG.txt', 'Simulated Annealing Cooling .55 NN')
plot('NN_OUTPUT/SA0.7_LOG.txt', 'Simulated Annealing Cooling .7 NN')
plot('NN_OUTPUT/SA0.95_LOG.txt', 'Simulated Annealing Cooling .95 NN')

plot('NN_OUTPUT/GA_50_10_10_LOG.txt', 'Genetic 10 Mate, 10 Mutate NN')
plot('NN_OUTPUT/GA_50_10_20_LOG.txt', 'Genetic 10 Mate, 20 Mutate NN')
plot('NN_OUTPUT/GA_50_20_10_LOG.txt', 'Genetic 20 Mate, 10 Mutate NN')
plot('NN_OUTPUT/GA_50_20_20_LOG.txt', 'Genetic 20 Mate, 20 Mutate NN')

plot_peaks('Continous Peaks')
plot_tsp('Traveling Salesman')
plot_flipflop('Flip Flop')


# iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,elapsed
# 0,0.293247932778,0.292920659286,0.300509176221,0.413502881713,0.414196242171,0.398981670061,0.588158795
# 10,0.068612203824,0.0699373695198,0.0645621181263,0.862775592352,0.86012526096,0.870875763747,7.978658689
# 20,0.0686569920033,0.0699354731105,0.0641716006192,0.862684109414,0.86012526096,0.871690427699,15.104390501
