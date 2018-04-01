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

def get_df(path):
    return pandas.read_csv('./results/' + path)

def print_latex_row(vals, sig=3):
    sigString = '{:0.' + str(sig) + 'f}'
    vals = ' & '.join(list(map(lambda x: sigString.format(x), vals)))
    print(vals + ' \\\\ \hline')

def save_plot(plot, title):
    plot.savefig('analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '.jpg')

def plot_sse(df, valName, title):
    clusters = df[df.columns[0]]
    vals = df[valName]
    print_latex_row(vals, 0)

    plt.figure()
    plt.title(title) 

    plt.xlabel("Number of Clusters")
    plt.ylabel("Inner Cluster Sum of Squared Errors")
    
    plt.grid()
    
    plt.plot(clusters, vals, 'o-', color="r",
             label="Sum of Squared Errors")

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_clustering_accuracy(folderFile, title): 
    ami = get_df(folderFile + ' adjMI.csv')
    acc = get_df(folderFile + ' acc.csv')

    clusters = ami.columns[1:].values

    gmm_ami = ami.loc[0].values[1:]
    gmm_acc = acc.loc[0].values[1:]

    k_means_ami = ami.loc[1].values[1:]
    k_means_acc = acc.loc[1].values[1:]

    print_latex_row(k_means_ami)
    print_latex_row(k_means_acc)
    print_latex_row(gmm_ami)
    print_latex_row(gmm_acc)

    newTitle = title + ' Scoring'
    plt.figure()
    plt.title(newTitle) 
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(clusters, k_means_ami, 'o-', color="g", label="k-means Adjusted Mutual Info Score")
    plt.plot(clusters, k_means_acc, 'o-', color="b", label="k-means Cluster Accuracy")

    plt.plot(clusters, gmm_ami, 'o-', color="r", label="Expectation Maximization Adjusted Mutual Info Score")
    plt.plot(clusters, gmm_acc, 'o-', color="black", label="Expectation Maximization Cluster Accuracy")

    plt.legend(loc="best")

    save_plot(plt, newTitle)
    plt.close()


def plot_clustering():
    sse = get_df('clustering/SSE.csv')

    plot_sse(sse, 'perm SSE (left)', 'Perm Visa SSE')
    plot_sse(sse, 'housing SSE (left)', 'Housing SSE')

    plot_clustering_accuracy('clustering/Housing', 'Housing Clustering')
    plot_clustering_accuracy('clustering/Perm', 'Permanent Visa Clustering')


plot_clustering()

# plot('NN_OUTPUT/BACKPROP_LOG.txt', 'Backprop NN')

# plot('NN_OUTPUT/RHC_LOG.txt', 'Randomized Hill Climbing NN')

# plot('NN_OUTPUT/SA0.15_LOG.txt', 'Simulated Annealing Cooling .15 NN')
# plot('NN_OUTPUT/SA0.35_LOG.txt', 'Simulated Annealing Cooling .35 NN')
# plot('NN_OUTPUT/SA0.55_LOG.txt', 'Simulated Annealing Cooling .55 NN')
# plot('NN_OUTPUT/SA0.7_LOG.txt', 'Simulated Annealing Cooling .7 NN')
# plot('NN_OUTPUT/SA0.95_LOG.txt', 'Simulated Annealing Cooling .95 NN')

# plot('NN_OUTPUT/GA_50_10_10_LOG.txt', 'Genetic 10 Mate, 10 Mutate NN')
# plot('NN_OUTPUT/GA_50_10_20_LOG.txt', 'Genetic 10 Mate, 20 Mutate NN')
# plot('NN_OUTPUT/GA_50_20_10_LOG.txt', 'Genetic 20 Mate, 10 Mutate NN')
# plot('NN_OUTPUT/GA_50_20_20_LOG.txt', 'Genetic 20 Mate, 20 Mutate NN')

# plot_peaks('Continous Peaks')
# plot_tsp('Traveling Salesman')
# plot_flipflop('Flip Flop')


# iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,elapsed
# 0,0.293247932778,0.292920659286,0.300509176221,0.413502881713,0.414196242171,0.398981670061,0.588158795
# 10,0.068612203824,0.0699373695198,0.0645621181263,0.862775592352,0.86012526096,0.870875763747,7.978658689
# 20,0.0686569920033,0.0699354731105,0.0641716006192,0.862684109414,0.86012526096,0.871690427699,15.104390501
