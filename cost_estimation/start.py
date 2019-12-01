import matplotlib.pyplot as plt

from .utils import cli, data
from .gp import gp


def main():
    # get config and dataset
    (dataset, config) = cli.init()
    # get train and test data
    (train_data, test_data, inputs) = data.parse(dataset, config)
    # run gp
    (best_individual, logbook) = gp.run(config, inputs, train_data, test_data)
    # draw statistics graph
    gp.draw_statistics(logbook)
    # draw best individual
    gp.draw_graph(best_individual)
    # show all figures
    plt.show()


if __name__ == '__main__':
    main()
