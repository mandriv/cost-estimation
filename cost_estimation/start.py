from .utils import cli, data

def main():
    # get config and dataset
    (dataset, config) = cli.init()
    # get train and test data
    (train_data, test_data) = data.parse(dataset, config)
    

if __name__ == '__main__':
    main()
