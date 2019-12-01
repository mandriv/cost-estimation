import arff

def parse(file, config):
    inputs = config['inputs']
    output = config['output']
    train_frac = config['train_frac']

    result = arff.load(file.name)
    data = []
    for row in result:
        element = {
            'inputs': [],
            'output': row[output]
        }
        for input in inputs:
            element['inputs'].append(row[input])
        data.append(element)
    data_size = len(data)
    train_data_size = int(train_frac * data_size)
    test_data_size = data_size - train_data_size
    train_data = []
    test_data = []
    for _ in range(train_data_size):
        train_data.append(data.pop(0))
    for _ in range(test_data_size):
        test_data.append(data.pop(0))
    return (train_data, test_data)
