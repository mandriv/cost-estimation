import json
import argparse
import os

def getJSON(file):
    return json.load(file)

def init():
    parser = argparse.ArgumentParser(
        description='Executes Cost Estimation genetic programming algorithm based on config file'
        )
    parser.add_argument(
        '-d', '--data',
        help='Location of the weka dataset',
        metavar='data_path',
        type=open
        )
    parser.add_argument(
        '-c', '--config',
        help='Location of the JSON configuration file, if not specified config.json will be used.',
        metavar='config_path',
        type=argparse.FileType('r'),
        )


    args = parser.parse_args()
    return (args.data, getJSON(args.config))
