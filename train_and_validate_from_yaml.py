import argparse
import os
import yaml


def main(yaml_path):
    with open(yaml_path, 'r') as f:
        parameters = yaml.load(f.read())
    for key in parameters.keys():
        if type(parameters[key]) is list:
            parameters[key] = " ".join(['"%s"' % str(value) for value in parameters[key]])
        elif type(parameters[key]) is str:
            parameters[key] = '"%s"' % parameters[key]
    command = "python train_and_validate.py"
    for key in parameters.keys():
        value = parameters[key]
        if not key.startswith("--"):
            command += " %s" % value
        else:
            command += " %s %s" % (key, value)
    os.system(command)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str)
    args_dict = parser.parse_args(args)
    return args_dict

if __name__ == "__main__":
    yaml_path = parse_args().yaml_path
    main(yaml_path)
