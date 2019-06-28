import os
import yaml

from visage import Visage


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), '../config.yml'), 'r') as configFile:
        cfg = yaml.load(configFile)

    visage = Visage(cfg)
    visage.run()
