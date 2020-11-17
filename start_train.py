import sys
from brobot.train.dataset import createRecord, create_dataset
from brobot.train.train import run


createRecord(sys.argv[1])
run()
