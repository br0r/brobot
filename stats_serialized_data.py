import sys
import matplotlib.pyplot as plt
from brobot.train.dataset import SerializedSequence


for f in sys.argv[1:]:
    plt.figure()
    plt.title(f)
    s = SerializedSequence(f, 128, mem=True, multi=True)
    x, y = zip(*s.data)
    plt.hist(y, bins=30)
    s = None
plt.show()
