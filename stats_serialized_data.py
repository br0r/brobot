import sys
from scipy import stats
import matplotlib.pyplot as plt
from brobot.train.dataset import SerializedSequence


for f in sys.argv[1:]:
    plt.figure()
    plt.title(f)
    s = SerializedSequence(f, mem=True, multi=True)
    x, y = zip(*s.data)
    plt.hist(y, bins=30)
    print(f)
    print(stats.describe(y))
    s = None
plt.show()
