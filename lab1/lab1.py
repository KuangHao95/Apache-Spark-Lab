# Kuang Hao A0191488N
import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
text_file = sc.textFile(sys.argv[1])
# Get initials to form a new RDD
def get_initial(x):
    if len(x) < 1:
        return
    # Return initial if begins with an alphabet.
    elif x[0].isalpha():
        return x[0]
    else:
        return 

words = text_file.flatMap(lambda l: re.split(" ",l)) \
.map(lambda cap: cap.title())
# Resonstruct RDD with only initials.
initials = words.map(get_initial)
counts = initials.map(lambda w: (w, 1)) \
.reduceByKey(lambda n1, n2: n1 + n2)
# Output to only one file.
counts.repartition(1).saveAsTextFile(sys.argv[2])
sc.stop()