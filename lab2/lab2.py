# Python 2.7.10
# Spark version: 2.4.0
# Kuang Hao
# Put 'lab2.py' in the same folder with 'stopwords.txt' and folder: datafiles.
import re
import sys
import io
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf = conf)

# Read all txt files.
text_files = sc.wholeTextFiles("./datafiles/*")

# Remove stopwords(step1) and count words(step2).
def word_remove(context):
    # Record file names
    path = context[0].split("/")[-1]
    # Delete all punctuations
    words = re.sub('[^a-z]+',' ',context[1].lower()).split()
    # Remove words according to stopwords
    refinedwords = [word for word in words if word not in stopwords]
    # Extract file names
    file = path.split(".")[0].strip('f')
    lists = [x + '#' + file for x in refinedwords]
    # Return format: (word#file_number, count)
    return lists

# Reconstruct words from (word#file_number, count) to (word, file_number#count)
def word_reconstruct(context):
    word_filename, count = context
    word, filename = word_filename.split('#')
    # Need to count if a word appears 10 times, so I put other things to value
    return (word, '{0}#{1}'.format(filename, count))

# Count how many times a word appears in all files
def word_count(context):
    word = context[0]
    filename_count_pairs = []
    counter = 0
    for filename_count in context[1]:
        # Counter documents how many files a word is in.
        counter = counter + 1
        filename, count = filename_count.split('#')
        filename_count_pairs.append((filename, count))
    result = []
    for (filename, count) in filename_count_pairs:
        word_filename = '{0}#{1}'.format(word, filename)
        count_docswithword = '{0}#{1}'.format(count, counter)
        # Extract the words appearing in all files
        if(counter == 10):
            result.append((word_filename.split('#')[0], int(count_docswithword.split('#')[0])))
    return result

data = io.open("./stopwords.txt","r",encoding="utf-8")
#stopwords = [x.strip() for x in data.readlines()]
stopwords = re.sub('[^a-z]+',' ',''.join(data)).split()
# Count words like I have done in Lab 1
step2_count = text_files.flatMap(word_remove)\
    .map(lambda w: (w,1)) \
    .reduceByKey(lambda x,y: x+y)
step2_count.repartition(1).saveAsTextFile('output_step2')

# Find words that appear in all files
step4_word = step2_count.map(word_reconstruct) \
    .groupByKey() \
        .flatMap(word_count)
step4_res = step4_word.reduceByKey(lambda x,y: min(x, y))
# x is the final target
x = sc.parallelize(sorted(step4_res.take(100), key=lambda x: x[1], reverse = True))
x.repartition(1).saveAsTextFile('output_step4')
sc.stop()
