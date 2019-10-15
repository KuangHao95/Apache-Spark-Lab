# Python 2.7.10 (compatible with 3.7+)
# Spark version: 2.4.0
# Kuang Hao
# Put 'lab3.py' in the same folder with 'stopwords.txt', 'query.txt' and folder: datafiles.
# The result will be stored in 'result.txt'.
from pyspark import SparkConf, SparkContext
from os import listdir
import io
import re
import math

# Step 1. Compute term frequency (TF) of every word in a document.
def compute_tf(context, word):
    if word:
        return context.flatMap(word_remove) \
        .map(lambda w: ((w, i), 1)) \
        .reduceByKey(lambda n1, n2: n1 + n2)
    else:
        tf = context.flatMap(word_remove) \
        .map(lambda w: (w, 1)) \
        .reduceByKey(lambda n1, n2: n1 + n2)
        vector = tf.collect()
        words = tf.keys().collect()
        scale = math.sqrt(tf.reduce(lambda x,y: x[1]**2+y[1]**2))
        return (tf, vector, words, scale)
    
# Compute DF.
def compute_df(context):
    return context.map(lambda x: (x[0][0], 1)) \
        .reduceByKey(lambda n1,n2: n1+n2)

# Remove stopwords.
def word_remove(context):
    words = context.split()
    words = [''.join([i for i in w if i.isalpha()]).lower() for w in words ]
    lists = [i for i in words if i not in stopwords]
    return lists


# Step 2. Compute TF-IDF of every word w.r.t a document.
def tf_idf(x):
    value = (1 + math.log10(x[1][0]))*math.log10(length / x[1][1])
    return (x[0], value)

def get_tfidf(context,i):
    df_rdd = context.map(lambda x: ((x[0],i),x[1]))
    doc_tfidf = tf_rdd.join(df_rdd).map(tf_idf)
    S_rt = math.sqrt(doc_tfidf.map(lambda x: (x[0][1], x[1])).values().map(lambda x: x*x).sum())
    return (doc_tfidf, S_rt)

# Step 3. Compute normalized TF-IDF of every word w.r.t. a document.
def normalize_tfidf(context, s):
    return context.map(lambda x: (x[0], x[1] / s))

# Step 4. Compute the relevance of each document w.r.t a query.
def get_relevance(context, words, tf_query):
    product = context.filter(lambda x: x[0][0] in words) \
        .map(lambda x: (x[0][0], x[1])).join(tf_query) \
        .map(lambda x: (x[0],x[1][0] * x[1][1])).values().sum()
    # doc_scale should be 1 since it is normalized.
    doc_scale = context.values().map(lambda x: x ** 2).sum()
    temp_relevance = sc.parallelize([(files_names[i], product/doc_scale/q_scale)])
    return relevance.union(temp_relevance)
   
# Step 5. Sort and get top-k documents.
def sort_n_result(context):
    result = context.sortBy(lambda x:x[1],ascending = False).collect()
    with open('./result.txt','w') as f:
        f.write('<docID> ' + ' <relevance score>' + '\n')
        for i in result:
            f.write('<' + str(i[0]) + '> <' + str(i[1]) + '>')
            f.write('\n')


files_names = listdir('datafiles')
length = len(files_names)

# Open stopwords
data = io.open("./stopwords.txt", "r", encoding = "utf-8")
stopwords = re.sub('[^a-z]+',' ',''.join(data)).split()

conf = SparkConf()
sc = SparkContext(conf=conf)

# Get TF, DF
tf_rdd = sc.parallelize([])
for i in range(length):
    tf = compute_tf(sc.textFile(f"datafiles/{files_names[i]}"), True)
    tf_rdd = sc.union([tf_rdd, tf])
df = compute_df(tf_rdd)

# Get Tf of query
(q_tf, q_vector, q_words, q_scale) = compute_tf(sc.textFile("./query.txt"), False)

relevance = sc.parallelize([])
for i in range(length):
    # Step 2
    (main_tfidf, ss) = get_tfidf(df,i)
    # Step 3
    norm_tfidf = normalize_tfidf(main_tfidf, ss)
    # Step 4
    relevance = get_relevance(norm_tfidf, q_words, q_tf)
# Step 5
sort_n_result(relevance)  
sc.stop()
