import sys
import math
import os
import re

from random import sample
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from draw_graphs import *

def normalize_rdd_sum(rdd):
    rdd_sum = rdd.map(lambda x: (0, x[1])).reduceByKey(lambda x, y: x + y).collect()[0][1]
    return rdd.map(lambda x: (x[0], x[1] / rdd_sum))

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
nodesPath = '../data/nodes_elab.csv'
edgesPath = '../data/edges_elab.csv'
weight_col = "mrho:double"

if len(sys.argv) >= 2:
    weight_col = sys.argv[1]
if len(sys.argv) >= 4:
    nodesPath = sys.argv[2]
    edgesPath = sys.argv[3]
if len(sys.argv) == 3 or len(sys.argv) > 4:
    print("Usage: spark-submit weighted_salsa.py [weight_col] [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath)

edgesDF = edgesDF.select("src:START_ID", "dst:END_ID", weight_col).withColumnRenamed("src:START_ID", "src_id").withColumnRenamed("dst:END_ID", "dst_id")

nodes = nodesDF.rdd
edges = edgesDF.rdd

# Compute out-degrees and in-degrees (simplified SALSA)
hubs = edges.map(lambda edge: (edge[0], edge[2])).reduceByKey(lambda x, y: x + y)
auths = edges.map(lambda edge: (edge[1], edge[2])).reduceByKey(lambda x, y: x + y)

hubs = normalize_rdd_sum(hubs)
auths = normalize_rdd_sum(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/SALSA/weightedSALSA_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/SALSA/weightedSALSA_authority_scores.txt")

# Take the top 50 hubs and authorities and multiply their score by 20 to draw them more clearly
hubs_dict = dict(hubs.mapValues(lambda score: score*20).take(50))
auths_dict = dict(auths.mapValues(lambda score: score*20).take(50))

# Sample nodes and edges from the graph
nodes_dict = dict(hubs.sample(False, 0.01, 81).collect())
edges_list = edges.map(lambda edge: (edge[0], edge[1])).sample(False, 0.004, 81).collect()

sc.stop()

print("Done!")

print("Top 10 hub scores:")
print(list(hubs_dict.items())[:10])
print("Top 10 authority scores:")
print(list(auths_dict.items())[:10])

print("Drawing graphs...")

draw_graphs("weightedSALSA", edges_list, nodes_dict, hubs_dict, auths_dict)