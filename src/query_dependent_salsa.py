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
num_iter = 8

if len(sys.argv) >= 2:
    topic_label = sys.argv[1]
if len(sys.argv) >= 3:
    num_iter = int(sys.argv[2])
if len(sys.argv) >= 5:
    nodesPath = sys.argv[3]
    edgesPath = sys.argv[4]
if len(sys.argv) == 1 or len(sys.argv) == 4 or len(sys.argv) > 5:
    print("Usage: spark-submit query_dependent_salsa.py topic_label [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath).withColumnRenamed("id:ID", "id")
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath)

edgesDF = edgesDF.select("src:START_ID", "dst:END_ID").withColumnRenamed("src:START_ID", "src_id").withColumnRenamed("dst:END_ID", "dst_id")

# Build neighborhood graph

# Get topic nodes
topic_nodesDF = nodesDF.filter(col("labels:LABEL") == topic_label).select("id")
edgesDF = edgesDF.withColumnRenamed("src:START_ID", "src_id").withColumnRenamed("dst:END_ID", "dst_id").select("src_id", "dst_id")

# Exclude edges involving two nodes that are not of the type specified by topic_label
edgesDF = edgesDF.join(topic_nodesDF, on=[(edgesDF.src_id == topic_nodesDF.id) | (edgesDF.dst_id == topic_nodesDF.id)], how="leftsemi")
edges = edgesDF.rdd.map(lambda edge: (edge[0], edge[1]))

# Get neighborhood graph nodes from edges
nodesDF = nodesDF.select("id").join(edgesDF, on=[(edgesDF.src_id == nodesDF.id) | (edgesDF.dst_id == nodesDF.id)], how="leftsemi")
nodes = nodesDF.rdd

# Compute out-degrees and in-degrees (simplified SALSA)
hubs = edges.map(lambda edge: (edge[0], 1)).reduceByKey(lambda x, y: x + y)
auths = edges.map(lambda edge: (edge[1], 1)).reduceByKey(lambda x, y: x + y)

hubs = normalize_rdd_sum(hubs)
auths = normalize_rdd_sum(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/SALSA/" + topic_label + "_dependent_SALSA_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/SALSA/" + topic_label + "_dependent_SALSA_authority_scores.txt")

# Take the top 50 hubs and authorities and multiply their score by 10 to draw them more clearly
hubs_dict = dict(hubs.mapValues(lambda score: score*10).take(50))
auths_dict = dict(auths.mapValues(lambda score: score*10).take(50))

# Sample nodes and edges from the graph
nodes_dict = dict(hubs.sample(False, 0.08, 81).collect())
edges_list = edges.sample(False, 0.01, 81).collect()

sc.stop()

print("Done!")

print("Top 10 hub scores:")
print(list(hubs_dict.items())[:10])
print("Top 10 authority scores:")
print(list(auths_dict.items())[:10])

print("Drawing graphs...")

draw_graphs(topic_label + "_dependent_SALSA", edges_list, nodes_dict, hubs_dict, auths_dict)