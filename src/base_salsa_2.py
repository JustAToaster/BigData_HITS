# Version of SALSA with mutual update
# Ranking is the same as simplified SALSA, it is just of theoric interest

import sys
import math
from random import sample
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from draw_graphs import *

def initialize_salsa(nodesDF, edgesDF, num_nodes):
    out_degreesDF = edgesDF.groupBy("src_id").count().withColumnRenamed("count", "out_degree") \
    .select("src_id", "out_degree").withColumnRenamed("src_id", "id")
    in_degreesDF = edgesDF.groupBy("dst_id").count().withColumnRenamed("count", "in_degree") \
    .select("dst_id", "in_degree").withColumnRenamed("dst_id", "id")

    # Put in-degrees in edgesT and out-degrees in edges
    edgesT = edgesDF.join(in_degreesDF, edgesDF.dst_id == in_degreesDF.id) \
    .select("dst_id", "src_id", "in_degree").rdd.map(lambda x: (x[0], (x[1], x[2])))
    edges = edgesDF.join(out_degreesDF, edgesDF.src_id == out_degreesDF.id) \
    .select("src_id", "dst_id", "out_degree").rdd.map(lambda x: (x[0], (x[1], x[2])))
    auths = nodesDF.rdd.map(lambda node: (node[0], 1.0/math.sqrt(num_nodes)))

    hubs = auths
    return auths, hubs, edges, edgesT

def normalize_rdd_sum(rdd):
    rdd_sum = rdd.map(lambda x: (0, x[1])).reduceByKey(lambda x, y: x + y).collect()[0][1]
    return rdd.map(lambda x: (x[0], x[1] / rdd_sum))

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
nodesPath = '../data/nodes_elab.csv'
edgesPath = '../data/edges_elab.csv'
num_iter = 8
beta = 0.8

if len(sys.argv) >= 2:
    num_iter = int(sys.argv[1])
if len(sys.argv) >= 3:
    beta = float(sys.argv[2])
if len(sys.argv) >= 5:
    nodesPath = sys.argv[3]
    edgesPath = sys.argv[4]
if len(sys.argv) == 4 or len(sys.argv) > 5:
    print("Usage: spark-submit base_salsa_2.py [num_iter] [beta] [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath)

nodesDF = nodesDF.withColumnRenamed("id:ID", "id").select("id")
edgesDF = edgesDF.withColumnRenamed("src:START_ID", "src_id").withColumnRenamed("dst:END_ID", "dst_id")

nodes = nodesDF.rdd
num_nodes = nodes.count()
edgesDF = edgesDF.select("src_id", "dst_id")

#edges.saveAsTextFile("../outputs/edges.txt")
#edgesT.saveAsTextFile("../outputs/edgesT.txt"

auths, hubs, edges, edgesT = initialize_salsa(nodesDF, edgesDF, num_nodes)

print("Nodes:")
print(nodes.take(10))
print("Edges: ")
print(edges.take(10))

for i in range(num_iter):
    print("Iteration ", str(i+1))

    # Hub: for each node a, accumulate authority scores from all links of the form (a,b). Divide by each in-degree.
    hubs = edgesT.join(auths).map(lambda x: (x[1][0][0], x[1][1]/x[1][0][1])) \
    .reduceByKey(lambda x, y: x + y)

    # Authority: for each node b, accumulate hub scores from all links of the form (a,b). Divide by each out-degree.
    auths = edges.join(hubs).map(lambda x: (x[1][0][0], x[1][1]/x[1][0][1])) \
    .reduceByKey(lambda x, y: x + y)

    # Normalize scores
    hubs = normalize_rdd_sum(hubs)
    auths = normalize_rdd_sum(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/SALSA/baseSALSA2_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/SALSA/baseSALSA2_authority_scores.txt")

# Take the top 50 hubs and authorities
hubs_dict = dict((hubs).take(50))
auths_dict = dict((auths).take(50))

# Sample nodes and edges from the graph
nodes_dict = dict(hubs.sample(False, 0.01, 81).collect())
edges_list = edges.sample(False, 0.004, 81).collect()

sc.stop()

print("Done!")

print("Top 10 hub scores:")
print(list(hubs_dict.items())[:10])
print("Top 10 authority scores:")
print(list(auths_dict.items())[:10])

print("Drawing graphs...")

draw_graphs("baseSALSA", edges_list, nodes_dict, hubs_dict, auths_dict)