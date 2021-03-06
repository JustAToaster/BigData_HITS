import sys
import math
from random import sample
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from draw_graphs import *

def initialize_hits(nodes):
    num_nodes = nodes.count()
    auths = nodes.map(lambda node: (node[0], 1.0/math.sqrt(num_nodes)))
    hubs = auths
    return auths, hubs

def normalize_rdd(rdd):
    rdd_norm_squared = rdd.map(lambda x: (0, x[1]*x[1])).reduceByKey(lambda x, y: x + y).collect()[0][1]
    rdd_norm = math.sqrt(rdd_norm_squared)
    return rdd.map(lambda x: (x[0], x[1] / rdd_norm))

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
    print("Usage: spark-submit random_teleport_hits.py [num_iter] [beta] [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath)

nodesDF = nodesDF.withColumnRenamed("id:ID", "id").select("id")

nodes = nodesDF.rdd
num_nodes = nodes.count()

edges = edgesDF.select("src:START_ID", "dst:END_ID").rdd.map(lambda edge: (edge[0], edge[1]))
edgesT = edges.map(lambda edge: (edge[1], edge[0]))

#edges.saveAsTextFile("../outputs/edges.txt")
#edgesT.saveAsTextFile("../outputs/edgesT.txt"

auths, hubs = initialize_hits(nodes)

print("Nodes:")
print(nodes.take(10))
print("Edges: ")
print(edges.take(10))

# hubSum = authSum = num_nodes/(math.sqrt(num_nodes))

for i in range(num_iter):
    print("Iteration ", str(i+1))

    # Hub: for each node a, accumulate authority scores from all links of the form (a,b).
    # Sum contribution from stochastic edges.
    hubs = edgesT.join(auths).map(lambda x: (x[1][0], x[1][1])) \
    .reduceByKey(lambda x, y: x + y) \
    .mapValues(lambda score: (beta*score + ((1-beta)/num_nodes)))

    # Authority: for each node b, accumulate hub scores from all links of the form (a,b).
    # Sum contribution from stochastic edges.
    auths = edges.join(hubs).map(lambda x: (x[1][0], x[1][1])) \
    .reduceByKey(lambda x, y: x + y) \
    .mapValues(lambda score: beta*score + ((1-beta)/num_nodes))

    # Normalize scores
    hubs = normalize_rdd(hubs)
    auths = normalize_rdd(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/teleport_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/teleport_authority_scores.txt")

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

draw_graphs("teleportHITS", edges_list, nodes_dict, hubs_dict, auths_dict)