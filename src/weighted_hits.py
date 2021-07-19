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
weight_col = "mrho:double"

if len(sys.argv) >= 2:
    num_iter = int(sys.argv[1])
if len(sys.argv) >= 3:
    weight_col = sys.argv[2]
if len(sys.argv) >= 5:
    nodesPath = sys.argv[3]
    edgesPath = sys.argv[4]
if len(sys.argv) == 4 or len(sys.argv) > 5:
    print("Usage: spark-submit weighted_hits.py [num_iter] [weight_col] [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath) \
    .select("src:START_ID", "dst:END_ID", weight_col)

nodes = nodesDF.rdd
edges = edgesDF.rdd.map(lambda edge: (edge[0], (edge[1], edge[2])))

# To build the transposed edge matrix with weights, we first transpose the dataframe with edges (a,b) to (b,a)
edgesT_DF = edgesDF.withColumnRenamed("src:START_ID", "dst:END_ID2") \
    .withColumnRenamed("dst:END_ID", "src:START_ID") \
    .withColumnRenamed("dst:END_ID2", "dst:END_ID") \
    .select("src:START_ID", "dst:END_ID")
# Then we join with the original dataframe with weights and fill non present returning edges with 0
edgesT = edgesT_DF.join(edgesDF, ['src:START_ID', 'dst:END_ID'], how="leftouter") \
    .na.fill(0).rdd.map(lambda edge: (edge[0], (edge[1], edge[2])))

#edges.saveAsTextFile("../outputs/edges.txt")
#edgesT.saveAsTextFile("../outputs/edgesT.txt")

auths, hubs = initialize_hits(nodes)

print("Hub scores:")
print(hubs.take(10))
print("Authority scores:")
print(auths.take(10))

for i in range(num_iter):
    print("Iteration ", str(i+1))

    # Hub: for each node a, accumulate weighted authority scores from all links of the form (a,b)
    hubs = edgesT.join(auths).map(lambda x: (x[1][0][0], x[1][0][1]*x[1][1])).reduceByKey(lambda x, y: x + y)
    
    # Authority: for each node b, accumulate weighted hub scores from all links of the form (a,b)
    auths = edges.join(hubs).map(lambda x: (x[1][0][0], x[1][0][1]*x[1][1])).reduceByKey(lambda x, y: x + y)

    # Normalize scores
    hubs = normalize_rdd(hubs)
    auths = normalize_rdd(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/weighted_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/weighted_authority_scores.txt")

# Take the top 50 hubs and authorities
hubs_dict = dict(hubs.take(50))
auths_dict = dict(auths.take(50))

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

draw_graphs("weightedHITS", edges_list, nodes_dict, hubs_dict, auths_dict)