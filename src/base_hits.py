import sys
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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

if len(sys.argv) == 2:
    num_iter = int(sys.argv[1])
if len(sys.argv) == 3 or len(sys.argv) > 4:
    print("Usage: spark-submit base_hits.py [num_iter] [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath)

nodes = nodesDF.rdd
edges = edgesDF.select("src:START_ID", "dst:END_ID").rdd.map(lambda edge: (edge[0], edge[1]))
edgesT = edges.map(lambda edge: (edge[1], edge[0]))

#edges.saveAsTextFile("../outputs/edges.txt")
#edgesT.saveAsTextFile("../outputs/edgesT.txt")

auths, hubs = initialize_hits(nodes)

print("Nodes:")
print(nodes.take(10))
print("Edges: ")
print(edges.take(10))

print("Hub scores:")
print(hubs.take(10))
print("Authority scores:")
print(auths.take(10))

for i in range(num_iter):
    print("Iteration ", str(i+1))

    # Hub: for each node a, accumulate authority scores from all links of the form (a,b)
    hubs = edgesT.join(auths).map(lambda x: (x[1][0], x[1][1])).reduceByKey(lambda x, y: x + y)
    
    # Authority: for each node b, accumulate hub scores from all links of the form (a,b)
    auths = edges.join(hubs).map(lambda x: (x[1][0], x[1][1])).reduceByKey(lambda x, y: x + y)

    # Normalize scores
    hubs = normalize_rdd(hubs)
    auths = normalize_rdd(auths)


print("Hub scores:")
print(hubs.take(10))
print("Authority scores:")
print(auths.take(10))

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

hubs.saveAsTextFile("../outputs/base_hub_scores.txt")
auths.saveAsTextFile("../outputs/base_authority_scores.txt")

sc.stop()