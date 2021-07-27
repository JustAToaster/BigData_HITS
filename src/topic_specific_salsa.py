import sys
import math
from random import sample
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from draw_graphs import *

def initialize_topic_specific_hits(nodesDF, edgesDF, num_topic_nodes):
    out_degreesDF = edgesDF.groupBy("src_id").count().withColumnRenamed("count", "out_degree") \
    .select("src_id", "out_degree").withColumnRenamed("src_id", "id")
    in_degreesDF = edgesDF.groupBy("dst_id").count().withColumnRenamed("count", "in_degree") \
    .select("dst_id", "in_degree").withColumnRenamed("dst_id", "id")

    nodesDF_label = nodesDF.select("id", "topic_specific")

    # Put in-degrees in edgesT and out-degrees in edges
    edgesT = edgesDF.join(in_degreesDF, edgesDF.dst_id == in_degreesDF.id) \
    .select("dst_id", "src_id", "in_degree").rdd.map(lambda x: (x[0], (x[1], x[2])))
    edges = edgesDF.join(out_degreesDF, edgesDF.src_id == out_degreesDF.id) \
    .select("src_id", "dst_id", "out_degree").rdd.map(lambda x: (x[0], (x[1], x[2])))
    auths = nodesDF_label.rdd.map(lambda node: (node[0], 0 if node[1] == 0 else 1.0/(2*num_topic_nodes)))
    
    hubs = auths
    return auths, hubs, edges, edgesT, nodesDF_label.rdd

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
topic_label = ""

if len(sys.argv) >= 2:
    topic_label = sys.argv[1]
if len(sys.argv) >= 3:
    num_iter = int(sys.argv[2])
if len(sys.argv) >= 4:
    beta = float(sys.argv[3])
if len(sys.argv) >= 6:
    nodesPath = sys.argv[4]
    edgesPath = sys.argv[5]
if len(sys.argv) == 1 or len(sys.argv) == 5 or len(sys.argv) > 6:
    print("Usage: spark-submit random_teleport_hits.py topic_label [num_iter] [beta] [nodes_csv] [edges_csv]")

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)
edgesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(edgesPath)

nodesDF = nodesDF.withColumnRenamed("id:ID", "id")
edgesDF = edgesDF.withColumnRenamed("src:START_ID", "src_id").withColumnRenamed("dst:END_ID", "dst_id")

nodes = nodesDF.rdd
num_nodes = nodes.count()
edges = edgesDF.select("src_id", "dst_id").rdd.map(lambda edge: (edge[0], edge[1]))
edgesT = edges.map(lambda edge: (edge[1], edge[0]))

# Give label 1 to topic-specific nodes, 0 to the other nodes
nodesDF = nodesDF.withColumn("topic_specific", when(nodesDF["labels:LABEL"] == topic_label, 1).otherwise(0))

num_topic_nodes = nodesDF.where(col("topic_specific") == 1).rdd.count()

#edges.saveAsTextFile("../outputs/edges.txt")
#edgesT.saveAsTextFile("../outputs/edgesT.txt")

auths, hubs, edges, edgesT, nodes_label = initialize_topic_specific_hits(nodesDF, edgesDF, num_topic_nodes)

print("Nodes:")
print(nodes.take(10))
print("Edges: ")
print(edges.take(10))

for i in range(num_iter):
    print("Iteration ", str(i+1))

    #Hub: for each node a, accumulate authority scores from all links of the form (a,b). Divide by each in-degree.
    # For topic-specific hubs, sum contributions from stochastic edges.
    hubs = edgesT.join(auths).map(lambda x: (x[1][0][0], x[1][1]/x[1][0][1])) \
    .reduceByKey(lambda x, y: x + y).join(nodes_label) \
    .mapValues(lambda node: (beta*node[0] if node[1] == 0 else beta*node[0] + ((1-beta)/(2*num_topic_nodes))))
    
    # Authority: for each node b, accumulate hub scores from all links of the form (a,b). Divide by each out-degree.
    # For topic-specific authorities, sum contributions from stochastic edges.
    auths = edges.join(hubs).map(lambda x: (x[1][0][0], x[1][1]/x[1][0][1])) \
    .reduceByKey(lambda x, y: x + y).join(nodes_label) \
    .mapValues(lambda node: (beta*node[0] if node[1] == 0 else beta*node[0] + ((1-beta)/(2*num_topic_nodes))))
    
    # Normalize scores
    hubs = normalize_rdd(hubs)
    auths = normalize_rdd(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/" + topic_label + "_specific_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/" + topic_label + "_specific_authority_scores.txt")

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

draw_graphs(topic_label + "_specific_HITS", edges_list, nodes_dict, hubs_dict, auths_dict)