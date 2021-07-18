import sys
import math
from random import sample
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import networkx as nx
import matplotlib.pyplot as plt

def initialize_hits(nodesDF, edgesDF, num_nodes):
    out_degreesDF = edgesDF.groupBy("src_id").count().withColumnRenamed("count", "out_degree")
    in_degreesDF = edgesDF.groupBy("dst_id").count().withColumnRenamed("count", "in_degree")
    degreesDF = out_degreesDF.join(in_degreesDF, in_degreesDF.dst_id == out_degreesDF.src_id).select("src_id", "in_degree", "out_degree").withColumnRenamed("src_id", "id")
    # Associate each node with its out-degree and in-degree
    auths = nodesDF.join(degreesDF, degreesDF.id == nodesDF.id).rdd.map(lambda node: (node[0], (1.0/math.sqrt(num_nodes), (node[2], node[3]))))
    hubs = auths
    degrees = degreesDF.rdd.map(lambda x: (x[0], (x[1], x[2])))
    return auths, hubs, degrees

def normalize_rdd(rdd):
    rdd_norm_squared = rdd.map(lambda x: (0, x[1][0]*x[1][0])).reduceByKey(lambda x, y: x + y).collect()[0][1]
    rdd_norm = math.sqrt(rdd_norm_squared)
    return rdd.map(lambda x: (x[0], (x[1][0] / rdd_norm, x[1][1])))

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
edgesDF = edgesDF.withColumnRenamed("src:START_ID", "src_id").withColumnRenamed("dst:END_ID", "dst_id")

nodes = nodesDF.rdd
num_nodes = nodes.count()
edgesDF = edgesDF.select("src_id", "dst_id")
edges = edgesDF.rdd.map(lambda edge: (edge[0], edge[1]))
edgesT = edges.map(lambda edge: (edge[1], edge[0]))

#edges.saveAsTextFile("../outputs/edges.txt")
#edgesT.saveAsTextFile("../outputs/edgesT.txt"

auths, hubs, degrees = initialize_hits(nodesDF, edgesDF, num_nodes)

print("Nodes:")
print(nodes.take(10))
print("Edges: ")
print(edges.take(10))

# hubSum = authSum = num_nodes/(math.sqrt(num_nodes))

for i in range(num_iter):
    print("Iteration ", str(i+1))

    #Hub: for each node a, accumulate authority scores from all links of the form (a,b). Divide by each in-degree.
    hubs = edgesT.join(auths).map(lambda x: (x[1][0], x[1][1][0]/x[1][1][1][0])) \
    .reduceByKey(lambda x, y: x + y) \
    .mapValues(lambda score: (beta*score + ((1-beta)/(2*num_nodes))))

    hubs = hubs.join(degrees)

    # Authority: for each node b, accumulate hub scores from all links of the form (a,b). Divide by each out-degree.
    auths = edges.join(hubs).map(lambda x: (x[1][0], x[1][1][0]/x[1][1][1][1])) \
    .reduceByKey(lambda x, y: x + y) \
    .mapValues(lambda score: beta*score + ((1-beta)/(2*num_nodes)))

    auths = auths.join(degrees)

    # Normalize scores
    hubs = normalize_rdd(hubs)
    auths = normalize_rdd(auths)

hubs = hubs.sortBy(lambda x: x[1], ascending=False)
auths = auths.sortBy(lambda x: x[1], ascending=False)

# For simplicity's sake, scores are saved as a single file (not recommended with a big dataset in a distributed environment)
hubs.coalesce(1, False).saveAsTextFile("../outputs/teleport_hub_scores.txt")
auths.coalesce(1, False).saveAsTextFile("../outputs/teleport_authority_scores.txt")

hubs = hubs.map(lambda x: (x[0], x[1][0]))
auths = auths.map(lambda x: (x[0], x[1][0]))

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

# Graph visualization: hubs
print("Drawing hubs graph...")
figure = plt.gcf()
figure.set_size_inches(120, 120)
G_hubs = nx.DiGraph()
G_hubs.add_edges_from(edges_list)
G_hubs.add_nodes_from(hubs_dict.keys())
G_hubs.add_nodes_from(nodes_dict.keys())

# Color hub nodes with red, other nodes with grey
node_colors_hubs = ['red' if node in hubs_dict else 'grey' for node in G_hubs.nodes()]

# Make node size proportional to hub score if in the top 50, else use a fixed 500 size
node_sizes_hubs = [hubs_dict[node] * 10000 if node in hubs_dict else 500 for node in G_hubs.nodes()]

pos = nx.spring_layout(G_hubs)
nx.draw_networkx_nodes(G_hubs, pos, cmap=plt.get_cmap('jet'), node_color=node_colors_hubs, node_size=node_sizes_hubs)
nx.draw_networkx_labels(G_hubs, pos)
nx.draw_networkx_edges(G_hubs, pos)

figure.savefig("../outputs/graph_teleportHITS_hub.png", format="PNG", dpi=100)
plt.clf()

# Graph visualization: authorities
print("Drawing authorities graph...")
figure = plt.gcf()
figure.set_size_inches(120, 120)
G_auths = nx.DiGraph()
G_auths.add_edges_from(edges_list)
G_auths.add_nodes_from(auths_dict.keys())
G_auths.add_nodes_from(nodes_dict.keys())

# Color authority nodes with blue, other nodes with grey
node_colors_auths = ['blue' if node in auths_dict else 'grey' for node in G_auths.nodes()]

# Make node size proportional to authority score if in the top 50, else use a fixed 500 size
node_sizes_auths = [auths_dict[node] * 10000 if node in auths_dict else 500 for node in G_auths.nodes()]

pos = nx.spring_layout(G_auths)
nx.draw_networkx_nodes(G_auths, pos, cmap=plt.get_cmap('jet'), node_color=node_colors_auths, node_size=node_sizes_auths)
nx.draw_networkx_labels(G_auths, pos)
nx.draw_networkx_edges(G_auths, pos)

figure.savefig("../outputs/graph_teleportHITS_authorities.png", format="PNG", dpi=100)
plt.clf()


# Graph visualization: both hub and authorities
print("Drawing hub and authorities graph...")
figure = plt.gcf()
figure.set_size_inches(120, 120)
G = nx.DiGraph()
G.add_edges_from(edges_list)
G.add_nodes_from(auths_dict.keys())
G.add_nodes_from(hubs_dict.keys())
G.add_nodes_from(nodes_dict.keys())

# Color hub nodes in red, authority nodes in blue, nodes that are both hub and authorities in purple and other nodes in grey
node_colors = ['purple' if node in hubs_dict and node in auths_dict \
    else 'red' if node in hubs_dict \
    else 'blue' if node in auths_dict \
    else 'grey' for node in G.nodes()]

# Make node size proportional to authority score if in the top 50, else use a fixed 500 size
node_sizes = [hubs_dict[node] * 10000 if node in hubs_dict \
    else auths_dict[node] * 10000 if node in auths_dict \
    else 500 for node in G.nodes()]

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)

figure.savefig("../outputs/graph_teleportHITS_HubAndAuthorities.png", format="PNG", dpi=100)

plt.close()