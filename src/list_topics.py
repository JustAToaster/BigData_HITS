import sys
import math
from random import sample
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import networkx as nx
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
nodesPath = '../data/nodes_elab.csv'

spark = SparkSession.builder.appName("Python").getOrCreate()
nodesDF = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(nodesPath)

topics = nodesDF.select("labels:LABEL").distinct().rdd.map(lambda x: x[0])

topics.coalesce(1, False).saveAsTextFile("../outputs/node_topics.txt")

sc.stop()