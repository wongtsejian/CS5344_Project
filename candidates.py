from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, split, explode, udf
from pyspark.sql.types import BooleanType

# initialise spark session
spark = SparkSession.builder.appName('candidates').getOrCreate()

def check_length(string):
    return len(string) > 2

# Register the UDF
length_check_udf = udf(check_length, BooleanType())

# read candidates.csv into spark dataframe and allow multiline data
candidates = spark.read.csv('candidates.csv', header=True,multiLine=True)

# drop columns unless header is NAME, PARTY
candidates = candidates.drop('DISTRICT', 'INCUMBENT', 'OFFICE', 'OFFICE GROUP', 'OFFICE TYPE', 'PARTY GROUP', 'PRIMARY VOTES', 'PRIMARY %', 'RUNOFF VOTES', 'RUNOFF %', 'RUNOFF STATUS', 'TOTAL VOTES', 'TOTAL %', 'Unnamed: 14')

# rename column in lowercase
candidates = candidates.select(lower(candidates.NAME).alias('name'), lower(candidates.PARTY).alias('party'))

# filter for candidates with party affiliation 'bjp' 
bjp = candidates.filter((candidates.party == 'bjp'))

# drop party column
bjp = bjp.drop('party')

# replace punctuations in names with whitespace
bjp_full = bjp.replace({'.':' ', ',':' ', '(':' ', ')':' ', "\'":' ', '\"':' ','\\':' '})

# tokenize names and create a set of unique names
bjp = bjp.withColumn('name', split(bjp.name, '[.,()\'\"\\ ]'))
bjp = bjp.withColumn('name', explode(bjp.name))
bjp = bjp.filter(length_check_udf(bjp['name']))

# filter for candidates with party affiliation 'inc'
inc = candidates.filter((candidates.party == 'inc'))

# drop party column
inc = inc.drop('party')

# replace punctuations in names with whitespace
inc_full = inc.replace({'.':' ', ',':' ', '(':' ', ')':' ', "\'":' ', '\"':' ','\\':' '})

# tokenize names and create a set of unique names
inc = inc.withColumn('name', split(inc.name, '[.,()\'\"\\ ]'))
inc = inc.withColumn('name', explode(inc.name))
inc = inc.filter(length_check_udf(inc['name']))

# find intersection of bjp and inc
intersection = bjp.intersect(inc).sort('name')

# remove intersection from bjp and inc
bjp = bjp.subtract(intersection)
inc = inc.subtract(intersection)

# concatenate bjp and bjp_full
bjp = bjp.union(bjp_full).sort('name')
bjp = bjp.select('name').distinct()

# concatenate inc and inc_full
inc = inc.union(inc_full).sort('name')
inc = inc.select('name').distinct()

# write to csv
bjp.coalesce(1).write.csv('candidates_bjp')
inc.coalesce(1).write.csv('candidates_inc')
intersection.coalesce(1).write.csv('candidates_intersection')
