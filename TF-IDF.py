from pyspark import SparkConf,SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

#Boilerplate stuff
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

#Load documents
rawData = sc.textFile("/Users/manish/Documents/UniversityofMissouri/Udemy/DataScience/Python_FrankKane/DataScience/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))


#Store the document names for later
documentNames = fields.map(lambda x: x[1])

#Now hash the words in each document to their term frequencies
hashingTF = HashingTF(100000)
tf = hashingTF.transform(documents)

#lets compute the TF*IDF

tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

#figure out hash value of "GettysBurg"
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

#Get score of gettysburg word in each article
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])


#we will zip in the document names so we can see which is which

zippedResults = gettysburgRelevance.zip(documentNames)

print("Best Document for gettysburg is: ")
print(zippedResults.max())