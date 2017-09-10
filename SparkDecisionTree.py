from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

# Spark stuff

conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)


def binary(YN):
	if (YN == 'Y'):
		return 1
	else:
		return 0


def mapEducation(degree):
	if (degree == 'BS'):
		return 1
	elif (degree == 'MS'):
		return 2
	elif (degree == 'PhD'):
		return 3
	else:
		return 0

def createLabeledPoints(fields):
	yearsExperience = int(fields[0])
	employed = binary(fields[1])
	previousEmployers = binary(fields[2])
	educationLevel = mapEducation(fields[3])
	topTier = binary(fields[4])
	interned = binary(fields[5])
	hired = binary(fields[6])

	return LabeledPoint(hired, array([yearsExperience, employed, previousEmployers, educationLevel,
						topTier, interned]))




# Load up the CSV

rawData = sc.textFile("/Users/manish/Documents/UniversityofMissouri/Udemy/DataScience/Python_FrankKane/DataScience/PastHires.csv")
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)



#Split each line into a list based on comma delimiters

csvData = rawData.map(lambda x: x.split(","))



#convert these lists to labeled points

trainingData = csvData.map(createLabeledPoints)



# creating a testcandidate for testing

testCandidates =  [array([10,1,3,1,0,0])]
testData = sc.parallelize(testCandidates)


#trainclassifier for decision tree

model = DecisionTree.trainClassifier(trainingData, numClasses = 2,
									categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
									impurity='gini', maxDepth=5, maxBins=32)


#Get predictions for unknown candidates

predictions = model.predict(testData)
print('Hire Prediction: ')
results = predictions.collect()
for result in results:
	print(result)


#Print the decision tree itself

print('Learned Classification tree Model: ')
print(model.toDebugString())