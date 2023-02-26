#To run on EMR successfully + output results:

#aws s3 sp c3://shanky008/sanket.csv ./
#spark-submit --executor-memory 1g myfile_aws.py 



from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
conf = SparkConf()
sc = SparkContext(conf = conf)

sqlContext = SQLContext(sc)

#### loading data 
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('s3n://shanky008/sanket.csv')
df.show(5)
df.printSchema()

from pyspark.sql.functions import col

df.groupBy("category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()
    
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover,CountVectorizer  
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline

#for tokenization using regular expression
regexTokenizer = RegexTokenizer(inputCol="article", outputCol="words", pattern="\\W")

# Stop words remover
add_stopwords = ["http","https","amp","rt"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filterwords").setStopWords(add_stopwords)

# bag of words count
countVectors = CountVectorizer(inputCol="filterwords", outputCol="features", vocabSize=10000, minDF=5)

# TF-IDF 
hashingTF = HashingTF(inputCol="filterwords", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms

#giving labels for each category
labels = StringIndexer(inputCol = "category", outputCol = "label")

# making a pipeline using count vectorizer 
pipeline1 = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, labels])

# making a pipeline using TF-IDF
pipeline2 = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, labels])

# Fit the pipeline1 to our data.
pipelineFit = pipeline1.fit(df)
dataset1 = pipelineFit.transform(df)
dataset1.show(5)

# spliting of our dataset
(trainingData1, testData1) = dataset1.randomSplit([0.8, 0.2], seed = 88)
print("Training Dataset Count: " + str(trainingData1.count()))
print("Test Dataset Count: " + str(testData1.count()))

# Fit the pipeline2 to our data.
pipelineFit = pipeline2.fit(df)
dataset2 = pipelineFit.transform(df)
dataset2.show(5)

# spliting of our dataset
(trainingData2, testData2) = dataset2.randomSplit([0.8, 0.2], seed = 88)
print("Training Dataset Count: " + str(trainingData2.count()))
print("Test Dataset Count: " + str(testData2.count()))

#################################################################################################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

### using Logistic Regression using count vectors features
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData1)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

predictions1 = lrModel.transform(testData1)
result1 = evaluator.evaluate(predictions1)
print ('Logistic Regression model using count vectors features gives accuracy = '+str(result1))


### using Logistic Regression using TF-IDF features
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData2)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

predictions2 = lrModel.transform(testData2)
result2 = evaluator.evaluate(predictions2)
print ('Logistic Regression model using TF-IDF features gives accuracy = '+str(result2))
#####################################################################################################

##naivebayes using count vectors features
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData1)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

predictions = model.transform(testData1)
result3 = evaluator.evaluate(predictions)
print ('naive bayes accuracy using count vectors features is : '+str(result3))


##naivebayes using TF-IDF features
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData2)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

predictions = model.transform(testData2)
result4 = evaluator.evaluate(predictions)
print ('naive bayes accuracy using TF-IDF features is : '+str(result4))
#################################################################################################

## random forest using count vectors features
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

# Train model with Training Data
rfModel = rf.fit(trainingData1)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

predictions = rfModel.transform(testData1)
result5 = evaluator.evaluate(predictions)
print ('random forest accuracy using count vectors features is: '+str(result5))


## random forest using TF-IDF features
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

# Train model with Training Data
rfModel = rf.fit(trainingData2)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

predictions = rfModel.transform(testData2)
result6 = evaluator.evaluate(predictions)
print ('random forest accuracy using TF-IDF features is: '+str(result6))

################################################################################################
## cross validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

cvModel = cv.fit(trainingData1)
predictions = cvModel.transform(testData1)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result7 = evaluator.evaluate(predictions)
print('cross validation using count vectors features with 5 fold is : '+ str(result7))

# Create 6-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=6)

cvModel = cv.fit(trainingData1)
predictions = cvModel.transform(testData1)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result8 = evaluator.evaluate(predictions)
print('cross validation using count vectors features with 6  fold is : '+ str(result8))

# Create 7-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=7)

cvModel = cv.fit(trainingData1)
predictions = cvModel.transform(testData1)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result9 = evaluator.evaluate(predictions)
print('cross validation using count vectors features with 7 fold is : '+ str(result9))

# Create 8-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=8)

cvModel = cv.fit(trainingData1)
predictions = cvModel.transform(testData1)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result10 = evaluator.evaluate(predictions)
print('cross validation using count vectors features with 8 fold is : '+ str(result10))

# Create 9-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=9)

cvModel = cv.fit(trainingData1)
predictions = cvModel.transform(testData1)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result11 = evaluator.evaluate(predictions)
print('cross validation using count vectors features with 9 fold is : '+ str(result11))

# Create 10-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=10)

cvModel = cv.fit(trainingData1)
predictions = cvModel.transform(testData1)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result12 = evaluator.evaluate(predictions)
print('cross validation using count vectors features with 10 fold is : '+ str(result12))

############################################################################################################

