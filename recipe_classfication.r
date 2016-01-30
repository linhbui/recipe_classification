rm(list=ls())
library(jsonlite)
library(dplyr)
library(ggplot2)
library(tm) # for text-mining
library(SnowballC)
library(rpart)
library(rpart.plot)

train <- fromJSON('train.json', flatten = TRUE)
test <- fromJSON('test.json', flatten = TRUE)

cuisineSummary = as.data.frame(table(train$cuisine))
labels = cuisineSummary[,1]
counts = cuisineSummary[, 2]
pie(counts, labels = labels, main = 'Cuisine Distribution')
# => Lots of Italian and Mexican recipes in training data set

# var ingredients have the corpus of ingredients train
# Have to combine both train & test ingredients to capture all ingredients in both training & test dataset 
ingredientsTotal <- c(Corpus(VectorSource(train$ingredients)), Corpus(VectorSource(test$ingredients)))

# Preprocessing: puts all words to lower cases & stemization
# Convert to lower case: http://stackoverflow.com/questions/13640188/converting-text-to-lowercase
# Setting mc.cores = 1 because of this: 
# http://stackoverflow.com/questions/18287981/tm-map-has-parallelmclapply-error-in-r-3-0-1-on-mac
ingredientsTotal <- tm_map(ingredientsTotal, PlainTextDocument, mc.cores = 1)
ingredientsTotal <- tm_map(ingredientsTotal, stemDocument, mc.cores = 1)
ingredientsTotal

# Create Document Term Matrix: list of all words & all recipes and whether each word appear in each recipe
matrix <- DocumentTermMatrix(ingredientsTotal)

# More preprocessing: remove ingredients that are rarely used to simplify models
removedSparse <- removeSparseTerms(matrix, 1-5/nrow(matrix))

dataTotal <- as.data.frame(as.matrix(removedSparse))

data <- dataTotal[1:nrow(train),] # training data
data$cuisine <- as.factor(train$cuisine)
dataTest <- dataTotal[-(1:nrow(train)),]


# Naive-Bayes: 
library(parallel)
library(MASS)
library(klaR)
data2 <- data
data2[] <- mclapply(data2, factor)
modelNB <- NaiveBayes(cuisine~., data = data2, fL = 1) # LaPlace correction 
predictedNB <- predict(modelNB)

errNB <- 1 - sum(predictedNB$class == data2$cuisine)/length(data2$cuisine)
errNB # 0.2657766

predictTestNB <- predict(modelNB, dataTest)

# SVM: 
library(e1071)
modelSVM <- svm(cuisine ~ ., data = data, kernel = 'linear')
predictedSVM <- predict(modelSVM, data[,1:1688])
errSVM <- 1 - sum(predictedSVM == data$cuisine)/length(data$cuisine)
errSVM # 0.03985015

predictTestSVM <- predict(modelSVM, dataTest)

# SVM rbf kernel
modelSVMrbf <- svm(cuisine ~ ., data = data)
predictedSVMrbf <- predict(modelSVMrbf, data[,1:1688])
errSVMrbf <- 1 - sum(predictedSVMrbf == data$cuisine)/length(data$cuisine)
errSVMrbf # 0.1455473

predictTestSVMrbf <- predict(modelSVMrbf, dataTest)

# SVM polynomial
modelSVMpoly <- svm(cuisine ~ ., data = data, kernel = 'polynomial')
predictedSVMpoly <- predict(modelSVMpoly, data[,1:1688])
errSVMpoly <- 1 - sum(predictedSVMpoly == data$cuisine)/length(data$cuisine)
errSVMpoly # 0.4456429

predictTestSVMpoly <- predict(modelSVMpoly, dataTest)

# Random Forest
library(randomForest)
modelRF <- randomForest(data[,1:1688],data$cuisine)
predictedRF <- predict(modelRF,data[, 1:1688])
errRF <- 1 - sum(predictedRF == data$cuisine)/length(data$cuisine)
errRF # 0.05071152 

predictTestRF <- predict(modelRF, dataTest)

# Decision tree
set.seed(9999)
modelDT <- rpart(cuisine ~ ., data = data, method = 'class', control=rpart.control(cp=0.01))
predictDT <- predict(modelDT, newdata = data[, 1:1688], type = 'class')
errDT <- 1 - sum(predictDT == data$cuisine)/length(data$cuisine)
errDT # 0.5915422 

predictTestDT <- predict(modelDT, newdata = dataTest, type = 'class')
prp(modelDT)
summary(predictDT)

### Create submission files
# build and write the submission file
# Baseline
submissionBase <- data.frame(id = test$id, cuisine = rep('italian', nrow(dataTest)))
write.csv(submissionBase, file = 'base_line.csv', row.names=F, quote=F)
# error rate = 0.80732

# Decision Tree
submissionDT <- data.frame(id = test$id, cuisine = predictTestDT)
write.csv(submissionDT, file = 'decision_tree.csv', row.names=F, quote=F)
# submission error rate = 0.59885

# Naive-Bayes
submissionNB <- data.frame(id = test$id, cuisine = predictTestNB$class)
write.csv(submissionNB, file = 'naive_bayes.csv', row.names=F, quote=F)
# error rate = 0.97938

# SVM linear
submissionSVM <- data.frame(id = test$id, cuisine = predictTestSVM)
write.csv(submissionSVM, file = 'svm.csv', row.names=F, quote=F)
# error rate = 0.26649

# SVM rbf
submissionSVMrbf <- data.frame(id = test$id, cuisine = predictTestSVMrbf)
write.csv(submissionSVMrbf, file = 'svm_rbf.csv', row.names=F, quote=F)
# error rate = 0.26961

# SVM polynomial
submissionSVMpoly <- data.frame(id = test$id, cuisine = predictTestSVMpoly)
write.csv(submissionSVMpoly, file = 'svm_poly.csv', row.names=F, quote=F)

# Random Forest
submissionRF <- data.frame(id = test$id, cuisine = predictTestRF)
write.csv(submissionRF, file = 'random_forest.csv', row.names=F, quote=F)
# error rate = 0.25312
