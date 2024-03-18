current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)
rm(list=ls())
library(ggplot2)
library(tidyverse)
library(e1071)
library(tree)
library(nnet)
library(randomForest)

# MSE function
mse <- function(num) mean(num^2); 

# Imports the data
data = read.table("adult.data", fileEncoding = "UTF-8", sep = ",")

# Name the columns (features) and target is income
colnames(data) = c('age','workclass','fnlwgt', 'education', 'educationnum', 
'maritalstatus','occupation', 'relationship','race','sex',
'capitalgain','capitalloss','hoursperweek','nativecountry', 'income')



# Convert each column to its respective data type
data$age = as.numeric(data$age)
data$workclass = as.factor(data$workclass)
data$education = as.factor(data$education)
data$maritalstatus = as.factor(data$maritalstatus)
data$occupation = as.factor(data$occupation)
data$relationship = as.factor(data$relationship)
data$race = as.factor(data$race)
data$sex = as.factor(data$sex)
data$capitalgain = as.numeric(data$capitalgain)
data$capitalloss = as.numeric(data$capitalloss)
data$hoursperweek = as.numeric(data$hoursperweek)
data$nativecountry = as.factor(data$nativecountry)

dim(data)
str(data)


# Saves the variable fnlwgt as weights
weights = data[, 'fnlwgt'] / sum(data[, 'fnlwgt'])

# Drop column fnlwgt for train and test 
data = subset(data, select = -c(fnlwgt))


### EDA
# Summary of the data
head(data)
summary(data)
##### plots #####
# Define the factor levels for relationship and marital-status
data$maritalstatus <- factor(data$maritalstatus, levels = unique(data$maritalstatus))
# Define the factor levels for education_level_number
data$relationship <- factor(data$relationship, levels = unique(data$relationship))

# Create ggplot for marital-status
ggplot(data, aes(x = maritalstatus, fill=income, color=income)) +
  geom_bar() +
  labs(title = "Marital status", x = "Marital status", y = "Frequency")

# Create ggplot for relationship
ggplot(data, aes(x = relationship, fill=income, color=income)) +
  geom_bar() +
  labs(title = "Relationship", x = "Relationship", y = "Frequency")


ggplot(data, aes(x=age, fill=income, color=income)) + 
  geom_histogram() + labs(x = 'Age', y= 'Frequency')

ggplot(data, aes(x=hoursperweek, fill=income, color=income)) +
  geom_histogram(position="identity") + labs(x = 'Hours per week', y= 'Frequency')
ggplot(data, aes(x=sex, fill=income, color=income)) + 
  geom_bar() + labs(x = 'Sex', y= 'Frequency')
ggplot(data, aes(x=race, fill=income, color=income)) + geom_bar() + 
  labs(x = 'Race', y= 'Frequency')


ggplot(data, aes(x=income, y=capitalloss)) +
  geom_point() + xlab("Income") + ylab('Capital loss')

ggplot(data, aes(x=income, y=capitalgain)) +
  geom_point() + xlab("Income") + ylab('Capital gain')

#### End of plots ####





# 1 if the person receives more than 50k, 0 otherwise
data[which(data[, 'income'] == ' >50K'), 'income'] = 1
data[which(data[, 'income'] == ' <=50K'), 'income'] = 0
data$income = as.factor(data$income)

# Drop column education-num 
data = subset(data, select = -c(educationnum))

# Outlier?
length(which(data[, 'capitalgain'] == 99999))
### It is not an outlier

#### Train and test
# Division into train and test sets for target and features
test_percentage = 0.2
test_indices = sample(nrow(data), round(test_percentage * nrow(data)), replace = FALSE)

# Test and train data division
train = data[-test_indices, ]
test = data[test_indices, ]

weights_train = weights[-test_indices]


##### Missing values
### Features with missing values
# workclass: categorical
# occupation: categorical
# native-country: categorical

# Number of missing values
vec1 = which(train$workclass == ' ?')
length(vec1)

vec2 = which(train$occupation == ' ?')
length(vec2)

vec3 = which(train$nativecountry == ' ?')
length(vec3)

length(unique(c(vec1, vec2, vec3)))
nrow(train)

dim(train)
str(train)


### Classification models
### Logistic regression
train_logistic = subset(train, select = -c(nativecountry))
test_logistic = subset(test, select = -c(nativecountry))

m1 <- glm(income ~ ., data = train_logistic, family = binomial('logit'))
#summary(m1)
prob <- predict(m1, test_logistic, type = 'response')
pred <- rep(0, length(prob))
pred[prob>=.5] <- 1
# confusion matrix 
tb <- table(pred, test_logistic$income)
tb
(tb[1,1] + tb[2,2])/(sum(tb))

### For model evaluation
CV_k_Log = function(data, k) {
  
  data = data[sample(nrow(data), nrow(data), replace = FALSE),]
  folds = k
  fold_assignments = rep(1:folds, length.out = nrow(data))
  table(fold_assignments)
  fold_mses = c()
  
  for (i in 1:folds) {
    val_indices = which(fold_assignments == i)
    validation = data[val_indices, ]
    train = data[-val_indices, ]
    
    m1 <- glm(income ~ ., data = train, family = binomial('logit'))
    prob <- predict(m1, validation, type = 'response')
    pred <- rep(0, length(prob))
    pred[prob>=.5] <- 1
    real = as.numeric(validation$income) - 1
    num = real - pred
    
    thisMSE = mse(num)
    fold_mses = c(fold_mses,thisMSE)
  }
  cvMSE = mean(fold_mses)
  return(cvMSE)
}
CV_k_Log(train_logistic, 5)




### Scaling of numeric features
train$age = scale(train$age)[,1]
train$capitalgain = scale(train$capitalgain)[,1]
train$capitalloss = scale(train$capitalloss)[,1]
train$hoursperweek = scale(train$hoursperweek)[,1]


### Random forest
### Parameter tuning for Random Forests
CV_k_RF = function(data, k, ntree) {
  
  data = data[sample(nrow(data), nrow(data), replace = FALSE),]
  folds = k
  fold_assignments = rep(1:folds, length.out = nrow(data))
  table(fold_assignments)
  fold_mses = c()
  
  for (i in 1:folds) {
    val_indices = which(fold_assignments == i)
    validation = data[val_indices, ]
    train = data[-val_indices, ]
    weights_k = weights_train[-val_indices]
    rf <- randomForest(income ~ ., data = train, ntree = ntree, weights = weights_k)
    rf.pred <- predict(rf, newdata = validation, type = 'class')
    
    pred = as.numeric(rf.pred) - 1
    real = as.numeric(validation$income) - 1
    num = real - pred
    
    thisMSE = mse(num)
    fold_mses = c(fold_mses,thisMSE)
  }
  cvMSE = mean(fold_mses)
  return(cvMSE)
}


ntrees = c(10, 100, 500, 1000, 2000)
MSE_ntrees = ntrees*0
for (i in 1:length(ntrees)){
  MSE_ntrees[i] = CV_k_RF(train, 5, ntrees[i])
}
MSE_ntrees
ntrees[which.min(MSE_ntrees)]
## Selected hyperparameter ntree
rf <- randomForest(income ~ ., data = train, ntree = ntrees[which.min(MSE_ntrees)], weights = weights_train)
rf.pred <- predict(rf, newdata = test, type = 'class')
pred = as.numeric(rf.pred) - 1
real = as.numeric(test$income) - 1
num = real - pred
1 - mse(num)
# confusion matrix 
tb <- table(rf.pred, test$income)
tb
(tb[1,1] + tb[2,2])/(sum(tb))


#### NN
##### Parameter tuning for NN
### Cross validation
CV_k_NN = function(data, k, size) {
  
  data = data[sample(nrow(data), nrow(data), replace = FALSE),]
  folds = k
  fold_assignments = rep(1:folds, length.out = nrow(data))
  table(fold_assignments)
  fold_mses = c()
  
  for (i in 1:folds) {
    val_indices = which(fold_assignments == i)
    validation = data[val_indices, ]
    train = data[-val_indices, ]
    weights_k = weights_train[-val_indices]
    nn = nnet(income ~ ., data = train, size = size, maxit = 200, weights = weights_k)
    nn_pred = predict(nn, newdata = validation, type = 'raw')
    
    pred = rep(0, length(nn_pred))
    pred[nn_pred>=.5] <- 1
    real = as.numeric(validation$income) - 1
    num = real - pred
    
    thisMSE = mse(num)
    fold_mses = c(fold_mses,thisMSE)
  }
  cvMSE = mean(fold_mses)
  return(cvMSE)
}


sizes = c(1, 2, 3, 4, 5, 6)
MSE_sizes = sizes*0
for (i in 1:length(sizes)){
  MSE_sizes[i] = CV_k_NN(train, 5, sizes[i])
}
MSE_sizes
which.min(MSE_sizes)
1-MSE_sizes
####### Here ends parameter tuning for NN

#### Final model for NN
size = which.min(MSE_sizes)
size
nn = nnet(income ~ ., data = train, size = size, maxit = 200, weights = weights_train)
nn.pred = predict(nn, newdata = test, type = 'raw')

pred = rep(0, length(nn.pred))
pred[nn.pred>=.5] <- 1

# confusion matrix 
tb = table(pred, test$income)
tb
(tb[1,1] + tb[2,2])/(sum(tb))
