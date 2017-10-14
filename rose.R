#libs
require("ROSE")
require("pROC")
require("rpart")
require("rpart.plot")
require("caret")
require("randomForest")
require("e1071")


#main file
cc <- data.frame(read.csv("data/cc.csv"))
cc <- cc[,c(2:31)]

#check balance
fraud <- nrow(cc[cc$Class == 1,])
notFraud <- nrow(cc) - fraud
paste("fraud: ", fraud, "|| not fraud: ", notFraud)

#make the size smaller for easier use
set.seed(65)
cc_simp <- cc[sample(nrow(cc), 25000), ]
fraud <- nrow(cc_simp[cc_simp$Class == 1,])
notFraud <- nrow(cc_simp) - fraud
paste("fraud: ", fraud, "|| not fraud: ", notFraud)

#split data for testing models
trainLength <- floor(.7*nrow(cc_simp))
testLength <- nrow(cc_simp) - trainLength

train_model <- cc_simp[1:trainLength,]
train_eval <- cc_simp[(trainLength + 1):nrow(cc_simp),]

#use rose synthetic to do some magic
fraud <- nrow(train_model[train_model$Class == 1,])
notFraud <- nrow(train_model) - fraud
paste("fraud: ", fraud, "|| not fraud: ", notFraud)
train_model <- ROSE(Class ~ ., data = train_model, seed = 1)$data


#functions for sd and se
mysd <- function(predict, target) {
  diff_sq <- (predict - mean(target))^2
  return(mean(sqrt(diff_sq)))
}

myse <- function(predict, target) {
  diff_sq <- (predict - target)^2
  return(mean(sqrt(diff_sq)))
}


#Model1 - Multiple Linear Regression - Base Line
mlr1 <- glm(Class~., data = train_model)
BIC(mlr1)
predict_mlr1 <- predict(mlr1, train_eval, type = 'response')
table(train_eval$Class, predict_mlr1 > 0.5)
mysd(predict_mlr1, train_eval$Class)
myse(predict_mlr1, train_eval$Class)
auc_mlr1 <- roc(train_eval$Class, predict_mlr1)
plot(auc_mlr1)


#Model2 - Poisson Model
poisson1 <- glm(Class ~ ., family = "poisson", data = train_model)
BIC(poisson1)
predict_poisson1 <- predict(poisson1, train_eval, type = 'response')
table(train_eval$Class, predict_poisson1 > 0.5)
mysd(predict_poisson1, train_eval$Class)
myse(predict_poisson1, train_eval$Class)
auc_poisson <- roc(train_eval$Class, predict_poisson1)
plot(auc_poisson)


#logit model
logit1 <- glm(Class ~., family = binomial(link='logit'), data = train_model)
BIC(logit1)
predict_logit1 <- predict(logit1, train_eval, type = 'response')
table(train_eval$Class, predict_logit1 > 0.5)
auc_logit1 <- roc(train_eval$Class, predict_logit1)
plot(auc_logit1)


# backward stepwise
stepwise1 <- glm(Class ~ ., data = train_model)
backward <- step(stepwise1, trace = 0)
BIC(backward)
predict_backward <- predict(backward, train_eval, type = 'response')
table(train_eval$Class, predict_backward > 0.5)
mysd(predict_backward, train_eval$Class)
myse(predict_backward, train_eval$Class)
auc_backward <- roc(train_eval$Class, predict_backward)
plot(auc_backward)


#forward stepwise
stepwise2 <- glm(Class ~ 1,data = train_model)
forward <- step(stepwise2, scope = list(lower=formula(stepwise2), upper=formula(stepwise1)), direction = "forward", trace = 0)
BIC(forward)
predict_forward <- predict(forward, train_eval, type = 'response')
table(train_eval$Class, predict_forward > 0.5)
mysd(predict_forward, train_eval$Class)
myse(predict_forward, train_eval$Class)
auc_forward <- roc(train_eval$Class, predict_forward)
plot(auc_forward)


# decision tree
decision <- rpart(Class ~ ., data = train_model, method = "class")
prp(decision)
predict_decision <- predict(decision, train_eval, type = "class")
confusionMatrix(train_eval$Class, predict_decision)


#decision tree random forest (kind of broke with raw data)
rforest <- randomForest(Class ~ ., data = train_model)
predict_rforest <- predict(rforest, train_eval)
table(train_eval$Class, predict_rforest > 0.5)
auc_rforest <- roc(train_eval$Class, predict_rforest)
plot(auc_rforest)


#svm (kind of broken with raw data)
svm <- svm(Class ~ ., data = train_model)
predict_svm <- predict(svm, train_eval)
table(train_eval$Class, predict_svm > 0.5)
auc_svm <- roc(train_eval$Class, predict_svm)
plot(auc_svm)


#tables only
table(train_eval$Class, predict_mlr1 > 0.5)
table(train_eval$Class, predict_poisson1 > 0.5)
table(train_eval$Class, predict_logit1 > 0.5)
table(train_eval$Class, predict_backward > 0.5)
table(train_eval$Class, predict_forward > 0.5)
confusionMatrix(train_eval$Class, predict_decision)
table(train_eval$Class, predict_rforest > 0.5)
table(train_eval$Class, predict_svm > 0.5)


#AUC plots only
plot(auc_mlr1)
plot(auc_poisson)
plot(auc_logit1)
plot(auc_backward)
plot(auc_forward)
prp(decision)
plot(auc_rforest)
plot(auc_svm)

accuracy.meas(train_eval$Class, predict_mlr1)
accuracy.meas(train_eval$Class, predict_poisson1)
accuracy.meas(train_eval$Class, predict_logit1)
accuracy.meas(train_eval$Class, predict_backward)
accuracy.meas(train_eval$Class, predict_forward)
accuracy.meas(train_eval$Class, predict_decision)
accuracy.meas(train_eval$Class, predict_rforest)
accuracy.meas(train_eval$Class, predict_svm)