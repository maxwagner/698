library(ROSE)
data(hacide)

table(hacide.train$cls)
prop.table(table(hacide.train$cls))

library(rpart)
treeimb <- rpart(cls ~ ., data = hacide.train)
pred.treeimb <- predict(treeimb, newdata = hacide.test)
accuracy.meas(hacide.test$cls, pred.treeimb[,2])
roc.curve(hacide.test$cls, pred.treeimb[,2], plotit = F)

#over sample
data_balanced_over <- ovun.sample(cls ~ ., data = hacide.train, method = "over",N = 1960)$data
table(data_balanced_over$cls)

#undersample
data_balanced_under <- ovun.sample(cls ~ ., data = hacide.train, method = "under", N = 40, seed = 1)$data
table(data_balanced_under$cls)

#both
data_balanced_both <- ovun.sample(cls ~ ., data = hacide.train, method = "both", p=0.5, N=1000, seed = 1)$data
table(data_balanced_both$cls)

#rose
data.rose <- ROSE(cls ~ ., data = hacide.train, seed = 1)$data
table(data.rose$cls)

#build decision tree models
tree.rose <- rpart(cls ~ ., data = data.rose)
tree.over <- rpart(cls ~ ., data = data_balanced_over)
tree.under <- rpart(cls ~ ., data = data_balanced_under)
tree.both <- rpart(cls ~ ., data = data_balanced_both)

#make predictions on unseen data
pred.tree.rose <- predict(tree.rose, newdata = hacide.test)
pred.tree.over <- predict(tree.over, newdata = hacide.test)
pred.tree.under <- predict(tree.under, newdata = hacide.test)
pred.tree.both <- predict(tree.both, newdata = hacide.test)

#AUC ROSE
roc.curve(hacide.test$cls, pred.tree.rose[,2])

#AUC Oversampling
roc.curve(hacide.test$cls, pred.tree.over[,2])

#AUC Undersampling
roc.curve(hacide.test$cls, pred.tree.under[,2])

#AUC Both
roc.curve(hacide.test$cls, pred.tree.both[,2])
