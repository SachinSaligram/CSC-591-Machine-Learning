library("gmum.r")
library("caret")

set.seed(100)

for(percent_unlabelled in c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)) {
  #data <- read.csv("code/dataset/sensor_readings_24.data.csv", sep = ";")
  data <- read.csv("code/dataset/waveform-+noise.data.csv", sep = ";")
  training_indices <- sample(nrow(data), nrow(data)*0.7)
  labeled_training_indices <- sample(training_indices, length(training_indices)*percent_unlabelled)
  
  
  print(paste('% unlabelled', percent_unlabelled))
  
  X_train <- data[labeled_training_indices,-ncol(data)]
  y_train <- data[labeled_training_indices,ncol(data)]
  X_test <- data[-training_indices,-ncol(data)]
  y_test <- data[-training_indices,ncol(data)]
  garbage <- capture.output(model <- SVM(X_train, y_train, core="libsvm", kernel = "rbf", gamma=1e3,  transductive.learning=FALSE))
  prediction <- predict(model,X_test)
  print(paste("SVM Accuracy:", sum(prediction==y_test)/length(y_test)))
  
  
  
  
  X_train <- data[labeled_training_indices,-ncol(data)]
  y_train <- data[labeled_training_indices,ncol(data)]
  X_test <- data[-training_indices,-ncol(data)]
  y_test <- data[-training_indices,ncol(data)]
  garbage <- capture.output(model <- train(X_train, y_train, method = "nnet"))
  prediction <- predict(model, X_test)
  print(paste("ANN Accuracy:", sum(prediction==y_test)/length(y_test)))
  
  
  
  
  

  #train_data <- data[labeled_training_indices,]
  #train_data$class <- factor(train_data$class)
  #X_test <- data[-training_indices,-ncol(data)]
  #y_test <- data[-training_indices,ncol(data)]
  #model <- glm(class ~ .,train_data, family=binomial)
  #prediction <- predict(model,X_test, type='response')
  #print(paste("Logistic Regression Accuracy:", sum(prediction==y_test)/length(y_test)))
  
  
  
  
  
  X_train <- data[training_indices,-ncol(data)]
  y_train <- as.character(data[training_indices,ncol(data)])
  X_test <- data[-training_indices,-ncol(data)]
  y_test <- as.character(data[-training_indices,ncol(data)])
  classes <- unique(y_train)
  acc <- 0
  for(class in classes) {
    store_y_train <- y_train
    y_train[store_y_train==class] <- 1
    y_train[store_y_train!=class] <- -1
    y_train[-labeled_training_indices] <- 0
    y_train <- as.factor(y_train)
    
    store_y_test <- y_test
    y_test[store_y_test==class] <- 1
    y_test[store_y_test!=class] <- -1
    y_test <- as.factor(y_test)
    
    svm_model <- SVM( X_train, y_train, core="libsvm", kernel = "rbf", gamma=1e3,  transductive.learning=TRUE)
    p <- predict(svm_model, X_test)
    acc <- acc + sum(y_test[y_test==1]==p[y_test==1])/sum(y_test==1)
    y_train <- store_y_train
    y_test <- store_y_test
  }
  print(paste("Accuracy:", acc))
}
