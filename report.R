# install.packages("dplyr")
# install.packages("caTools")
# install.packages("pROC")  
# install.packages("devtools")

library(dplyr)
library(caTools)
library(pROC)
library(devtools)

# load & process data
data <- read.csv("https://raw.githubusercontent.com/silpa-benny/ASMLC---qnph29/refs/heads/main/heart_failure.csv")  
data <- na.omit(data)

# standardise features
X <- scale(as.matrix(data %>% select(-fatal_mi)))  
y <- data$fatal_mi  

# define MLP training function
train_mlp <- function(X_train, y_train, n_hidden1, n_hidden2, n_hidden3, lr, epochs) {
  set.seed(123)
  
  # initialise weights and biases
  input_size <- ncol(X_train)
  output_size <- 1
  
  W1 <- matrix(runif(input_size * n_hidden1, -0.5, 0.5), nrow = input_size, ncol = n_hidden1)
  b1 <- rep(0, n_hidden1)
  W2 <- matrix(runif(n_hidden1 * n_hidden2, -0.5, 0.5), nrow = n_hidden1, ncol = n_hidden2)
  b2 <- rep(0, n_hidden2)
  W3 <- matrix(runif(n_hidden2 * n_hidden3, -0.5, 0.5), nrow = n_hidden2, ncol = n_hidden3)
  b3 <- rep(0, n_hidden3)
  W4 <- matrix(runif(n_hidden3 * output_size, -0.5, 0.5), nrow = n_hidden3, ncol = output_size)
  b4 <- 0
  
  # activation function
  sigmoid <- function(x) 1 / (1 + exp(-x))
  sigmoid_derivative <- function(x) x * (1 - x)
  
  for (epoch in 1:epochs) {
    # forward pass
    hidden_input1 <- X_train %*% W1 + b1
    hidden_output1 <- sigmoid(hidden_input1)
    
    hidden_input2 <- hidden_output1 %*% W2 + b2
    hidden_output2 <- sigmoid(hidden_input2)
    
    hidden_input3 <- hidden_output2 %*% W3 + b3
    hidden_output3 <- sigmoid(hidden_input3)
    
    final_input <- hidden_output3 %*% W4 + b4
    final_output <- sigmoid(final_input)
    
    # calculate loss
    loss <- -mean(y_train * log(final_output) + (1 - y_train) * log(1 - final_output))
    
    # backpropagation
    error <- final_output - y_train
    d_output <- error * sigmoid_derivative(final_output)
    
    hidden_error3 <- d_output %*% t(W4)
    d_hidden3 <- hidden_error3 * sigmoid_derivative(hidden_output3)
    
    hidden_error2 <- d_hidden3 %*% t(W3)
    d_hidden2 <- hidden_error2 * sigmoid_derivative(hidden_output2)
    
    hidden_error1 <- d_hidden2 %*% t(W2)
    d_hidden1 <- hidden_error1 * sigmoid_derivative(hidden_output1)
    
    # update weights
    W4 <- W4 - lr * t(hidden_output3) %*% d_output
    b4 <- b4 - lr * sum(d_output)
    
    W3 <- W3 - lr * t(hidden_output2) %*% d_hidden3
    b3 <- b3 - lr * colSums(d_hidden3)
    
    W2 <- W2 - lr * t(hidden_output1) %*% d_hidden2
    b2 <- b2 - lr * colSums(d_hidden2)
    
    W1 <- W1 - lr * t(X_train) %*% d_hidden1
    b1 <- b1 - lr * colSums(d_hidden1)
  }
  
  return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, W3 = W3, b3 = b3, W4 = W4, b4 = b4))
}

# prediction function
predict_mlp <- function(model, X_test) {
  sigmoid <- function(x) 1 / (1 + exp(-x))
  
  hidden_input1 <- X_test %*% model$W1 + model$b1
  hidden_output1 <- sigmoid(hidden_input1)
  
  hidden_input2 <- hidden_output1 %*% model$W2 + model$b2
  hidden_output2 <- sigmoid(hidden_input2)
  
  hidden_input3 <- hidden_output2 %*% model$W3 + model$b3
  hidden_output3 <- sigmoid(hidden_input3)
  
  final_input <- hidden_output3 %*% model$W4 + model$b4
  final_output <- sigmoid(final_input)
  
  return(final_output)
}

# 5-fold cross-validation function with final precision, AUC & ROC
cross_validate_mlp <- function(X, y, n_folds = 5, n_hidden1 = 5, n_hidden2 = 15, n_hidden3 = 5, lr = 0.008, epochs = 1000) {
  set.seed(123)
  folds <- sample(rep(1:n_folds, length.out = nrow(X)))
  accuracies <- c()
  
  y_pred_prob_all <- c()
  y_test_all <- c()
  
  for (fold in 1:n_folds) {
    train_idx <- which(folds != fold)
    test_idx <- which(folds == fold)
    
    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    y_test <- y[test_idx]
    
    mlp_model <- train_mlp(X_train, y_train, n_hidden1, n_hidden2, n_hidden3, lr, epochs)
    
    y_pred_prob <- predict_mlp(mlp_model, X_test)
    y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
    
    acc <- mean(y_pred == y_test)
    accuracies <- c(accuracies, acc)
    
    cat(sprintf("Fold %d Accuracy: %.4f\n", fold, acc))
    
    # store predictions and true labels for final metrics
    y_pred_prob_all <- c(y_pred_prob_all, y_pred_prob)
    y_test_all <- c(y_test_all, y_test)
  }
  
  # compute and plot final ROC curve & AUC
  roc_curve <- roc(y_test_all, y_pred_prob_all)
  
  dev.new()
  plot(roc_curve, 
       main = "5-Fold Cross-Validation ROC Curve", 
       col = "blue", 
       lwd = 2, 
       xlim = c(0, 1),  
       ylim = c(0, 1))  
  
  auc_value <- auc(roc_curve)
  cat(sprintf("\nFinal 5-Fold Cross-Validation AUC: %.4f\n", auc_value))
  
  # final precision
  conf_matrix <- table(y_test_all, ifelse(y_pred_prob_all > 0.5, 1, 0))
  if (ncol(conf_matrix) == 2) {  # Ensure no indexing error
    precision_final <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
    cat(sprintf("Final 5-Fold Cross-Validation Precision: %.4f\n", precision_final))
  } else {
    cat("Precision calculation skipped due to class imbalance.\n")
  }
  
  mean_acc <- mean(accuracies)
  cat(sprintf("\nFinal 5-Fold Cross-Validation Accuracy: %.4f\n", mean_acc))
  return(mean_acc)
}

# run 5-fold cross-validation
final_acc <- cross_validate_mlp(X, y, n_folds = 5, n_hidden1 = 5, n_hidden2 = 15, n_hidden3 = 5, lr = 0.008, epochs = 1000)
