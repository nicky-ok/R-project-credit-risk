# Install necessary packages
install.packages("class")       # Classification package
install.packages("party")       # Decision tree package
install.packages("e1071")       # SVM package
install.packages("caret")       # Cross-validation package

# Load libraries
library(class)
library(party)
library(e1071)
library(caret)
library(ggplot2)
library(mfx)
library(randomForest)
library(caret)

compare_models <- function(data, response_var, n_runs) {
  
  library(pROC)
  
  # Remove the response variable from the dataset
  predictors <- setdiff(names(data), response_var)
  
  results <- data.frame(
    Run = integer(),
    Model_Type = character(),
    Variables = character(),
    AIC = numeric(),
    Accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:n_runs) {
    # Randomly select 7 to 10 predictors
    selected_vars <- sample(predictors, sample(5:7, 1))

    formula <- as.formula(paste(response_var, "~", paste(selected_vars, collapse = " + ")))
    
    # Fit logit model
    logit_model <- glm(formula, family = binomial(link = "logit"), data = data)
    logit_pred <- ifelse(predict(logit_model, type = "response") > 0.5, 1, 0)
    logit_accuracy <- mean(logit_pred == data[[response_var]])
    logit_aic <- AIC(logit_model)
    
    # Add logit results to the results data frame
    results <- rbind(results, data.frame(
      Run = i,
      Model_Type = "Logit",
      Variables = paste(selected_vars, collapse = ", "),
      AIC = logit_aic,
      Accuracy = logit_accuracy
    ))
    
    # Fit probit model
    probit_model <- glm(formula, family = binomial(link = "probit"), data = data)
    probit_pred <- ifelse(predict(probit_model, type = "response") > 0.5, 1, 0)
    probit_accuracy <- mean(probit_pred == data[[response_var]])
    probit_aic <- AIC(probit_model)
    
    # Add probit results to the results data frame
    results <- rbind(results, data.frame(
      Run = i,
      Model_Type = "Probit",
      Variables = paste(selected_vars, collapse = ", "),
      AIC = probit_aic,
      Accuracy = probit_accuracy
    ))
  }
  
  # Return the best results for logit and probit
  best_logit <- results[results$Model_Type == "Logit",][which.min(results[results$Model_Type == "Logit",]$AIC),]
  best_probit <- results[results$Model_Type == "Probit",][which.min(results[results$Model_Type == "Probit",]$AIC),]
  
  list(
    Best_Logit = best_logit,
    Best_Probit = best_probit,
    All_Results = results
  )
}



# Load dataset
credit_dataset_CSV <- read.csv("Credit_Rating_Dataset.csv", header = TRUE, stringsAsFactors = TRUE)

# Initial exploratory analysis
summary(credit_dataset_CSV)

# Remove the ID column as it is not a useful predictor
credit_dataset_CSV$ID <- NULL

# Convert categorical variables to factors
credit_dataset_CSV$EDUCATION <- as.factor(credit_dataset_CSV$EDUCATION)
credit_dataset_CSV$SEX <- as.factor(credit_dataset_CSV$SEX)
credit_dataset_CSV$MARRIAGE <- as.factor(credit_dataset_CSV$MARRIAGE)
credit_dataset_CSV$PAY_0 <- as.factor(credit_dataset_CSV$PAY_0)
credit_dataset_CSV$PAY_2 <- as.factor(credit_dataset_CSV$PAY_2)
credit_dataset_CSV$PAY_3 <- as.factor(credit_dataset_CSV$PAY_3)
credit_dataset_CSV$PAY_4 <- as.factor(credit_dataset_CSV$PAY_4)
credit_dataset_CSV$PAY_5 <- as.factor(credit_dataset_CSV$PAY_5)
credit_dataset_CSV$PAY_6 <- as.factor(credit_dataset_CSV$PAY_6)

# Normalize numeric variables
credit_dataset_CSV$LIMIT_BAL <- scale(credit_dataset_CSV$LIMIT_BAL)
credit_dataset_CSV$BILL_AMT1 <- scale(credit_dataset_CSV$BILL_AMT1)
credit_dataset_CSV$BILL_AMT2 <- scale(credit_dataset_CSV$BILL_AMT2)
credit_dataset_CSV$BILL_AMT3 <- scale(credit_dataset_CSV$BILL_AMT3)
credit_dataset_CSV$BILL_AMT4 <- scale(credit_dataset_CSV$BILL_AMT4)
credit_dataset_CSV$BILL_AMT5 <- scale(credit_dataset_CSV$BILL_AMT5)
credit_dataset_CSV$BILL_AMT6 <- scale(credit_dataset_CSV$BILL_AMT6)



# Check for missing values
sum(is.na(credit_dataset_CSV))    # Output: 0 (if no missing values)


# Run the model comparison function
set.seed(123)
data <- credit_dataset_CSV
response_var <- "default.payment.next.month"

set.seed(123)
credit_dataset_CSV$default.payment.next.month = as.factor(credit_dataset_CSV$default.payment.next.month)

# Getting AIC for picked Predictors
logit_model <- glm(
  default.payment.next.month ~ LIMIT_BAL+PAY_0+AGE+BILL_AMT1+PAY_AMT1+EDUCATION+MARRIAGE,
  data = credit_dataset_CSV,
  family = binomial
)
summary(logit_model)


# Training for the picked predictors
train_Control <- trainControl(method = "cv", number = 5)
formula <- default.payment.next.month ~ PAY_0 + PAY_2 + PAY_3 + EDUCATION + LIMIT_BAL + BILL_AMT1 + AGE
model_fit <- train(
  formula, 
  data = credit_dataset_CSV, 
  method = "knn", 
  trControl = train_Control, 
  tuneLength = 10
)
model_fit

# Getting best predictors
results <- compare_models(data, response_var, n_runs = 10)

# Printing all Results
results

# Extract the best logit model variables
best_logit_vars <- results$Best_Logit$Variables

# Remove any unnecessary spaces or commas
best_logit_vars <- gsub(", ", "+", best_logit_vars)  # Ensure variables are concatenated correctly

# Create the formula dynamically for the 'train' function
logit_formula <- as.formula(paste(response_var, "~", best_logit_vars))

# Printing Logit Formula to be used to train
print(logit_formula)



# List of models to try
models <- c("rf","naive_bayes", "svmLinear", "knn")

# List of trainControl values to try (number of cross-validation folds)
train_folds <- c(3,4,5)

# Create a data frame to store results
model_results <- data.frame(
  Model_Type = character(),
  Train_Folds = integer(),
  Accuracy = numeric(),
  stringsAsFactors = FALSE
)


set.seed(123)
# Loop over models and trainControl values
for (model in models) {
  for (folds in train_folds) {
    # Train control setup
    train_Control <- trainControl(method = "cv", number = folds)
    
    # Train the model using the selected algorithm
    model_fit <- train(
      logit_formula, 
      data = credit_dataset_CSV, 
      method = model, 
      trControl = train_Control, 
      tuneLength = 10
    )
    
    # Extract accuracy of the model
    accuracy <- max(model_fit$results$Accuracy)
    
    # Store the results
    model_results <- rbind(model_results, data.frame(
      Model_Type = model,
      Train_Folds = folds,
      Accuracy = accuracy
    ))
    
    # Print the accuracy after each run
    cat(sprintf("Model: %s | Folds: %d | Accuracy: %.4f\n", model, folds, accuracy))
  }
}

# Find the best model based on accuracy
best_model <- model_results[which.max(model_results$Accuracy),]
cat(sprintf("Best Model: %s with %d folds and Accuracy: %.4f\n", 
            best_model$Model_Type, best_model$Train_Folds, best_model$Accuracy))


# Load ggplot2 (if not already loaded)
library(ggplot2)

# Plot model accuracy
ggplot(model_results, aes(x = Model_Type, y = Accuracy, fill = as.factor(Train_Folds))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Accuracy Comparison Across Models & Folds",
    x = "Model",
    y = "Accuracy",
    fill = "CV Folds"
  ) +
  theme_minimal(base_size = 14) +
  scale_fill_brewer(palette = "Set2")

library(randomForest)

# Train Random Forest again if needed
rf_model <- randomForest(
  default.payment.next.month ~ PAY_0 + PAY_2 + PAY_3 + EDUCATION + LIMIT_BAL + BILL_AMT1 + AGE,
  data = credit_dataset_CSV,
  importance = TRUE
)

# Plot importance
varImpPlot(rf_model, main = "Variable Importance - Random Forest")
