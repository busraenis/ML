library(caret)
library(randomForest)
library(ROSE)
library(e1071)


# Load the dataset 
data <- read.csv("creditcard.csv")

# Explore the data
str(data)
summary(data)

# Convert "Class" variable to a factor if it's not already
data$Class <- as.factor(data$Class)

# Handle missing values (if any)
data <- na.omit(data)

# Scale numeric features
data[, -ncol(data)] <- scale(data[, -ncol(data)])

# Split the data into training and testing sets
set.seed(2)
trainIndex <- createDataPartition(data$Class, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Calculate the number of observations in the minority class
minority_size <- sum(train_data$Class == 1)

# Oversample the minority class using ROSE
rose_data <- ROSE(Class ~ ., data = train_data, seed = 2, N = 2 * minority_size)$data

# Ensure "Class" is a factor in both training and test sets
rose_data$Class <- as.factor(rose_data$Class)
test_data$Class <- as.factor(test_data$Class)

# Ensure the levels are consistent
levels(test_data$Class) <- levels(rose_data$Class)

# Train a machine learning model (for example, using Random Forest)
model <- randomForest(Class ~ ., data = rose_data)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data[, -ncol(test_data)])
test_data$Class <- as.factor(test_data$Class)
levels(test_data$Class) <- levels(predictions)

# Evaluate the model
confusionMatrix(predictions, test_data$Class)
table(predictions, test_data$Class)

# Re-define as factors and adjust levels
rose_data$Class <- as.factor(rose_data$Class)
test_data$Class <- as.factor(test_data$Class)

# If Class levels are still different, correct them
levels(test_data$Class) <- levels(rose_data$Class)

# Retrain the model
model <- randomForest(Class ~ ., data = rose_data)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data[, -ncol(test_data)])

# Evaluate the model's performance
confusionMatrix(predictions, test_data$Class)


# Calculate the number of observations in the minority class
minority_size <- sum(train_data$Class == 1)

# Oversample the minority class using ROSE
rose_data <- ROSE(Class ~ ., data = train_data, seed = 2, N = 2 * minority_size)$data

# Train a Naive Bayes model
nb_model <- naiveBayes(Class ~ ., data = rose_data)

# Make predictions
nb_predictions <- predict(nb_model, newdata = test_data[, -ncol(test_data)])

# Evaluate the Naive Bayes model
confusionMatrix(nb_predictions, test_data$Class)



library(pROC)

# Calculate ROC curve and AUC for the Random Forest model
roc_curve_rf <- roc(test_data$Class, as.numeric(predictions))
plot(roc_curve_rf, col = "blue", main = "ROC Curve - Random Forest Model")

# Calculate ROC curve and AUC for the Naive Bayes model
roc_curve_nb <- roc(test_data$Class, as.numeric(nb_predictions))
plot(roc_curve_nb, col = "red", add = TRUE)
legend("bottomright", legend = c("Random Forest", "Naive Bayes"), col = c("blue", "red"), lwd = 2)
