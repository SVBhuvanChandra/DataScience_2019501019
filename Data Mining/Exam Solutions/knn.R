# Importing the data from CSV file.

liver_data <- read.csv("Liver_data.csv")

n <- nrow(liver_data)

for (row in 1:n) {
  price <- liver_data[row, 6]
  if (!is.null(price) & !is.na(price) & length(price) > 0 & price < 5 & price >= 0) {
    liver_data[row, 6] <- 1
  } else if (!is.null(price) & !is.na(price) & length(price) > 0 & price < 10 & price >= 5) {
    liver_data[row, 6] <- 2
  } else if (!is.null(price) & !is.na(price) & length(price) > 0 & price < 15 & price >= 10) {
    liver_data[row, 6] <- 3
  }else if (!is.null(price) & !is.na(price) & length(price) > 0 & price <= 20 & price >= 15) {
    liver_data[row, 6] <- 4
  }
}

liver_data[,6] = as.integer(liver_data[,6])

head(liver_data)

# Generate a random number that is 90% of the total number of rows.
rand <- sample(1:nrow(liver_data), 0.9* nrow(liver_data))

# Creating Normalization function.
norm <- function(x) {(x -min(x)) / (max(x) -min(x))}

# Perfirming Normalization on the below predictor columns [1 to 6]
liver_data_norm <- as.data.frame(lapply(liver_data[, c(1,2,3,4,5,6)], norm))

summary(liver_data_norm)


# Train data
training_data <- liver_data_norm[rand,]

# Test data
testing_data <- liver_data_norm[-rand,]


# Extracting column-7 of Train data;
# Its is the argument of knn
liver_data_target_category <- liver_data[rand, 7]


# Extracting column-7 in Test data;
# Measuring the accuracy by using it.
liver_data_test_category <- liver_data[-rand, 7]

library(class)

# KNN function
anyNA(liver_data_target_category)


perform_knn <- function(n) {
  prc <- knn(training_data, testing_data, cl=liver_data_target_category, k=n)
  # Confusion Matrix
  tabl <- table(prc, liver_data_test_category)
  print(tabl)
  # funtion to find accuracy : correct predictions / total predictions
  accuracy <- function(x) {sum(diag(x)/(sum(rowSums(x)))) * 10}
  accuracy(tabl)
}

perform_knn(1)
perform_knn(2)
perform_knn(3)
