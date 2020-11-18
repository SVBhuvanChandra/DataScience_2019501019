

# Importing class library

library(class)

# Loading TEST and TRAIN data
train = read.csv("E:/IntroToML/DataScience_2019501019/Data Mining/Assignment 5/sonar_train.csv")
test = read.csv("E:/IntroToML/DataScience_2019501019/Data Mining/Assignment 5/sonar_test.csv")

#plotting the data
data = test[,1:2]
plot(data,pch=19,xlab=expression(data[1]),ylab=expression(data[2]))

# Fitting the model
fit = kmeans(data, 2)
fit

points(fit$centers, pch=19, col='blue', cex=2)
c(-1,1)
knnfit = knn(fit$centers,data,as.factor(c(-1,1)))
points(data, col=1+1*as.numeric(knnfit),pch=19)

out = test[,61]
1-sum(knnfit==out)/length(out)

# Question 4

k = test[,1:60]
kmeanfit = kmeans(k,2)
points(kmeanfit$centers, pch=19, col="red", cex=2)
knnfit = knn(kmeanfit$centers, k, as.factor(c(-1,1)))
points(k, col(1+1*as.numeric(knnfit), pch = 19))



























