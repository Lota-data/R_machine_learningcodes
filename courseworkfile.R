

# Alcohol QCM Sensor Dataset Data Set 
# datafile location  <http://archive.ics.uci.edu/ml/datasets.php>


setwd("C:/Users/Otiji Lotanna Brian/Desktop/BIG DATA/DATA MINING AND VISUALIZATION/coursework")
df1 = read.csv("QCM3.csv", header = TRUE, sep=";")
df2 = read.csv("QCM6.csv", header = TRUE, sep=";")
df3 = read.csv("QCM7.csv", header = TRUE, sep=";")
df4 = read.csv("QCM10.csv", header = TRUE, sep=";")
df5 = read.csv("QCM12.csv", header = TRUE, sep=";")



df=rbind(df1,df2,df3,df4,df5)  # Combined the 5 similar data files
View(df)
 
plot(df)        
            
typeof(colnames(df))


head(df, 2)


head(df, 10)
str(df) # The feature values are numerical already  

library(dplyr)  # This library will help me and my group-mate detect null values
summary(df)

df[(is.na(df))] #Used to check for null values in all data
sum(is.na(df)) # NO NULL VALUES
dim(df)

# Checked for outliers and found no whiskers
boxplot(df[1:10])


colnames(df)

newdata = cbind(df[11:15])  #new df for the classes
head(newdata)
View(newdata)
newdata$comb <- 0  # A new column(6) called comb
head(newdata[,-6])
# newdata[,2][1]
# colnames(newdata)[2]
# nrow(newdata)

# We used this function to combine the alcohol classes in a single column based on their classification
combfxn <- function(){
  for (i in 1:nrow(newdata)){
    newdata[,6][i] = ifelse(newdata[,1][i] == 1,colnames(newdata)[1], newdata[,6][i])
    newdata[,6][i] = ifelse(newdata[,2][i] == 1,colnames(newdata)[2], newdata[,6][i])
    newdata[,6][i] = ifelse(newdata[,3][i] == 1,colnames(newdata)[3], newdata[,6][i])
    newdata[,6][i] = ifelse(newdata[,4][i] == 1,colnames(newdata)[4], newdata[,6][i])
    newdata[,6][i] = ifelse(newdata[,5][i] == 1,colnames(newdata)[5], newdata[,6][i])
    
    #OR 
    
    #if (newdata[,1][i] == 1){newdata[,6][i] = colnames(newdata)[1]}
    #else if (newdata[,2][i] == 1){newdata[,6][i]= colnames(newdata)[2]}
    #else if (newdata[,3][i] == 1){newdata[,6][i]= colnames(newdata)[3]}
    #else if (newdata[,4][i] == 1){newdata[,6][i]= colnames(newdata)[4]}
    #else {newdata[,6][i]= colnames(newdata)[5]}
  }
  return (newdata$comb)
  # return (View(newdata))
}

combfxn()
newdata$comb <- combfxn()
View(newdata)

#Here We created a new dataframe for the features and combined it with the already created comb column.
feat = cbind(df[1:10]) # features
head(feat, 3)

#Here we combined the features column and the newly combined labels
maindata <- cbind(feat, newdata["comb"])
View(maindata)
colnames(maindata)[11] <- "Alcohol_Type"  #Here We changed the col name
str(maindata) 
colnames(maindata)[1]
#From the structure, "alcohol type" is a character, but this should be a factor because its the label needed for classification
maindata$`Alcohol_Type` = as.factor(maindata$`Alcohol_Type`)
levels(maindata$`Alcohol_Type`)  # Now the levels are present
str(maindata)
write.csv(maindata,"maindata.csv", row.names=TRUE)

#  DATA SPLIT INTO TRAIN AND TEST
set.seed(92203)
data_size<- floor(nrow(maindata)* 0.80)
sampling <- sample(1: nrow(maindata), size = data_size)
sampling


train <- maindata[sampling,]
test <- maindata[-sampling,]

head(train,2)

View(train)
head(test)
str(train)
train$Alcohol_Type
is.data.frame(train)
#train$Alcohol_Type <- sapply(as.character(train$Alcohol_Type), switch, "X1.isobutanol" = 1, "X1.Octanol" = 2, "X1.Propanol" = 3, "X2.Butanol" = 4,"X2.propanol" = 5, USE.NAMES = F)
#train$Alcohol_Type <- as.factor(train$Alcohol_Type)


# We later chose RandomForest because it is best used for labels that are more than 2.
library(randomForest)
library(caret)  # We used it for the confusion matrix 


#We trained the model here
rf = randomForest(train$Alcohol_Type ~ .,data = train, mtry=4, ntree = 1500, importance= TRUE)
# mtry - number of variables sampled at each split
# ntree - number of trees 
# No need to scale(normalize) labels because the values are not far off

rf
plot(rf)
varImpPlot(rf,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf)

# partialPlot(rf, train, X0.799_0.201 , "X1.isobutanol")


# Running Test Data
resolve <- data.frame(test$Alcohol_Type, predict(rf, test[,1:10],type = "response"))
resolve

resolve2 <- predict(rf, test[,1:10],type = "response")
confusionMatrix(resolve2, test$Alcohol_Type)



# We inserted new gas sample concentrations to predict the type of alcohol.
new_observations <- data.frame(X0.799_0.201 = c(-100), X0.799_0.201.1 = c(-40),X0.700_0.300 = c(-30),X0.700_0.300.1 = c(-25), X0.600_0.400 = c(-72),X0.600_0.400.1 = c(-86), X0.501_0.499 = c(-35),X0.501_0.499.1 = c(-65), X0.400_0.600 = c(-32), X0.400_0.600.1 = c(-100))
View(new_observations)

new_observations_pred <- predict(rf, new_observations , type = "response")
new_observations_pred
plot(new_observations_pred)

# Helped us determine the probabilistic values for the labels
new_observations_prob <- predict(rf, new_observations , type = "prob")
new_observations_prob
barplot(new_observations_prob, main = "Probability values for new observation (rf)", xlab = "Alcohol", ylab = "Probability")


# ----------------------------------------------------------
#LDA 
table(maindata$Alcohol_Type)
require(MASS)
r <- lda(formula = Alcohol_Type ~ ., 
         data = maindata, 
         prior = c(25,25,25,25,25)/125)
r

r$prior  # prior probabilities

r$counts 

r$means

r$svd

r$scaling

prop = r$svd^2/sum(r$svd^2)

prop


r2 <- lda(formula = Alcohol_Type ~ ., 
          data = maindata, 
          prior = c(1,1,1,1,1)/5,
          CV = TRUE)
head(r2$class)
head(r2$posterior, 3)
# LDA classification

# lda_train <- sample(1:125, 80)
r3 <- lda(Alcohol_Type ~ ., 
          train, 
          prior = c(1,1,1,1,1)/5)

plda <- predict(object = r3, # predictions
                newdata = test)
head(plda$class) # classification result
barplot(plda$posterior, xlab = "Alcohol", ylab = "Probability")
head(plda$posterior, 3) # posterior prob.
head(plda$x, 3) # LD projections

plda
confusionMatrix(plda$class, test$Alcohol_Type)  "Confusion Matrix for the test result"

New_pred <- predict(object = r3, # predictions
                newdata = new_observations )
New_pred$class
New_pred$posterior
barplot(New_pred$posterior,main = "Probability values for new observation (lda)", xlab = "Alcohol", ylab = "Probability")


#-----------------------------------------------------
#SVM Classification
library (e1071)
plot(maindata[,-11], col= maindata$Alcohol_Type, pch = 19)
svm_mode <- svm(Alcohol_Type ~ ., 
                data = train,
                probability= TRUE,
                type = "C",
                kernel ='linear',
                scale = FALSE)
summary(svm_mode)
svm_mode
test_svm <- predict(svm_mode, test, probability=TRUE)
summary(test_svm)
test_svm
plot(test_svm)

svm_newobservation <- predict(svm_mode, new_observations, probability=TRUE)
attr(svm_newobservation, "probabilities")
barplot(attr(svm_newobservation, "probabilities"), main = "Probability values for new observation (svm)", xlab = "Alcohol", ylab = "Probability")

confusionMatrix(test_svm, test$Alcohol_Type)
