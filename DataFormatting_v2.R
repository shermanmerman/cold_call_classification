# Remove all previous stuff.
rm(list=ls())

# Install packages
#install.packages("lubridate")
#install.packages("Hmisc")
#install.packages("classIntervals")
#install.packages("pROC")
#install.packages("caret")
#install.packages('e1071', dependencies=TRUE)

# Load packages.
#library(ggplot2)
#library(reshape2)
#library(corrplot)
#library(lubridate)
#library(forcats)
#library(Hmisc)
#library(classIntervals)

# Set working directory.
#setwd("C:/Users/Henrik/Desktop/TKK/DSfB I/Project/Kaggle_data")

# Read original training data.
train_og <- read.csv("carInsurance_train.csv",
                  header = T,
                  sep = ",")

# View data structure.
head(train_og)
str(train_og)
summary(train_og)
View(train_og)

## FEATURE ENGINEERING

# Merge date and time into single columns for
# previous call start and end, respectively.
MergeDateTimeStart <- paste(train_og$LastContactDay,
                            train_og$LastContactMonth,
                            train_og$CallStart)

MergeDateTimeEnd <- paste(train_og$LastContactDay,
                          train_og$LastContactMonth, 
                          train_og$CallEnd)

# Change aforementioned from factor type to POSIXct
# Note that year is implicitly assumed 2018 for all obs,
# since there is no information on year.
# This is not harmful, as we do not utilize the
# year info in consequent analysis.
DateTimeStart <- as.POSIXct(MergeDateTimeStart, 
                            format = "%d %b %H:%M:%S", 
                            tz = "GMT")

DateTimeEnd <- as.POSIXct(MergeDateTimeEnd, 
                          format = "%d %b %H:%M:%S",
                          tz = "GMT")

# Calculate the last call durations in seconds and
# add this new variable (CallDuration) to frame as integer column.
train_og$CallDuration <- as.integer(DateTimeEnd - DateTimeStart)

# Create a factor variable describing the call hour and change
# type to variable type to factor (require "lubridate").
train_og$CallHour <- hour(DateTimeStart)
train_og$CallHour <- as.factor(train_og$CallHour)
train_og$CallHour <- relevel(train_og$CallHour, 
                             ref = "9") # set 9am as base

# Factorize certain features.
train_og$Default <- as.factor(train_og$Default)
train_og$HHInsurance <- as.factor(train_og$HHInsurance)
train_og$CarLoan <- as.factor(train_og$CarLoan)
train_og$CarInsurance <- as.factor(train_og$CarInsurance)

# Encode NA's as 'unknowns' in Outcome, Education and Communication vars
train_og$Outcome <- fct_explicit_na(train_og$Outcome, "unknown")
train_og$Education <- fct_explicit_na(train_og$Education, "unknown")
train_og$Communication <- fct_explicit_na(train_og$Communication, "unknown")

# And choose the baselines
train_og$Outcome <- relevel(train_og$Outcome, ref = "unknown")
train_og$Education <- relevel(train_og$Education, ref = "unknown")
train_og$Communication <- relevel(train_og$Communication, ref = "unknown")



# Set LastContactMonth to ordered 
train_og$LastContactMonth <- factor(train_og$LastContactMonth, 
                                    ordered = TRUE,
                                    levels = c("jan", "feb", "mar",
                                               "apr", "may", "jun",
                                               "jul", "aug", "sep",
                                               "oct", "nov", "dec"))


# Create new factor variable by binning the ages to 8 
# buckets of equal lenght.
train_og$AgeBucket <- cut(train_og$Age, 10)
levels(train_og$AgeBucket) # check created levels
train_og$AgeBucket <- relevel(train_og$AgeBucket, 
                              ref = "(17.9,25.7]") # set yongest as base

# Compare distribution of Age and AgeBucket
#hist(train_og$Age)
#barplot(table(train_og$AgeBucket))

# Check for outliers
ggplot(train_og, aes(x=CarInsurance, y=Balance)) + geom_boxplot()

# Find the outlier in balance and create new frame without outlier
train_og[train_og$Balance > 90000, ] # find
train_ol_removed <- train_og[-c(1743), ] # remove
train_ol_removed[train_ol_removed$Balance > 90000, ] # check

# Remove N=19 obs where Job is NA
removed_rows <- which(complete.cases(train_ol_removed) == FALSE)
train_ol_removed <- train_ol_removed[-removed_rows, ]

# Create a reasoned classes for balance: Negative, BelowAverage
# and AboveAverage
train_ol_removed$BalanceBucket <- ifelse(train_ol_removed$Balance <= 0, "Negative",
                        ifelse(train_ol_removed$Balance <= mean(train_ol_removed$Balance),
                               "BelowAverage", "AboveAverage"))

train_ol_removed$BalanceBucket <- as.factor(train_ol_removed$BalanceBucket)
#train_og$BalanceBucket <- factor(train_ol_removed$BalanceBucket, ordered = TRUE, 
#                                    levels = c("Negative",
#                                               "BelowAverage", 
#                                               "AboveAverage"))

train_ol_removed$BalanceBucket <- relevel(train_ol_removed$BalanceBucket, 
                                 ref = "BelowAverage") # set as base

#train_og$LastContactMonth <- relevel(train_og$LastContactMonth,
#                                     ref = "jan")

# Inspect the newly created CallDuration variable.
# As sanity check, see if tail events match the data.
hist(train_ol_removed$CallDuration)
which(train_ol_removed$CallDuration > 1000) # correct

# Miscellaneous checks.
summary(train_ol_removed)
View(train_ol_removed)
str(train_ol_removed)

# Visual inspection.
barplot(table(train_ol_removed$LastContactMonth))
barplot(table(train_ol_removed$CarInsurance))
barplot(table(train_ol_removed$CallHour))
barplot(table(train_ol_removed$Job))

# Sanity check: does the information logical across variables?
# How many times customers were not previously contacted?
length(which(train_ol_removed$DaysPassed == -1)) # same for this (N=3027)
length(which(train_ol_removed$PrevAttempts == 0)) # and this (N=3027)

# Create new data frame where irrelevant columns dropped.
train_mod <- subset(train_ol_removed, 
                    select = -c(CallStart, 
                                CallEnd))

View(train_mod)

# LogReg_1 model for explorative analysis.
LogReg_1 <- glm(CarInsurance ~ 
                AgeBucket + Job + Marital + Education + Default + 
                  BalanceBucket + HHInsurance + CarLoan + 
                  Communication + LastContactMonth + NoOfContacts + 
                  DaysPassed*PrevAttempts + 
                  Outcome + CallDuration + CallHour, 
                data = train_mod, family = "binomial")

# LogReg_1 results.
summary(LogReg_1)

# LogReg_1 residuals.
#residuals(LogReg_1, type="deviance") # residuals 

# Plot
#plot(LogReg_1$fitted.values)
#plot(LogReg_1$residuals) # shows peculiar outlier

#hist(LogReg_1$fitted.values)
#hist(LogReg_1$residuals, xlim = c(-100, 100), breaks = 100)

#plot(LogReg_1$fitted.values, 
#     LogReg_1$residuals, 
#     ylim = c(-20, 20)) # what does this mean?

# Freakonometrics testing - DOES NOT WORK!!!
#plot(predict(LogReg_1), residuals(LogReg_1), col = c("blue", "red")[1 + CarInsurance])
#abline(h=0,lty=2,col="grey")

# Working lowess curve
#plot(predict(LogReg_1),residuals(LogReg_1))
#abline(h=0,lty=2,col="grey")
#lines(lowess(predict(LogReg_1),residuals(LogReg_1)),col="black",lwd=2)

# Write to csv
#write.csv(train_og, file = "train_og.csv")

## PLOTTING

# Age density plots with semi-transparent fill
AgeDensities_Response <- ggplot(train_mod, aes(x=Age, fill=CarInsurance)) +
  geom_density(alpha=.3) + theme(legend.position="top")

AgeDensities_Response

# Balance density plots with semi-transparent fill
BalanceDensities_Response <- ggplot(train_mod, aes(x=Balance, fill=CarInsurance)) +
  geom_density(alpha=.3) + theme(legend.position="top")

BalanceDensities_Response

# CallDuration density plots with semi-transparent fill
CallDurationDensities_Response <- ggplot(train_mod, aes(x=CallDuration, fill=CarInsurance)) +
  geom_density(alpha=.3) + theme(legend.position="top")

CallDurationDensities_Response

# Data frame of continuous vars for plotting
train_cont <- subset(train_mod, 
                    select = c(CarInsurance, # include response
                               Age, 
                               Balance, 
                               LastContactDay,
                               NoOfContacts, 
                               DaysPassed,
                               PrevAttempts,
                               CallDuration))
                          
All_Histogram <- ggplot(melt(train_cont), aes(x=value, fill=CarInsurance)) +
  geom_histogram() +
  facet_wrap(~variable, scales = "free")

All_Histogram

# Data frame of factor vars for plotting
train_fact <- subset(train_mod,
                       select = c(CarInsurance,
                                  Job,
                                  Marital,
                                  Education,
                                  Default,
                                  HHInsurance,
                                  CarLoan,
                                  Communication,
                                  LastContactMonth,
                                  Outcome,
                                  CallHour,
                                  AgeBucket,
                                  BalanceBucket))

All_Barplot <- ggplot(train_fact, aes(x=Job, fill=CarInsurance)) +
  geom_bar() + theme(legend.position="top") + coord_flip() # TO THEME: , axis.text.x  = element_text(angle=270)) 

All_Barplot


# Miscellaneous stuff
summary(train_og$Balance)
quantile(train_og$Balance, 0.999)

barplot(sort(table(train_og$Job),decreasing=T))


## ACCURACY

# Using same data without partition
LogReg1_prob <- predict(LogReg_1, train_mod, type = "response")

# Plot the ROC curve and find AUC for the new model
library(pROC)
ROC <- roc(train_mod$CarInsurance, LogReg1_prob)
plot(ROC, col = "red")
auc(ROC)


# Partition data
# Total number of rows in the credit data frame
n <- nrow(train_mod)

# Number of rows for the training set (80% of the dataset)
n_train <- round(0.8 * n) 

# Create a vector of indices which is an 80% random sample
train_indices <- sample(1:n, n_train)

# Subset the credit data frame to training indices only
train_mod_80 <- train_mod[train_indices, ]  

# Exclude the training indices to create the test set
train_mod_20 <- train_mod[-train_indices, ]

# LogReg_train model for explorative analysis.
LogReg_train <- glm(CarInsurance ~ 
                  AgeBucket + Job + Marital + Education + Default + 
                  BalanceBucket + HHInsurance + CarLoan + 
                  Communication + LastContactMonth + NoOfContacts + 
                  DaysPassed*PrevAttempts + 
                  Outcome + CallDuration + CallHour, 
                data = train_mod_80, family = "binomial")

# Predictions for the validation set
LogReg_train_prob <- predict(LogReg_train, 
                             train_mod_20, 
                             type = "response")

ROC_test <- roc(train_mod_20$CarInsurance, LogReg_train_prob)
plot(ROC_test, col = "blue")
auc(ROC_test)


# Calculate the confusion matrix for the test set
library(caret)
library(e1071)
pred <- ifelse(LogReg_train_prob > 0.50, 1, 0)
confusionMatrix(data = pred,
                reference = train_mod_20$CarInsurance)


# Generate predicted classes using the model object
#class_prediction <- predict(object = credit_model,  
#                            newdata = credit_test,   
#                            type = "class")  

# Base LogReg ROC graph (red without partition, blue 80/20 partition)
plot(ROC, col = "red")
plot(ROC_test, col = "blue", add = TRUE)

# Split to X and y for Python
train_mod_X_Python <- subset(train_mod, 
                                select = -c(CarInsurance))

train_mod_y_Python <- subset(train_mod, 
                             select = c(CarInsurance))

# Write train_mod to csv for Python
write.csv(train_mod_X_Python, "train_mod_X_Python.csv")
write.csv(train_mod_y_Python, "train_mod_y_Python.csv")
