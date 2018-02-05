# Remove all previous stuff.
rm(list=ls())

# Install packages
#install.packages("lubridate")
#install.packages("Hmisc")
#install.packages("classIntervals")

# Load packages.
library(ggplot2)
#library(corrplot)
#library(lubridate)
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

# Create new factor variable by binning the ages to 8 
# buckets of equal lenght.
train_og$AgeBucket <- cut(train_og$Age, 10)
levels(train_og$AgeBucket) # check created levels
train_og$AgeBucket <- relevel(train_og$AgeBucket, 
                              ref = "(17.9,25.7]") # set yongest as base

# Compare distribution of Age and AgeBucket
#hist(train_og$Age)
#barplot(table(train_og$AgeBucket))

# Create a reasoned classes for balance: Negative, BelowAverage
# and AboveAverage
train_og$BalanceBucket <- ifelse(train_og$Balance <= 0, "Negative",
                        ifelse(train_og$Balance <= mean(train_og$Balance),
                               "BelowAverage", "AboveAverage"))

# Change class to logical (binary) in certain features.
train_og$Default <- as.logical(train_og$Default)
train_og$HHInsurance <- as.logical(train_og$HHInsurance)
train_og$CarLoan <- as.logical(train_og$CarLoan)
train_og$CarInsurance <- as.logical(train_og$CarInsurance)

# Create new data frame where irrelevant columns dropped.
# LastContactDay is of no use, since we do not know the year,
# and thus cannot infer the weekday, which would otherwise 
# have been interesting. 
train_mod <- subset(train_og, 
                    select = -c(LastContactDay, 
                                CallStart, 
                                CallEnd))

# Inspect the newly created CallDuration variable.
# As sanity check, see if tail events match the data.
hist(train_mod$CallDuration)
which(train_mod$CallDuration > 1000) # correct

# Miscellaneous checks.
summary(train_mod)
View(train_mod)
str(train_mod)

# Visual inspection.
barplot(table(train_mod$LastContactMonth))
barplot(table(train_mod$CarInsurance))

# Sanity check: does the information logical across variables?
# How many times customers were not previously contacted?
length(which(train_mod$DaysPassed == -1)) # same for this (N=3042)
length(which(train_mod$PrevAttempts == 0)) # and this (N=3042)

# LogReg_1 model for explorative analysis.
LogReg_1 <- glm(CarInsurance ~ 
                AgeBucket + Job + Marital + Education + Default + 
                  BalanceBucket + HHInsurance + CarLoan + 
                  Communication + LastContactMonth + NoOfContacts + 
                  DaysPassed * PrevAttempts + 
                  Outcome + CallDuration + CallHour, 
                data = train_mod, family = binomial())

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
plot(predict(LogReg_1),residuals(LogReg_1))
abline(h=0,lty=2,col="grey")
lines(lowess(predict(LogReg_1),residuals(LogReg_1)),col="black",lwd=2)

# Write to csv
#write.csv(train_og, file = "train_og.csv")

## PLOTTING

# Density plots with semi-transparent fill
AgeDensities_Response <- ggplot(train_og, aes(x=Age, fill=CarInsurance)) + geom_density(alpha=.3)

BalanceDensities_Response <- ggplot(train_og, aes(x=Balance, fill=CarInsurance)) + geom_density(alpha=.3) +
  + xlim(0, 10000) 

AgeDensities_Response
BalanceDensities_Response

sort(train_og$Balance)

?sort
head(train_og$Balance)
