current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)
rm(list=ls())
library('readxl')
library('lmtest')
library('ggplot2')
library('rmarkdown')
library('DescTools')

ID = 202338054
source("XYZprofile.r")
XYZprofile(ID)

df <- read_excel('data.xlsx', sheet = 1)

# Define the type of the variables
df$Age = as.numeric(df$Age)
df$Centre = as.factor(df$Centre)
df$Gender = as.factor(df$Gender)
### For boxplots
ggplot(df, aes(x=as.factor(Age), y=Pass/Total)) + 
  geom_boxplot(fill="slateblue", alpha=0.2) + 
  xlab("Age") + ylab('Passing rate')

########## Logistic regression ##########

# Formula for the model with the independent variables to use
mod.form = "cbind(Pass,NoPass) ~ Gender + Age + Centre"
# Fitting the logistic regression
glm.out = glm(mod.form, family=binomial(logit), data=df)
# Analysis of deviance table      
anova(glm.out, test="Chisq")
# Summary of the results of the model
summary(glm.out)

# Creates a dataframe with the desired values of the variables
d1 <- data.frame(Centre = 'Worcester', Age = 25, Gender = 'Male')
# Makes a prediction for the logistic regression and calculates standard error
preds1 = predict(glm.out, d1 , type="response", se.fit = TRUE)
# Confidence interval for the passing rate
critval <- 1.96 
upr1 <- preds1$fit + (critval * preds1$se.fit)
lwr1 <- preds1$fit - (critval * preds1$se.fit)
fit1 <- preds1$fit
fit1


# Creates a dataframe with the desired values of the variables
d2 <- data.frame(Centre = 'Wood Green', Age = 25, Gender = 'Male')
# Makes a prediction for the logistic regression and calculates standard error
preds2 = predict(glm.out, d2 , type="response", se.fit = TRUE)
# Confidence interval for the passing rate
critval <- 1.96 ## approx 95% CI
upr2 <- preds2$fit + (critval * preds2$se.fit)
lwr2 <- preds2$fit - (critval * preds2$se.fit)
fit2 <- preds2$fit
fit2

########## Confidence interval plot ##########

df2 <- data.frame(
  TestCentre = c("Worcester", "Wood Green"),
  Mean = c(fit1, fit2),
  Lower = c(lwr1, lwr2),
  Upper = c(upr1, upr2))
# Create a ggplot object
p <- ggplot(df2, aes(x = TestCentre, y = Mean, fill = TestCentre)) +
  # Plot the error bars with a more visually attractive appearance
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.9), width = 0.4, color = "black", size = 1.5) +
  # Customize the plot appearance
  theme_minimal() +
  labs(title = "Pass Rate Confidence Intervals",
       x = "Test Centre",
       y = "Pass Rate")
print(p)


########## Wald test ##########

# Retrieves the data for 25-year-old males in Wood Green and Worcester, respectively
passed_woodgreen = df$Pass[df$Age == 25 & df$Gender == 'Male' & df$Centre == 'Wood Green']
passed_worcester = df$Pass[df$Age == 25 & df$Gender == 'Male' & df$Centre == 'Worcester']

Tot_woodgreen = df$Total[df$Age == 25 & df$Gender == 'Male' & df$Centre == 'Wood Green']
Tot_worcester = df$Total[df$Age == 25 & df$Gender == 'Male' & df$Centre == 'Worcester']

# Calculates passing rate
rate_woodgreen = passed_woodgreen/Tot_woodgreen 
rate_worcester = passed_worcester/Tot_worcester
# Calculates mean
X_woodgreen = mean(rate_woodgreen)
X_worcester = mean(rate_worcester)
# Calculates variance
Var_woodgreen = Var(rate_woodgreen)
Var_worcester = Var(rate_worcester)
# Number of observations
n_woodgreen = length(rate_woodgreen)
n_worcester = length(rate_worcester)
# Calculates standard error of the difference of means
se = sqrt(Var_woodgreen/n_woodgreen + Var_worcester/n_worcester)
# Wald test statistic
W = (X_worcester - X_woodgreen)/se 
W
length(rate_woodgreen)