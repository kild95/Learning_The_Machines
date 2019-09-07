## See https://statquest.org/2018/10/23/statquest-ridge-lasso-and-elastic-net-regression-in-r/

## This code is based on the code Josh Day's example here:
## https://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_Net_-_Examples.html  {link to original website}

library(glmnet)  
# Package to fit ridge/lasso/elastic net models
# glmnet (interesting but complicated): https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#lin

# glmnet solves: min {??0,??} 1/N ??? {i=1:N} wil(yi,??0+??Txi) + ??[(1?????)||??||2norm^2/2+??||??||1norm]
# so ?? = 0: Ridge ; ?? = 1:Lasso

##############################################################
##
## Example 1
## Assuming 4085 useless variables in the model, only 15 that are useful.
## Also, not much data relative to the number of parameters.
## 1,000 samples and 5,000 parameters.
##
############################################################## {description of example}

set.seed(42)  # Set seed for reproducibility

n <- 1000  # Number of observations
p <- 5000  # Number of predictors included in model
real_p <- 15  # Number of true predictors

## Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

## Split data into training and testing datasets.
## 2/3rds of the data will be used for Training and 1/3 of the
## data will be used for Testing. {Explaining train and test split}
train_rows <- sample(1:n, .66*n)
x.train <- x[train_rows, ]
x.test <- x[-train_rows, ]

y.train <- y[train_rows]
y.test <- y[-train_rows]


## Now we will use 10-fold Cross Validation to determine the
## optimal value for lambda for...

################################
##
## alpha = 0 i.e. Ridge Regression
##
################################
alpha0.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
                        alpha=0, family="gaussian") # family = binomial for logistic regression
alpha0.fit.coefs = head(coef(alpha0.fit)) # these values will then be used to predict
# see: https://cran.r-project.org/web/packages/glmnet/glmnet.pdf
# coefs are returned for lambda.1se only by default

alpha0.predicted <- 
  predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)
## s = is the "size" of the penalty that we want to use, and
##     thus, corresponds to lambda. (I believe that glmnet creators
##     decided to use 's' instead of lambda just in case they 
##     eventually coded up a version that let you specify the 
##     individual lambdas, but I'm not sure.)
##
##     In this case, we set 's' to "lambda.1se", which is
##     the value for lambda that results in the simplest model
##     such that the cross validation error is within one 
##     standard error of the minimum.
##     
##     If we wanted to to specify the lambda that results in the
##     model with the minimum cross valdiation error, not a model
##     within one SE of of the minimum, we would 
##     set 's' to "lambda.min".
##
##     Choice of lambda.1se vs lambda.min boils down to this...
##     Statistically speaking, the cross validation error for 
##     lambda.1se is indistinguisable from the cross validation error
##     for lambda.min, since they are within 1 SE of each other. 
##     So we can pick the simpler model without
##     much risk of severely hindering the ability to accurately
##     predict values for 'y' given values for 'x'.
##
##     All that said, lambda.1se only makes the model simpler when
##     alpha != 0, since we need some Lasso regression mixed in
##     to remove variables from the model. However, to keep things
##     consistant when we compare different alphas, it makes sense
##     to use lambda.1se all the time.
##
## newx = is the Testing Dataset

# Adding my own comments to the above: https://stackoverflow.com/questions/45895238/lambda-1se-not-being-in-one-standard-error-of-the-error
# The simplest model corresponds to the largest lambda (within 1 s.e of the min lambda)
# Remember: Larger lambda leads to a simpler model because it fits
# the training set less well, thus generalising better 
# (provided lambda is not too large, as this leads to underfitting)
# When Josh says that "lambda.1se only makes the model simpler when alpha != 0",
# I think he means simpler in terms of potentially removing a variable (and ridge cannot remove vars),
# but I talk about simpler in terms of less 'overfit' to the traning set, irrespective of no.of vars {General comments - VERY USEFUL!}

## Lastly, let's calculate the Mean Squared Error (MSE) for the model
## created for alpha = 0.
## The MSE is the mean of the sum of the squared difference between 
## the predicted 'y' values and the true 'y' values in the 
## Testing dataset... {Basic MSE comments...}
alpha0.mse = mean((y.test - alpha0.predicted)^2)

# diverging to do some exploring...

simple_fit = glmnet(x.train, y.train, alpha=0) # %Dev is a generalised version of R-squared, according to Trevor Hastie @ approx 35 mins: https://www.youtube.com/watch?v=BU2gjoLPfDc
plot(simple_fit)
plot(simple_fit, xvar = "lambda", label = T) # could interpret the plot as evidence that variables that e
# nter the model early are the most predictive and 
# variables that enter the model later are less important {Comments on this (mess of a) plot}
plot(simple_fit, xvar = "dev", label = T)


tpred = predict(simple_fit, x.test)
mte = apply((tpred-y.test)^2, 2, mean)
plot(alpha0.fit) # cv (default = 10 in cv.glmnet) is performed on each lambda (default = 100 in glmnet)
# i.e 10-fold cv models are fit on each lambda
# the red dot is the mean of the 10 cv's
# the bars signify the standard error of the mean across the 10-fold cv for each lambda { My intuition of the plot}
points(log(simple_fit$lambda),mte,col="blue",pch = 4)
legend("topleft", legend = c("10 fold CV", "Test"), pch = c(19,4), col = c("red","blue"))

log(alpha0.fit$lambda.min) # lowest point on graph above
alpha0.fit$cvm


# ...end of exploring

################################
##
## alpha = 1 i.e. Lasso Regression
##
################################
alpha1.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
                        alpha=1, family="gaussian")

alpha1.predicted <- 
  predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)

alpha1.mse = mean((y.test - alpha1.predicted)^2)

################################
##
## alpha = 0.5, a 50/50 mixture of Ridge and Lasso Regression i.e. ElasticNet
##
################################
alpha0.5.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
                          alpha=0.5, family="gaussian")

alpha0.5.predicted <- 
  predict(alpha0.5.fit, s=alpha0.5.fit$lambda.1se, newx=x.test)

alpha0.5.mse = mean((y.test - alpha0.5.predicted)^2)

################################
##
## However, the best thing to do is just try a bunch of different
## values for alpha rather than guess which one will be best.
##
## The following loop uses 10-fold Cross Validation to determine the
## optimal value for lambda for alpha = 0, 0.1, ... , 0.9, 1.0
## using the Training dataset.
##
## NOTE, on my dinky laptop, this takes about 2 minutes to run
##
################################ {General comments on next step}

list.of.fits <- list()
for (i in 0:10) {
  ## Here's what's going on in this loop...
  ## We are testing alpha = i/10. This means we are testing
  ## alpha = 0/10 = 0 on the first iteration, alpha = 1/10 = 0.1 on
  ## the second iteration etc.
  
  ## First, make a variable name that we can use later to refer
  ## to the model optimized for a specific alpha.
  ## For example, when alpha = 0, we will be able to refer to 
  ## that model with the variable name "alpha0". {Explaining this loop}
  fit.name <- paste0("alpha", i/10)
  
  ## Now fit a model (i.e. optimize lambda) and store it in a list that 
  ## uses the variable name we just created as the reference.
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure="mse", alpha=i/10, 
              family="gaussian")
}

## Now we see which alpha (0, 0.1, ... , 0.9, 1) does the best job
## predicting the values in the Testing dataset.
results <- data.frame()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  
  ## Use each model to predict 'y' given the Testing dataset
  predicted <- 
    predict(list.of.fits[[fit.name]], 
            s=list.of.fits[[fit.name]]$lambda.1se, newx=x.test)
  
  mse <- mean((y.test - predicted)^2)
  
  ## Store the results
  temp <- data.frame(alpha=i/10, mse=mse, fit.name=fit.name)
  results <- rbind(results, temp)
}

## View the results
results
# if you compare the values here to, say, alpha0.mse, they will differ slightly
# because before regularisation/optimisation, the initial parameter values are randomly initialised
# just to be clear, glment uses coordinate descent, hence this discrepancy arises. No big deal, just FYI! {quick comment on results - USEFUL!}

##############################################################
##
## Example 2
## 3500 useless variables, 1500 useful (so lots of useful variables)
## 1,000 samples and 5,000 parameters
##
##############################################################

set.seed(42) # Set seed for reproducibility

n <- 1000    # Number of observations
p <- 5000     # Number of predictors included in model
real_p <- 1500  # Number of true predictors

## Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

# Split data into train (2/3) and test (1/3) sets
train_rows <- sample(1:n, .66*n)
x.train <- x[train_rows, ]
x.test <- x[-train_rows, ]

y.train <- y[train_rows]
y.test <- y[-train_rows]

list.of.fits <- list()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure="mse", alpha=i/10, 
              family="gaussian")
}

results <- data.frame()
for (i in 0:10) {
  fit.name <- paste0("alpha", i/10)
  
  predicted <- 
    predict(list.of.fits[[fit.name]], 
            s=list.of.fits[[fit.name]]$lambda.1se, newx=x.test)
  
  mse <- mean((y.test - predicted)^2)
  
  temp <- data.frame(alpha=i/10, mse=mse, fit.name=fit.name)
  results <- rbind(results, temp)
}

results