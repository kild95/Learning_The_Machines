# Comment out a block: Ctrl + C
# Fold a block of code: Alt + L or use "----". Alt + Shift + L reverses folding
# Get code behind a function: F2 or execute the function name e.g. run "cv.glmnet"

# Define some functions
# Function: Plotting
Plot_LinearReg = function(actual, estimate, title){
  plot(actual, pch=19, bty='l',type="b",col="black", main=title, ylim = c(0,max(actual,estimate)))
  lines(estimate, col="blue", pch=17, type="b")
  legend("topleft", legend=c(deparse(substitute(actual)), deparse(substitute(estimate))), pch=c(19,17), col=c("black","blue"), bty="n")
}
# Function: Linear Regression
Linear_Regression = function(target, pred_vars, title){ # pred_vars should be a df of IV's
  linear_regression = lm(target~., pred_vars)
  summary(linear_regression)
  m = length(data[,1])
  pred = matrix(NA,m)
  for(i in 1:m){
    pred[i] = coef(linear_regression)[1] + coef(linear_regression)[2]*pred_vars[i,1] + coef(linear_regression)[3]*pred_vars[i,2] + coef(linear_regression)[4]*pred_vars[i,3] + coef(linear_regression)[5]*pred_vars[i,4]
  }
  # Cost: J = (1/2m)(Sum{i = 1:n}(y[i]-y_hat[i]))
  lm_cost = 1/(2*m)*((target-pred)^2)
  # Plotting: lm predictions
  Plot_LinearReg(target, pred, title)
  text(x=m/2, y=min(target), labels = bquote(bold(Cost == .(lm_cost))))
}
# fitmodel (Shows that you need more cases than predictors) ----
# fitmodel <- function(n, k) {
#   # n: sample size
#   # k: number of predictors
#   # return linear model fit for given sample size and k predictors
#   x <- data.frame(matrix( rnorm(n*k), nrow=n))
#   names(x) <- paste("x", seq(k), sep="")
#   x$y <- rnorm(n)  
#   lm(y~., data=x)
# }
# 
# summary(fitmodel(n=9, k=10))
# summary(fitmodel(n=10, k=10))
# summary(fitmodel(n=11, k=10))
# summary(fitmodel(n=12, k=10))

# Defining data ----
salary = c(30000, 50000, 45000, 120000, 90000, 60000, 200000, 250000, 80000, 100000)
age = c(22,30,23,35,40,45,50,55,28,34)
sex = c(1,0,0,1,1,1,1,0,0,1)
training_years = c(0,3,1,10,9,10,20,20,4,6)
cfa_level = c(0,2,2,3,3,2,2,3,1,3)

preds = data.frame(age,sex,training_years,cfa_level)
target = data.frame(salary) #str(target)
data = data.frame(preds, target)


# Plotting simple pairs data----
pairs(data)

# Mutliple Regression ----
# Option 1 - individually typing IV's ----
reg_individual = lm(salary ~ age+sex+training_years+cfa_level)
summary(reg_individual)
# Option 2 - referencing dataframe of IV's ----
reg_df = lm(salary~.,preds)
summary(reg_df)
coef(reg_df)
salary_pred = matrix(NA,m)
for(i in 1:m){
  salary_pred[i] = coef(reg_df)[1] + coef(reg_df)[2]*age[i] + coef(reg_df)[3]*sex[i] + coef(reg_df)[4]*training_years[i] + coef(reg_df)[5]*cfa_level[i]
}
# Cost: J = (1/2m)(Sum{i = 1:n}(y[i]-y_hat[i]))
lm_cost = 1/2*(length(data[,1])*(salary-salary_pred)^2)
# Plotting: lm predictions
Plot_LinearReg(salary, salary_pred, "Salary v Salary_Pred")
# Option 3 (Preferred) - Using my predefined function ----
Linear_Regression(salary, preds, "Salary v Salary_pred")

  



# Gradient Descent ----
m = length(data[,1]) 
learning_rate = 0.001 #.001
# I could create one beta vector: Beta = [1,1,1,1,1] and call them in regression using Beta[1] etc...
Beta_0 = 1
Beta_1 = 1
Beta_2 = 1
Beta_3 = 1
Beta_4 = 1

common_derivative_portion = matrix(NA,length(data[,1]),1)
salary_hat_during_updates = matrix(NA,1,5)
updating_cost = matrix(NA,150000)

# 150k epochs - could use a tolerance instead perhaps... ----
# I plotted the loss(cost) fn for up to 1,000,000 epochs but not much was learned after 150k epochs
for(j in 1:150000){
  for(i in 1:m){ # I think this is probably slower than simply doing the calculation within thhe del_Beta_i loop each time, but it makes the code neater
    common_derivative_portion[i] = ((Beta_0 + Beta_1*age[i] + Beta_2*sex[i] + Beta_3*training_years[i] + Beta_4*cfa_level[i])-salary[i])
  }
  # del_Beta_0
  Beta_0_derivative_portion = 0
  for(i in 1:m){
    Beta_0_derivative_portion = Beta_0_derivative_portion + common_derivative_portion[i]
  }
  del_Beta_0 = 1/m * Beta_0_derivative_portion
  Beta_0 = Beta_0 - learning_rate*del_Beta_0
  
  # del_Beta_1
  Beta_1_derivative_portion = 0
  for(i in 1:m){
    Beta_1_derivative_portion = Beta_1_derivative_portion + (common_derivative_portion[i]*age[i])
  }
  del_Beta_1 = 1/m * Beta_1_derivative_portion
  Beta_1 = Beta_1 - learning_rate*del_Beta_1
  
  # del_Beta_2
  Beta_2_derivative_portion = 0
  for(i in 1:m){
    Beta_2_derivative_portion = Beta_2_derivative_portion + (common_derivative_portion[i]*sex[i])
  }
  del_Beta_2 = 1/m * Beta_2_derivative_portion
  Beta_2 = Beta_2 - learning_rate*del_Beta_2
  
  # del_Beta_3
  Beta_3_derivative_portion = 0
  for(i in 1:m){
    Beta_3_derivative_portion = Beta_3_derivative_portion + (common_derivative_portion[i]*training_years[i])
  }
  del_Beta_3 = 1/m * Beta_3_derivative_portion
  Beta_3 = Beta_3 - learning_rate*del_Beta_3
  
  # del_Beta_4
  Beta_4_derivative_portion = 0
  for(i in 1:m){
    Beta_4_derivative_portion = Beta_4_derivative_portion + (common_derivative_portion[i]*cfa_level[i])
  }
  del_Beta_4 = 1/m * Beta_4_derivative_portion
  Beta_4 = Beta_4 - learning_rate*del_Beta_4
  
  for(i in 1:m){
    salary_hat_during_updates[i] = Beta_0 + Beta_1*age[i] + Beta_2*sex[i] + Beta_3*training_years[i] + Beta_4*cfa_level[i]
  }
  cost = (1/2*m)*(salary - salary_hat_during_updates)^2
  updating_cost[j] = mean(cost)
  
}

# Calculating estimates and cost
salary_hat = matrix(NA,1,5)
for(i in 1:m){
  salary_hat[i] = Beta_0 + Beta_1*age[i] + Beta_2*sex[i] + Beta_3*training_years[i] + Beta_4*cfa_level[i]
}
cost = (1/2*m)*((salary - salary_hat)^2)
mean_cost = mean(cost)
Beta_100k = c(Beta_0, Beta_1,Beta_2,Beta_3,Beta_4)

# Plotting: y vs y_hat
Plot_LinearReg(salary, salary_hat, "150k Epochs")

# Plotting: Cost Function
plot(updating_cost[50:150000]) # first few values were huge, so truncating them  
# I think this would take on a more quadratic shape if my learning rate was larger, thus causing the loss not to converge smoothly
# Given the small learning rate, convergence is gentle here

# Another attempt at gradient descent (same as above w/ different code) ----
# This is not working right now
m = length(data[,1]) 
learning_rate = 0.001 
Beta = matrix(1, 5) 
del_Beta = matrix(NA, 5)
X = data.frame(1,age,sex,training_years,cfa_level)
y = salary

k=1
for(e in 1:150000){
  for(j in 1:5){
    for(i in 1:m){
      temp = (Beta[k]*preds[i,k]+Beta[(k+1)]*preds[i,(k+1)]+Beta[(k+2)]*preds[i,(k+2)]+Beta[(k+3)]*preds[i,(k+3)]+Beta[(k+4)]*preds[i,(k+4)]-y[i])*X[i,j]
      del_Beta[j] = del_Beta[j] + temp
    }
    Beta[j] = (1/m)*del_Beta[j]
  }
}


# Trying different learning rates ----
m = length(data[,1]) 
learning_rates = c(0.001, 0.005, 0.01, .05, 0.1) # I read somewhere that alpha is often set to .001 and increased in multiples of 3 from there
salary_hat = matrix(0, length(learning_rates), m)
Beta = matrix(0, length(learning_rates), 5)
# I could create one beta vector: Beta = [1,1,1,1,1] and call them in regression using Beta[1] etc...
Beta_0 = 1
Beta_1 = 1
Beta_2 = 1
Beta_3 = 1
Beta_4 = 1


for(l in 1:length(learning_rates)){
  learning_rate = learning_rates[l]
  for(j in 1:50000){
    #learning_rate = learning_rate * 1.5
    for(i in 1:m){ # I think this is probably slower than simply doing the calculation within thhe del_Beta_i loop each time, but it makes the code neater
      common_derivative_portion[i] = ((Beta_0 + Beta_1*age[i] + Beta_2*sex[i] + Beta_3*training_years[i] + Beta_4*cfa_level[i])-salary[i])
    }
    # del_Beta_0
    Beta_0_derivative_portion = 0
    for(i in 1:m){
      Beta_0_derivative_portion = Beta_0_derivative_portion + common_derivative_portion[i]
    }
    del_Beta_0 = 1/m * Beta_0_derivative_portion
    Beta_0 = Beta_0 - learning_rate*del_Beta_0
    
    # del_Beta_1
    Beta_1_derivative_portion = 0
    for(i in 1:m){
      Beta_1_derivative_portion = Beta_1_derivative_portion + (common_derivative_portion[i]*age[i])
    }
    del_Beta_1 = 1/m * Beta_1_derivative_portion
    Beta_1 = Beta_1 - learning_rate*del_Beta_1
    
    # del_Beta_2
    Beta_2_derivative_portion = 0
    for(i in 1:m){
      Beta_2_derivative_portion = Beta_2_derivative_portion + (common_derivative_portion[i]*sex[i])
    }
    del_Beta_2 = 1/m * Beta_2_derivative_portion
    Beta_2 = Beta_2 - learning_rate*del_Beta_2
    
    # del_Beta_3
    Beta_3_derivative_portion = 0
    for(i in 1:m){
      Beta_3_derivative_portion = Beta_3_derivative_portion + (common_derivative_portion[i]*training_years[i])
    }
    del_Beta_3 = 1/m * Beta_3_derivative_portion
    Beta_3 = Beta_3 - learning_rate*del_Beta_3
    
    # del_Beta_4
    Beta_4_derivative_portion = 0
    for(i in 1:m){
      Beta_4_derivative_portion = Beta_4_derivative_portion + (common_derivative_portion[i]*cfa_level[i])
    }
    del_Beta_4 = 1/m * Beta_4_derivative_portion
    Beta_4 = Beta_4 - learning_rate*del_Beta_4
  }
  # Calculating estimates and cost
  for(i in 1:m){
    salary_hat[l,i] = Beta_0 + Beta_1*age[i] + Beta_2*sex[i] + Beta_3*training_years[i] + Beta_4*cfa_level[i]
  }
  Beta[l,] = c(Beta_0, Beta_1,Beta_2,Beta_3,Beta_4)
}
salary_matrix = matrix(rep(salary,5), ncol = 7)
simplified_cost = (salary_matrix - salary_hat)^2
mean_simplified_cost_by_learning_rate = matrix(NA,5)
for(r in 1:length(simplified_cost[,1])){
  mean_simplified_cost_by_learning_rate[r] = mean(simplified_cost[r,])
}
cost_by_learning_rate = (1/(2*m))*(mean_simplified_cost_by_learning_rate)
simplified_mean_cost = mean(mean_cost_by_learning_rate)


#plot(salary_hat,col='blue',ylim = c((min(salary_hat)-(min(salary_hat)*0.25)),(max(salary_hat)+(max(salary_hat)*0.25))))
#points(salary, col='red')

# Ridge Regression (glmnet) ----
library(tidyverse)
library(broom)
library(glmnet)

X = as.matrix(data.frame(age,sex,training_years,cfa_level))
y = salary
lambdas = seq(0,5,.5)

ridge.mod = glmnet(X, y, alpha = 0, lambda = lambdas)
summary(ridge.mod)
predict(ridge.mod, s = 0, exact = T, type = "coefficients")
cv.out <- cv.glmnet(X, y, alpha = 0) # did not specify lambda to see what is returned
best_lambda = cv.out$lambda.min