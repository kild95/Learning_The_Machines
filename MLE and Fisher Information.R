# code - https://rpubs.com/Koba/MLE-Normal
# intuition - https://www.quora.com/What-is-an-intuitive-explanation-of-Fisher-information

normalF <- function(x, parvec) {
  # Log of likelihood of a normal distribution
  # parvec[1] - mean
  # parvec[2] - standard deviation
  # x - set of observations. Should be initialized before MLE
  LL = sum ( -0.5* log(parvec[2]) - 0.5*(x - parvec[1])^2/parvec[2] )
}

x_mat <- matrix(0,100,50)
for(i in 1:50){
  x_mat[,i] <- rnorm(100, 5, 25)
}

MLEs <- matrix(0,10,1)
for(i in 1:length(MLEs[,1])){
  x <- x_mat[,i]
  MLE = optim(x=x, c(0.1,0.1), # initial values for mu and sigma
              fn = normalF, # function to maximize
              method = "L-BFGS-B", # this method lets set lower bounds
              lower = 0.00001, # lower limit for parameters
              control = list(fnscale = -1), # maximize the function
              hessian = T # calculate Hessian matricce because we will need for confidence intervals
  )
  MLEs[i,1] = MLE$par[1]
}
mean(MLEs)

par(mfrow=c(1,2))
plot(x_mat[,1],dnorm(x_mat[,1],MLEs[1],25))
x = x_mat[,1]
for(i in 1:length(MLEs)){
  curve(dnorm(x,MLEs[i],25), add=T)
}




