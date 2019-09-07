# Neural Net architecture: 3 input nodes, 1 hidden layer w/ 3 nodes, 2 output nodes
# Activation: Sigmoid

X <- matrix(data=c(2,3,4,5,6,7,8,9,10), nrow = 3, ncol = 3)
X <- cbind(1,X)

y <- matrix(data=c(10,12,16,20,22,28), nrow = 2, ncol = 3)

# declaring a_l's is redundant here - just doing so everything is clearly set up
a_1 <- matrix(0,nrow = 3)
a_1 <- rbind(1, a_1)
a_2 <- matrix(0,nrow = 2) # no need for bias term here as this is output layer

rand_data_1 <- rnorm(n = (length(a_1)-1)*(length(X[1,])), 0, .1) # need to init theta with random weights
theta_1 <- matrix(data = rand_data_1, nrow = length(a_1)-1, ncol = length(X[1,]))

rand_data_2 <- rnorm(n = (length(a_2))*(length(a_1)), 0, .1) # need to init theta with random weights
theta_2 <- matrix(data = rand_data_2, nrow = length(a_2), ncol = length(a_1))

sigmoid <- function(z){
  1/(1+exp(-z))
}

cost = matrix(0,nrow = 2) # at least in the simple case where I don't fullly vectorize

# Forward Propagation
# run through first trainig row only
z_1 <- theta_1 %*% X[1,]
a_1 <- sigmoid(z_1)
a_1 <- rbind(1, a_1)

z_2 <- theta_2 %*% a_1
a_2 <- sigmoid(z_2)

cost <- (a_2-y[,1])^2 # this is just for first training set

# Backprop
# del_C/del_theta_2 = (del_z_2/del_theta_2) * (del_a_2/del_z_2) * (del_C/del_a_2)
C = (a_2-y[,1]) # this is (del_C/del_a_2)
delC_deltheta_2 = a_1 %*% t(a_2) %*% (1-a_2) %*% t(C) 
t(delC_deltheta_2) # think I want matrix to be 2x4
# alternatively, delC_deltheta_2_test = C %*% t(a_2) %*% (1-a_2) %*% t(a_1) 

delC_deltheta_2_11 = a_1[2]*a_2[1]*(1-a_2[1])*(a_2[1]-y[1])

######
# working out using my code: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# Neural Net architecture: 2 input nodes, 1 hidden layer w/ 2 nodes, 2 output nodes
# Activation: Sigmoid

X <- matrix(data=c(.05, .1), nrow = 1, ncol = 2)
X <- cbind(1,X)

y <- matrix(data=c(.01, .99), nrow = 2, ncol = 1)

# declaring a_l's is redundant here - just doing so everything is clearly set up
a_1 <- matrix(0,nrow = 2)
a_1 <- rbind(1, a_1)
a_2 <- matrix(0,nrow = 2) # no need for bias term here as this is output layer

#rand_data_1 <- rnorm(n = (length(a_1)-1)*(length(X[1,])), 0, .1) # need to init theta with random weights
theta_1 <- matrix(data = c(.35,.35,.15,.25,.2,.3), nrow = length(a_1)-1, ncol = length(X[1,]))

#rand_data_2 <- rnorm(n = (length(a_2))*(length(a_1)), 0, .1) # need to init theta with random weights
theta_2 <- matrix(data = c(.6,.6,.4,.5,.45,.55), nrow = length(a_2), ncol = length(a_1))

sigmoid <- function(z){
  1/(1+exp(-z))
}

cost <- matrix(0,nrow = 2) # at least in the simple case where I don't fullly vectorize

alpha <- 0.1 # learning rate

#### Forward Propagation ----
# run through first trainig row only
z_1 <- theta_1 %*% X[1,]
a_1 <- sigmoid(z_1)
a_1 <- rbind(1, a_1)

z_2 <- theta_2 %*% a_1
a_2 <- sigmoid(z_2)

cost <- (1/2)*(a_2-y[,1])^2 # this is just for first training set
sum(cost)

####  Backprop ----

# del_C/del_theta_2 = (del_z_2/del_theta_2) * (del_a_2/del_z_2) * (del_C/del_a_2)

delC.dela_2 <- (a_2-y[,1]) # del_C/del_a_2
# simply put del_a_2/del_z_2 = a_2(1-a_2)
# delC_deltheta_2 = a_1 %*% t(a_2) %*% (1-a_2) %*% t(C) # this doesn't work out - need to do some simple scalar multiplication as below
# finicky repeating matrices below to make multiplication work in a general sense (I hope)
rep_mat <- matrix(rep((1-a_2),times=3), nrow = 2) # doesn't repeat as expected with nrow=3, so I transpose it below instead
rep_delC.dela_2 <- matrix(rep(delC.dela_2, times=3), nrow = 2)

delC.deltheta_2 <- t((a_1 %*% t(a_2))*t(rep_mat)) * rep_delC.dela_2 # Check element using: delC_deltheta_2_11 = a_1[2]*a_2[1]*(1-a_2[1])*(a_2[1]-y[1]) # note I have a_1[2] because the index for bias is a_1[1] in R instead of a_1[0] convention

### Updating weights for theta_2

# Will not use these again in this backprop though, they will be used starting in next forward pass 
# i.e. continue to use initial weights while in current backprop iteration
theta_2_updated = theta_2 - alpha*delC.deltheta_2



###### Some good stuff below with bias separate to weights ##########

# attempt from https://peterroelants.github.io/posts/neural-network-implementation-part04/
######
# working out using my code: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# Neural Net architecture: 2 input nodes, 1 hidden layer w/ 2 nodes, 2 output nodes
# Activation: Sigmoid

X <- matrix(data=c(.05, .1), nrow = 2, ncol = 1)
# X <- cbind(1,X) bias separate

y <- matrix(data=c(.01, .99), nrow = 2, ncol = 1)

a_1 <- matrix(0,nrow = 2)
# a_1 <- rbind(1, a_1) leaving this out for bias instead
b_1 <- matrix(0,nrow = 2)
a_2 <- matrix(0,nrow = 2) # no need for bias term here as this is output layer

theta_1 <- matrix(data = c(.15,.25,.2,.3), nrow = length(a_1), ncol = length(X[,1]))
b1_weights <- matrix(data=c(.35,.35), nrow = 2)

theta_2 <- matrix(data = c(.4,.5,.45,.55), nrow = length(a_2), ncol = length(a_1))
b2_weights <- matrix(data=c(.6,.6), nrow = 2)

sigmoid <- function(z){
  1/(1+exp(-z))
}

cost <- matrix(0,nrow = 1) # at least in the simple case where I don't fullly vectorize

alpha <- 0.5 # learning rate

#### Forward Propagation ----
z_1 <- theta_1 %*% X[,1] + b1_weights 
a_1 <- sigmoid(z_1)

z_2 <- theta_2 %*% a_1 + b2_weights
a_2 <- sigmoid(z_2)

cost <- (1/2)*(a_2-y[,1])^2 # this is just for first training set
sum(cost)

####  Backprop ----
del.out <- (a_2-y[,1]) # = derivative of cost # keeping as matrix for now
delC.deltheta_2 <- ((del.out)*(a_2*(1-a_2))) %*% t(a_1) # worked out from https://sudeepraja.github.io/Neural/
# check: (a_2[1]-y[1,1])*(a_2[1]*(1-a_2[1]))*a_1[1]
del.b2 <- del.out # he does not use bias weights in https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
theta_2_updated <- theta_2 - alpha*delC.deltheta_2

# trying different order
del.hidden <- t(a_1*(1-a_1))*(t(del.out) %*% theta_2)
del.hidden <- (theta_2 %*% del.out) * (a_1*(1-a_1))
delC.deltheta_1 <- del.hidden %*% t(X)
del.b1 <- del.hidden # he does not use bias weights in https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# working out separately:
delEo1.delouth1 <- (del.out[1]*a_2[1]*(1-a_2[1]))*theta_2[1,1] # del.Eo1/del.outh1
delEo2.delouth1 <- (del.out[2]*a_2[2]*(1-a_2[2]))*theta_2[2,1] # del.Eo2/del.outh1
(del.out[1]*a_2[2]*(1-a_2[2]))*theta_2[1,2] # del.Eo1/del.outh2
(del.out[2]*a_2[1]*(1-a_2[1]))*theta_2[2,2] # del.Eo2/del.outh2

delEtotal.deltheta_1_1 <- (delEo1.delouth1 + delEo2.delouth1)*a_1[1]*(1-a_1[1])*X[1]

