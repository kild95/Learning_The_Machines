# Understanding Bayesian Statistics

# Corresponding Youtube vids: https://www.youtube.com/watch?v=3OJEae7Qb_o&t=31s
# Skeleton code from: http://www.sumsar.net/files/posts/2017-bayesian-tutorial-exercises/modeling_exercise1.html

# i) from http://www.sumsar.net/files/posts/2017-bayesian-tutorial-exercises/modeling_exercise1.html
# Number of random draws from the prior
n_draws <- 10000
n <- 16
observed_data <- 6 #6/16 

prior <- runif(n_draws, 0, 1) # Here you sample n_draws draws from the prior  
hist(prior) # It's always good to eyeball the prior to make sure it looks ok.

# Here you define the generative model
generative_model <- function(parameters) {
  rbinom(1, n, parameters)
}

# Here you simulate data using the parameters from the prior and the 
# generative model
sim_data <- rep(NA, n_draws)
for(i in 1:n_draws) {
  sim_data[i] <- generative_model(prior[i])
}

# Here you filter off all draws that do not match the data.
posterior <- prior[sim_data == observed_data] 

hist(posterior) # Eyeball the posterior
length(posterior) # See that we got enought draws left after the filtering.
# There are no rules here, but you probably want to aim
# for >1000 draws.

# Now you can summarize the posterior, where a common summary is to take the mean
# or the median posterior, and perhaps a 95% quantile interval.
post_rate <- median(posterior)
quantile(posterior, c(0.025, 0.975))

### MLE instead of Bayes###

# MLE using optimise function
yf <- rep(0,length = 10)
ys <- rep(1,length = 6)
y <- c(ys,yf) #y<-c(0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1)
n<-1 # why not length(y) i.e 16? Because we have 16 bernoulli trials which is dbinom with n=1
# formulation for the log likelihood for the binomial 
logL <- function(p) sum(log(dbinom(y, n, p)))
# we can test the function for one value of p
logL(0.7)
#plot logL
p.seq <- seq(0, 0.99, 0.01)
plot(p.seq, sapply(p.seq, logL), type="l")
#optimum:
optimize(logL, lower=0, upper=1, maximum=TRUE)

# Alternative MLE
# binomial distn MLE is simply sum(yi)/sum(ni)
# sum(ni) is the length of the dataset i.e 16 here
mle_manual = 6/16

# ii) 
ans <- sum(posterior > 0.2)/length(posterior)

# iii)
signups <- rbinom(n=length(posterior), size = 100, prob = posterior) # 100 signups simulated n times. Each n simulation has different prob posterior[i] for i in 1:n
hist(signups)

quantile(signups, c(0.025, 0.975))


### Extra Reading
# Frequentist v Bayesian example - https://365datascience.com/bayesian-vs-frequentist-approach/
# http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/