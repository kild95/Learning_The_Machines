t = c(0.10, 0.23, 0.36, 0.49, 0.61, 0.74,  0.87, 1.00)
q = c(0.84, 0.30, 0.69, 0.45, 0.31, 0.09, -0.17, 0.12)

n = length(q) # = length(t)
df = n-2 # sample size - no. of params 

tq_data = data.frame(t,q)

scatter.smooth(t,q)
plot(density(q)) # q is approx normal so lin regression is satisfied on that front at least

# built-in linear regression package
q_lm = lm(q ~ t)
summary(q_lm)
# NB: Use "summary.lm" (without parentheses) to get how the numbers are calculated automatically
# interpretation: https://feliperego.github.io/blog/2015/10/23/Interpreting-Model-Output-In-R 
# useful link: http://r-statistics.co/Linear-Regression.html

tq_data$predicted = predict(q_lm)
tq_data$residuals = residuals(q_lm)

# install.packages("dplyr")
library(dplyr)
tq_data %>% select(q,predicted,residuals) %>% head() # %>% is piping as far as I am concerned

# Residuals: would want to see around zero - need to check graphically for homoscedasticity
par(mfrow=c(2,2))
plot(q_lm)

par(mfrow=c(1,1)) # Return plotting panel to 1 section

# ggplot - # install.packages("ggplot2")
# library(ggplot2)
ggplot(tq_data, aes(x = t, y = q)) +  # Set up canvas with outcome variable on y-axis
  ggtitle("Simple Linear Regression & Residuals") +
  geom_smooth(method = "lm", se = FALSE, color = "lightblue") +  # Plot regression slope
  geom_segment(aes(xend = t, yend = predicted), alpha=.2) +
  # Color And size adjustments made here...
  geom_point(aes(color = abs(residuals), size = abs(residuals))) + # size also mapped
  scale_color_continuous(low = "black", high = "red") +
  guides(color = FALSE, size = FALSE) +
  geom_point(aes(y = predicted), shape = 1) + # y=predicted looks at tq$predicted because ggplot function as a whole is looking at tq_data
  theme_bw()

# Residual standard error - http://r.789695.n4.nabble.com/How-to-calculate-standard-error-of-estimate-S-for-my-non-linear-regression-model-td4712808.html
sqrt(sum((q - predict(q_lm))^2)/df) # = sigma(q_lm)
# note: predict works the same as fitted here I believe

# Standard Error of Estimates
ssx = 0
for(j in 1:n){
  ssx = ssx + ((t[j]-mean(t))^2)
}

# Slope SE i.e. Beta_1 SE
estimate_se = sqrt(((1/((length(q)-2))*sum(resid(q_lm)^2))/ssx)) # = sqrt(sigma(q_lm)^2/ssx)

# Intercept SE i.e Beta_0 SE
sqrt(sigma(q_lm)^2*(1/n+t_mean^2/ssx))


# linear regression from scratch
# see link for derivations: https://www.youtube.com/watch?v=Pc99IUBWC_Q&index=62&list=PLbxFfU5GKZz3eiOEkcl2By5pYO2CJxZK7

# model form : beta_1*t + beta_0 = q

q_sum = sum(q)
q_sum_t_sum = 0
t_sum = sum(t)
t_squared_sum = sum(t^2)

n = length(q) # = length(t)
df = length(q)-2

for(i in 1:length(q)){
  q_sum_t_sum = q_sum_t_sum + t[i]*q[i]
}

# Setting up simultaneous equations and solving them  
C = matrix(data=c(t_squared_sum, t_sum, t_sum, length(q)), nrow=2, ncol=2, byrow=TRUE)
D = matrix(data=c(q_sum_t_sum, q_sum), nrow=2, ncol=1, byrow=TRUE)

beta_1 = solve(C,D)[1]
beta_0 = solve(C,D)[2]

params = c(beta_0, beta_1) 
p = length(params)

# best fit line coordinates
line = vector(mode="double", length = length(q))
for (i in 1:n){
   line[i] = beta_1*t[i]+beta_0   
}

# scatter.smooth(t,q) # fits loess line by default
# keep things simple for now with basic plot
plot(t,q,title("Simple Linear Regression"))
lines(t,line,col='cyan')
abline(h=q_mean,col='blue',lty=2)
# text(locator(), labels = c("", "q_mean")) # can click on lines and label them this way
legend(0.7, 0.8, legend = c("q_mean","fit"), col=c("blue","cyan"),lty=2:1)

# SST = SSR + SSE i.e. Total Sum of Squares = Sum of Squares of Regression + Sum of Squared Errors

# SSR: the explained sum of squares
ssr = vector(mode="double", length = length(q)) 
for (i in 1:n){
  ssr[i] = ((q[i]-(beta_1*t[i]+beta_0))^2)
}
sum_ssr = sum(ssr) # sum of squared regression # = resid(q_lm)

# SSE: Sum of Squared Errors
q_mean = mean(q)
t_mean = mean(t)
sse = vector(mode="double", length = length(q)) 
for (i in 1:n){
  sse[i] = (((beta_1*t[i]+beta_0)-q_mean)^2)
}
sum_sse = sum(sse)

#SST: Total Sum of Squares
sst = sum_ssr + sum_sse

# Standard Error of Estimates

# get sum of squares of the x (t in this case) values
# needed for standard error calcs
ssx = 0 
for(j in 1:n){
  ssx = ssx + ((t[j]-mean(t))^2)
}

sigma_squared = sum_ssr/df

# t and p values: http://blog.minitab.com/blog/statistics-and-quality-data-analysis/what-are-t-values-and-p-values-in-statistics
# beta0_SE & beta0_tvalue
t_mean_squared = t_mean^2
beta0_Var = sigma_squared*(1/n + t_mean_squared/ssx)
beta0_SE = sqrt(beta0_Var)
# t is the difference between the estimate and the hypothesized value in units of standard error
beta0_tvalue = beta_0/beta0_SE 
# t follows a normal distn
# Most of the time, you'd expect to get t-values close to 0, because...
# if you randomly select representative samples from a population, the mean of most of those random samples from the population should be close to the overall population mean, making their differences (and thus the calculated t-values) close to 0.
# Pr(>|t|) or p-value is the probability that you get a t-value as high or higher than the observed value when the Null Hypothesis (the β coefficient is equal to zero or that there is no relationship) is true.
# It is absolutely important for the model to be statistically significant before we can go ahead and use it to predict (or estimate) the dependent variable, otherwise, the confidence in predicted values from that model reduces and may be construed as an event of chance.
beta0_pvalue =  2*pt(-abs(beta0_tvalue), df=n-2) # pt is two sided. beta0_tvalue could be negative or positive side so we have to allow for two tails
# prob(t < -x) = prob(t > x) and pt func uses < than. abs ensures we always take negative of abs to avoid potentially taking negative of an already negative

# beta1_SE & beta1_tvalue
beta1_Var = sigma_squared/ssx
beta1_SE = sqrt(beta1_Var)
beta1_tvalue = beta_1/beta1_SE
beta1_pvalue = 2*pt(-abs(beta1_tvalue), df=n-2)

# Residual standard error - http://r.789695.n4.nabble.com/How-to-calculate-standard-error-of-estimate-S-for-my-non-linear-regression-model-td4712808.html
RSE = sqrt(sum((q - (beta_1*t+beta_0))^2)/df) # = sigma(q_lm) 
# or equivalently sqrt(sum_ssr/df) # = sigma(q_lm)
# RSE is the avg. amount that the response will deviate from the true regression line.
# (RSE / Intercept Coefficient) is the % that the model will be off with it's preds on average:
average_overall_model_error = RSE/beta_0 # lovely concisely named var...


# R-Squared - provides a measure of how well the model is fitting the actual data
# Doesn't care about sample size...is only concerned with provided data
# formula: http://thestatsgeek.com/2013/10/28/r-squared-and-adjusted-r-squared/
r_sq = 1-(sum_ssr/sst)

# Adjusted R-Squared: See link in R-Squared section above, cares about sample size
# Also: https://stats.stackexchange.com/questions/48703/what-is-the-adjusted-r-squared-formula-in-lm-in-r-and-how-should-it-be-interpret
adj_r_sq = 1 - (1-r_sq)*((n-1)/(n-p)) #residuals df = n-p. See summary.lm



########rough#########

sqrt(sum(resid(q_lm)^2)/df)

X = model.matrix(q_lm)
fitted(q_lm)





