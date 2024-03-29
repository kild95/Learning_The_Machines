---
title: "Time Series Regression"
output: 
  # bookdown::gitbook: default # https://bookdown.org/yihui/bookdown/
  html_document:
    toc: yes
    keep_md: true
bibliography: TimeSeriesRegression_Bib.bib # Open file from within Rstudio if you wish to view it.
link-citations: yes # bibloigraphy was created using https://zbib.org/
---


# Introduction

## Motivation

This project looks to assess the suitability of an analysis based on the linear regression of a univariate time series $X_t$ onto its one lagged difference $X_{t-1}$. The analysis is first applied to one particular Auto-Regressive(AR) time series and then I apply this analysis to other generalised time series generated using Bootstrapping and Monte Carlo methods. The idea is to define $α$ (through manual assignment or simulation within a loop), and use Linear Regression to make an estimate $\hat{α}$. Then, I calculate the Squared Error(SE) i.e. $(α_i-\hat{α}_i)^2$ and get the Mean Squared Error(MSE) if suitable. I will compare the MSE's across the methods detailed below to decide whether Linear Regression is 
  i. Suitable
  ii. Robust, where robust means that the MSE's are the same across the three methods.





## Important Comments Before Beginning (Spoilers)

Due to the use of bootstrapping and Monte Carlo methods, there is a random element to the generated data in this project. Thus, when plotting the confidence intervals of the MSE's from the three methods only once and running the Kruskal-Wallis test only once to attain one p-value, the comments and conclusions could vary depending on the output each time that I run the script. Therefore, I will do the following:

1) I will break the project into multiple pieces to explain each step to the reader. I will not make any specific conclusions during this portion of the project. Rather, I will describe to the reader, the conclusions that I could make, depending on the plot or the result of the statistical test. 

2) In order to make this project more scientifically viable, and make sincere conclusions, I will then run all of my code together in a loop $Z$ times. This means that I will generate the $n = 1000$ MSE's for each of the three methods, $Z$ times. Let's arbitrarily choose $Z = 300$ for now. Then, for each $z$, I will produce a distinct p-value via the Kruskall-Wallis test. This will allow me to plot a histogram of the p-values to assess my hypothesis. The question is, do I plot all of the confidence intervals on one plot, whereby I would have $3\times300$ CI's?

A less scientific alternative to assure reproducibility would be to use set.seed() to generate the same data each time we run the notebook. This is evidently a hacky alternative so I am not opting to go with it here. 

More detail will be given on the above procedures as we work through the notebook, so bear with me. Of course, this project has not been reviewed by anybody, so some of my assumptions/ponderings in the later stages might be incorrect. One of the parts of this notebook that I am uncomfortable with, is checking that the Linear Regression assumptions are met. Typically, I make use of R's four diagnostic plots as visual checks, and of course, assessing $Z \times 4$ visual plots is not a viable option. I could look into a more rigorous numerical method of checking the assumptions, but for now, I will check them once and hope that you, the reader, will grant me a little bit of leeway. In general, I have tried to be as scientifically correct as possible.


# Modelling

## Fitting One Single Linear Regression Model 
1. Simulate an $AR(1)$ model, $X_t = αX_{t-1} + ϵ_t$. This will be done using `` `arima.sim` ``. $AR(1)$ corresponds to $ARIMA(1,0,0)$.
2. Remove the first time point, to create $X_t$ with 999 data points.
3. Remove the last time point, to create $X_{t-1}$ with 999 data points, also. This is the once lagged dataset that I use for prediction.
4. Plot $X_t$. There is no advantage to plotting both as $X_{t-1}$ is simply once lagged.


```r
x <- arima.sim(list(order=c(1,0,0), ar=.5), n=1000) # simulating AR(1) model 
n <- length(x)

xt <- x[-1]
xt_1 <-	x[-n] # one lagged time series

df = data.frame(xt, xt_1)
df_melt = melt(df)
```

```
## No id variables; using all as measure variables
```

```r
df_melt['Time'] = rep(seq(1,999), 2)
ggplot(df_melt, aes(x=Time, y=value)) + geom_line() +
  facet_wrap(~ variable, scales = 'free_y', ncol = 1) +
  ggtitle("Simulated AR(1): Original and Lagged Time Series")
```

![](Time-Series-Regression_Knit_Git_files/figure-html/creating time series-1.png)<!-- -->

Fit a simple linear regression model. I am viewing $X_{t-1}$ as the explanatory variable to indicate where $X_t$ is at each time point.

I fit $0$ as the intercept because $ARIMA(1,0,0)$ means that $μ=0$.

The summary gives back an estimate for $α$, which I expect to be close to 0.5 if my linear regression model is suitable.


```r
lm1<-lm(xt~0+xt_1)

summary(lm1) 
```

```
## 
## Call:
## lm(formula = xt ~ 0 + xt_1)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.4591 -0.6480 -0.0135  0.7047  3.4815 
## 
## Coefficients:
##      Estimate Std. Error t value Pr(>|t|)    
## xt_1  0.48654    0.02767   17.58   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.9775 on 998 degrees of freedom
## Multiple R-squared:  0.2365,	Adjusted R-squared:  0.2357 
## F-statistic: 309.1 on 1 and 998 DF,  p-value: < 2.2e-16
```

## Assumptions

Check that the usual Linear Regression assumptions are met using residual diagnostic plots.
* **Linearity**
* **Homoscedasticity/Homogeneity of Variance**
  - We want the distribution of residuals to exhibit random noise, as this would indicate that the line fits the data equally well everywhere, hence linear regression is suitable. See a *Residuals vs Fitted* plot to assess this.
* **Normality** 
  i. For confidence intervals around a parameter to be accurate, the paramater must come from a normal distribution. 
  ii. For significance tests of models to be accurate, the sampling distribution of the thing you’re testing must be normal. 
  iii. To get the best estimates of parameters (i.e. betas in a regression equation), the residuals in the population must be normally distributed.
* **Independence/Autocorrelation** 
    - I use Durbin Watson. Instead, I could look for a lack of pattern in the time series plot (`` `plot.ts` ``) I believe, but not entirely sure on whether that would be the correct plot to assess.
* No influential outliers (Not an absolute requirement)

[Useful Link on Linear Regression Assumptions](https://ademos.people.uic.edu/Chapter12.html)

2. Test for Autocorrelation
* Durbin Watson Test
  * $H_0$: There is no autocorrelation present
  * $H_A$: There is autocorrelation present

As is usual with hypothesis testing, if the p_value > 0.05, I cannot reject $H_0$.

```r
par(mfrow=c(2,2))
plot(lm1)
```

![](Time-Series-Regression_Knit_Git_files/figure-html/assumptions-1.png)<!-- -->

```r
durbinWatsonTest(lm1)
```

```
##  lag Autocorrelation D-W Statistic p-value
##    1      0.02263305      1.954556   0.522
##  Alternative hypothesis: rho != 0
```


**Conclusion**: p-value > 0.05, therefore, Linear Regression assumptions are satisfied and it is suitable for use. As previously discussed, I am extending this conclusion to say that Linear Regression is suitable for AR(1) models so that, later in the project, I can freely use Linear Regression on muliple generated AR(1) models and bypass the need to check the assumptions each time.

## Bootstrapping
I applied bootstrapping to generate 1000 numbers (with resampling because..well..bootstrapping) in the interval [1,999]. I then used these numbers to refer to indices of my time series $X_t$ and $X_{t-1}$ in order to create two new times series. Again, just to reiterate, $X_{t-1}$ is still simply the once lagged version of $X_t$. During each iteration $i$, the current $X_{t_i}$ was regressed on $X_{t_i-1}$. Bootstrapping was necessary to decide if Linear Regression was robust across different AR(1) time series i.e. could it be used on time series other than my intial simulated series.

I calculate the Squared Error from each iteration $i$ by taking the square of the difference between the estimated coefficient $\hat{α}_i$ from the linear model and the $α$ chosen at the start. I store the Squared Error from each iteration in `` `se_boot` `` and I calculate the Mean Squared Error `` `mse_boot` `` after the loop has finished.

I then plot the diagnostics of the last model, just as a pseudo-sanity check. I use the term pseudo here, because just looking at the diagnostic plots of the last Linear Regression model doesn't tell me anything about the previous `` `b` ``-1 models. I don't see the need to go any further into this for now though.


```r
b <- 1000
coef_boot <- matrix(NA, nrow=b, ncol=2)
mse_boot <- matrix(NA, nrow=b, ncol=1)
for(i in 1:b){ # Perform this b times, to create b different samples and lm models
  is <- sample(c(1:n-1), size<-n, replace<-TRUE) # bootstrapping to create a sample of size n
  out <- lm(xt[is]~ 0 + xt_1[is]) # fit lm model on bootstrapped sample (by referencing bootstrapped indices in original sample)
  summary(out)
  coef_boot[i,] <- coef(out) # fill row i with coefficients of the lm model
  mse_boot[i,] <- ((coef(lm1)-coef(out))^2) # return the squared error
}

par(mfrow=c(2,2))
plot(out) # just checking the assumptions on the last particular iteration out of interest
```

![](Time-Series-Regression_Knit_Git_files/figure-html/bootstrapping-1.png)<!-- -->

## Monte Carlo (MC)

I apply Monte Carlo Simulation in two separate instances. 

### Monte Carlo Simulation: 1st method - Using arima.sim

The first MC method uses the `` `arima.sim()` `` function in R to generate an ARIMA time series i.e. $Xt = μ + α(X_{t-1} - μ) + ϵ_t$ with $μ=0$ and $α$ chosen from a Uniform Distribution in the interval (-1,1). The noise terms $ϵ_t$  are generated by a random Normal Distribution centred around 0 ($\mathcal{N}(\mu, \sigma^{2})$). Using this method, I generate `` `M` `` time series which are totally unrelated to my previously series. As with bootstrapping, during each iteration $m$, the current $X_{t_{m_c}}$ was regressed on $X_{t_{m_{c}-1}}$. (Aside: $c$ here is not an iterable, I just use it to tie in with $m$onte $c$arlo. This may be confusing for readers, so I might remove at a later point.)


Again, I calculate the Squared Error from each iteration $m$ by taking the square of the difference between the estimated coefficients $\hat{α}_m$ from the linear model and the $α_m$ newly simulated by the uniform distribtion within each iteration. I store it in `` `se_mc` `` and I calculate the Mean Squared Error `` `mse_mc` `` after the loop has finished.

I plot a boxplot of the Squared Errors. This is not particularly informative, but not to worry, as I will be comparing all my methods in plots after modelling is complete. 


```r
M <- 1000
n_mc <- 1000
mse_mc <- numeric(M)
for(m in 1:M){
  a_mc <- runif(1,-1,1) # generating a using a distribution (in this case uniform). This is the Monte-Carlo element
  error.model=function(n_mc){rnorm(n_mc, 0, 1)}
  x_mc <- arima.sim(model<-list(order<-c(1,0,0), ar=a_mc), n_mc, rand.gen=error.model) 
  xt_mc <- x_mc[-1]
  xt_mc_1 <- x_mc[-n]
  lm_mc <- lm(xt_mc~0+xt_mc_1)
  mse_mc[m] <- mean((a_mc - summary(lm_mc)$coefficient[1,1])^2)
}
```

### Monte Carlo Simulation: 2nd method - using AR(1) formula manually

The second MC method involves coding the AR(1) formula manually instead of `` `arima.sim()` ``. Remember, from earlier, the AR(1) formula is $X_t = αX_{t-1} + ϵ_t$. Apart from that, I follow the same steps as with the first MC method. Theoretically, my two MC methods are the same, but there will be slight fluctuations just to the random distribution generators `` `runif()` `` and `` `rnorm()` ``. This second method is simply for pedagogical purposes.


```r
mse_mc_ar <- numeric(M) # using _mc_ar to mean '_montecarlo_ar manually'

for(m in 1:M){
  a_mc_ar <- runif(1,-1,1) # I believe that keeping 'a' w/in (-1,1) ensures stationarity
  x_mc_ar <- numeric(n_mc) # no need to redefine n (n_mc)
  e_mc_ar <- rnorm(n_mc, 0, 1)
  for(i in 2:n_mc){ # starting from 2 because I need to reference X_mc_ar[2-1] inside the loop. R is 1-indexed, not 0-indexed!
    x_mc_ar[i] <- a_mc_ar*x_mc_ar[i-1]+e_mc_ar[i]
  }
  xt_mc_ar <- x_mc_ar[-1]
  xt_mc_ar_1 <- x_mc_ar[-n_mc]
  lm_mc_ar <- lm(xt_mc_ar~0+xt_mc_ar_1)
  
  mse_mc_ar[m] <- mean(a_mc_ar - summary(lm_mc_ar)$coefficient[1,1])^2
}

plot.ts(x_mc_ar, main="Xt: 2nd Monte Carlo Method", ylab="Xt_mc_ar")
```

![](Time-Series-Regression_Knit_Git_files/figure-html/monte carlo 2-1.png)<!-- -->

# Statistically Significant Differences in MSE's - Boxplots and Kruskall-Wallis

## Distributions of MSE's

Let's visualise the distribution of the MSE's returned by the three methods.


```r
df_mse = data.frame(mse_boot, mse_mc, mse_mc_ar)

ggplot(stack(df_mse), aes(x=values)) + geom_density(aes(group=ind, color=ind, fill=ind), alpha=.3, ) +
  ggtitle("Distributions of MSE's")
```

![](Time-Series-Regression_Knit_Git_files/figure-html/MSE distributions-1.png)<!-- -->

Given the framing of our problem, I assume that the shapes of these distributions will always be right-skewed and similar to one another. This allows me to apply the Kruskal-Wallis test when running these steps on multiple different AR(1) models below.

## Boxplots

Next, I will create a boxplot for the set of SE's generated by each method. I decided to use Notched Boxplots, because the notch around the median allows me to assess whether there is a significant difference between the medians. I will reference @chambers_graphical_1983 in the following few paragraphs, but I want to note that the particular book was heavily influenced by John Tukey who is credited with creating the first box plot. 

Notches are calculated by:

\begin{equation} 
  CI = M \pm 1.57\times\dfrac{IQR}{\sqrt{n}} 
  (\#eq:tukey)
\end{equation}

Here $M$, $IQR$ and $n$ are the median, interquatile range and number of observations, respectively, for each subset of the data in turn. 



Statistical details of the derivation of \@ref(eq:tukey) are provided in  
  * https://stats.stackexchange.com/questions/184516/why-is-the-95-ci-for-the-median-supposed-to-be-%C2%B11-57iqr-sqrtn *https://stats.stackexchange.com/questions/228719/box-plot-notches-vs-tukey-kramer-interval .

I am comfortable referencing Stack Exchange answers by user glen_b because I regularly come across his answers, and they are always detailed and applauded by other well accredited or admin users on the site.

According to Chambers, *"if the notches for two boxes do not overlap, we can regard it as strong evidence that a similar difference in levels would be seen in other sets of data collected under similar circumstances"*. In simpler terms, when the notches do not overlap, we can conclude that the samples are from different populations. Regarding more than two boxplot comparisons, Chambers states that in *"cases where there are three or more box plots, we can use the notches to make a similar judgement about each pair of data sets"*. \@ref(eq:tukey) is based on the formal concept of a hypothesis test such that *"if the two data sets are independent and identically distributed (iid.) random samples from two populations with unknown medians but with a normal distributional shape in the central portion, then the notches provide an approximate 95% test of the null hypothesis that the true medians are equal"*. Now, this, as you will see, is a cause for some contemplation on my part. I take *central portion* to mean that portion of the boxplot that lies between the whiskers (i.e. excluding only the outliers at each end). As you will see after I plot the boxplots, what I take to mean the *central portion* does not resemble a normal distribution, so, perhaps the use of \@ref(eq:tukey) to assess statistically significant differences in the medians is not suitable here. Upon further reading, Chambers comments that *"notches are useful guides for comparing median levels even when the requirements for the hypothesis test are not strictly met"*. In addition, according to @wickham_stryjewski_boxplots *"The length of the confidence interval is determined heuristically so that non-overlapping intervals imply (approximately) a difference at the 5% level, regardless of the underlying distribution."* An important point to note here is that Chambers does not talk about sample size, perhaps due to the fact that $n$ is expicitly accounted for in \@ref(eq:tukey). Regardless, given that I have three samples of 1000 points, I cannot imagine that I am violating any hitherto untold sample size requirements. Finally, there is the question about multiple comparisons, which rears its head we are comparing more than two subsets of data, as in this case. The notches are not adjusted to allow for multiple comparisons, and so it *"should be kept firmly in mind in interpreting the plot that even under the null hypothesis that all of the p medians are equal, the probability that at least one of the pairs of notches do not overlap will be greater than .05"* (@chambers_graphical_1983). Technical adjustments are possible, akin to Bonferroni Correction, I suspect, but I am not going to look into that for now, as it is more technical than I need to go, considering that I will use another different formal test below. 

It is worth noting that @arnold_enhancing_2011 suggest using a very similar formula to \@ref(eq:tukey) to make conclusions from boxplots. They use the more traditional term of confidence intervals in lieu of notches, but the essence is the same. I prefer the rigid approach of Chambers, but I am including Arnold et al.'s formula for reference.

\begin{equation} 
  M \pm 1.5\times\dfrac{IQR}{\sqrt{n}} 
  (\#eq:arnold)
\end{equation}

Of course, inspection of boxplots constitutes an informal test, but, due to the detailed statistics behind the formation of the notches, I view the above assessment of the boxplots as being very close to a formal test. That said, due the presence of non-normal distributions, I shall couple the inspection of the boxplots with an official non-parametric formal test, namely, the Kruskall-Wallis test, to make a decision on statistically significant differences in the medians.

### Boxplot Links

* Boxplots in R: https://www.r-graph-gallery.com/boxplot.html
* Notes on Notched Boxplots: https://sites.google.com/site/davidsstatistics/home/notched-box-plots
* Relevant pages of Chambers' book: https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxkYXZpZHNzdGF0aXN0aWNzfGd4OjYxNTMzNTE4ZmY4Y2ZhNGQ


```r
stack(df_mse) %>%
  ggplot(aes(x=ind, y=values)) +
    geom_boxplot(aes(x=ind, y=values, fill=ind), alpha=.3, # color=ind, 
      
      # Notch?
        notch=TRUE,
        notchwidth = 0.8
    ) +
    # geom_jitter(color="black", size=0.4, alpha=0.9) + # to see the underlying distribution 
    ggtitle("MSE Boxplots for Comparison (w/o jitter)") +
    xlab("")
```

![](Time-Series-Regression_Knit_Git_files/figure-html/boxplots-1.png)<!-- -->

Given that the notches of the boxplots are quite narrow, it is very hard to assess if they overlap. Thus, I will return the values of \@ref(eq:tukey) instead of relying on visual inspection of the boxplots. I cannot get the values of the notches from the boxlpot function, so I'm doing it manually instead.


```r
medians = c() 
for(i in 1:dim(df_mse)[2]){
  medians[i] <- median(df_mse[[i]]) # [[]] returns a list, [] returns a dataframe - https://www.r-bloggers.com/r-accessors-explained/
}

iqrs=c()
for(i in 1:dim(df_mse)[2]){
  iqrs[i] <- IQR(df_mse[[i]])
}

tukey_formula <- function(M, IQR, N){
  CI_lower <- M-1.57*(IQR/sqrt(N))
  CI_upper <- M+1.57*(IQR/sqrt(N))
  c(CI_lower, CI_upper)
}

CIs <- data.frame()
for(i in 1:dim(df_mse)[2]){
  j=1
  CIs[i, j] = tukey_formula(medians[i], iqrs[i], dim(df_mse)[1])[j]
  CIs[i, j+1] = tukey_formula(medians[i], iqrs[i], dim(df_mse)[1])[j+1]
}

# Changing (by transposing) the shape of dataframe to plot it easily
tCIs = data.frame(values=c(t(data.matrix(CIs))), ind=rep(colnames(df_mse),each=2))
tCIs %>%
  ggplot(aes(x=ind, y=values)) + geom_line(aes(x=ind, y=values, color=ind)) +
  ggtitle("Lines Corresponding to Notches (Zoomed in)") +
  xlab("")
```

![](Time-Series-Regression_Knit_Git_files/figure-html/zoomed notches plot-1.png)<!-- -->

## Comment on Zoomed Plot 

I outline both potential scenarios due to the randomness of the data. 

* Scenario 1: All of the notches (indicated by the lines in the zoomed plot) overlap. Therefore, we do not have any statistically significant difference between the populations of MSE's calculated by the three methods. Let's follow this up with the Kruskal- Wallis test. 

* Scenario 2: One or more of the notches do not overlap. Therefore, we have a statistically significant difference between whichever methods do not have overlapping notches. Again, let's follow up with the Kruskall-Wallis test to be sure.

## Kruskal-Wallis

As seen earlier, the MSE's are non-normally distributed, so I cannot perform an ANOVA to assess if there is a statistically significant difference between their means or medians (medians in the case of non-parametric tests). Although the data appears to resemble a log-normal distribution, I am going to opt for a non-parametric version of an ANOVA. The Kruskal-Wallis (H) test is a non-parametric equivalent to the One-way ANOVA. It extends the Mann–Whitney U test, which is used for comparing only two groups.

Given that shapes of the distributions of the three groups are similar, I am going to use the Kruskal-Wallis (H) test to assess whether or not the medians of the levels are equal. If the shapes of the distributions were not similar, then the null hypothesis would be that the "Mean ranks of the groups are equal". In summary:

  * $H_0$: The three probability distributions are the same.
    * Specifically (given our distributions here) the medians of the groups are equal.
  * $H_A$: The three probability distributions are **not** the same i.e. at least two are different from each other.
    * Specifically (given our distributions here) the medians of at least two groups are different.
  
Another way of writing $H_0$ using words specific to this project would be "there is no difference in the MSE's calculated by the three different methods".


```r
K_W <- kruskal.test(values ~ ind, data = stack(df_mse))
```

Again, I will detail both scenarios:

* Scenario 1: p-value > 0.05 is non-significant at the 95% confidence level. Thus, we cannot reject $H_0$, and so we conclude that the three sample distributions come from the same population.

* Scenario 2: p-value < 0.05 is significant at the 95% confidence level. Thus, we reject $H_0$, and we conclude that the three sample distributions do not come from the same population.




## Tansformation of Data ?

Note: I could transform my log-normal data to make it normal in order to assess using different statistical tests. I am not doing this right now.

# Running it all together

Writing all the code in one block in order to repeat my steps multiple times and thus return a histogram showing distribution of p-values. 
From http://varianceexplained.org/statistics/interpreting-pvalue-histogram/ :
The distribution of p-values will be right skewed (similar to log-normal) if $H_0$ is false. If $H_0$ is true, I expect the p-value distribution to be uniform.
Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6164648/


```r
K = c() # Vector K will contain p-value for each iteration k of the outer loop. R code: is.vector(K)

# Bootstrapping
b <- 1000
coef_boot <- matrix(NA, nrow=b, ncol=2)
mse_boot <- matrix(NA, nrow=b, ncol=1)

# Monte Carlo
M <- 1000
n_mc <- 1000
mse_mc <- numeric(M)
mse_mc_ar <- numeric(M) # using _mc_ar to mean '_montecarlo_ar manually'

  
for(k in 1:10){ # K=1000 takes too long to run
  x <- arima.sim(list(order=c(1,0,0), ar=.5), n=1000) # simulating AR(1) model 
  n <- length(x)
  
  xt <- x[-1]
  xt_1 <-	x[-n] # one lagged time series
  
  # Bootstrapping
  for(i in 1:b){ # Perform this b times, to create b different samples and lm models
    is <- sample(c(1:n-1), size<-n, replace<-TRUE) # bootstrapping to create a sample of size n
    out <- lm(xt[is]~ 0 + xt_1[is]) # fit lm model on bootstrapped sample (by referencing bootstrapped indices in original sample)
    summary(out)
    coef_boot[i,] <- coef(out) # fill row i with coefficients of the lm model
    mse_boot[i,] <- ((coef(lm1)-coef(out))^2) # return the squared error
  }
  
  # Monte Carlo
  for(m in 1:M){
    a_mc <- runif(1,-1,1) # generating a using a distribution (in this case uniform). This is the Monte-Carlo element
    error.model <- function(n_mc){rnorm(n_mc, 0, 1)}
    x_mc <- arima.sim(model<-list(order<-c(1,0,0), ar=a_mc), n_mc, rand.gen=error.model) 
    xt_mc <- x_mc[-1]
    xt_mc_1 <- x_mc[-n]
    lm_mc <- lm(xt_mc~0+xt_mc_1)
    mse_mc[m] <- mean((a_mc - summary(lm_mc)$coefficient[1,1])^2)
  }
  
  # Monte Carlo - manual AR
  for(m in 1:M){
    a_mc_ar <- runif(1,-1,1) # I believe that keeping 'a' w/in (-1,1) ensures stationarity
    x_mc_ar <- numeric(n_mc) # no need to redefine n (n_mc)
    e_mc_ar <- rnorm(n_mc, 0, 1)
    for(i in 2:n_mc){ # starting from 2 because I need to reference X_mc_ar[2-1] inside the loop. R is 1-indexed, not 0-indexed!
      x_mc_ar[i] <- a_mc_ar*x_mc_ar[i-1]+e_mc_ar[i]
    }
    xt_mc_ar <- x_mc_ar[-1]
    xt_mc_ar_1 <- x_mc_ar[-n_mc]
    lm_mc_ar <- lm(xt_mc_ar~0+xt_mc_ar_1)
    
    mse_mc_ar[m] <- mean(a_mc_ar - summary(lm_mc_ar)$coefficient[1,1])^2
  }
  
  df_mse <- data.frame(mse_boot, mse_mc, mse_mc_ar) 
  K_W <- kruskal.test(values ~ ind, data = stack(df_mse))
  K[k] <- K_W$p.value
}

ggplot() + aes(K) + 
  geom_histogram(colour="black", fill="white", breaks = seq(0, 1, by=1/10)) +
  ggtitle("Distribution of p-values") + xlab("p-value") + scale_x_continuous(breaks = seq(0,1,.1))
```

![](Time-Series-Regression_Knit_Git_files/figure-html/all together-1.png)<!-- -->

# Comments/Conclusion

Given the right skewed distribution, I conclude that $H_0$ is false. Thus, there is a difference in the MSE's across the three methods.
Although Linear Regression is suitable (as the assumptions are met [Assumed from one assessment!]), I cannot conclude that it is a robust model to predict AR(1) time series.

# Possible Further Improvements

Plot all the confidence intervals also.

Report effect size (strength of association)..but of what?!

Use confidence intervals and p-values everywhere for two reasons:
1. P-values are not reproducible when using different samples from the same distribution. This is discussed in @taleb_pvalue_distribution.
2. CI's give more info about the variance of the parameter of interest.

Use wilcoxon test or something similar to assess which group(s) are different from the others. All I can say after K_W test is that a difference exists.

Try to extend this analysis of Linear Regression to predicting AR(2),...,AR(n) time series.
Attempt the analysis on MA(1),...,MA(n) time series

## References
