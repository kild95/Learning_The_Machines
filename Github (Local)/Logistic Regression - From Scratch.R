# derivations: http://personal.psu.edu/jol2/course/stat597e/notes2/logit.pdf
data(BreastCancer, package="mlbench") # description of vars: https://sites.google.com/a/googlesciencefair.com/science-fair-2012-project-64a91af142a459cfb486ed5cb05f803b2eb41354-1333130785-87/observations
bc <- BreastCancer[complete.cases(BreastCancer),] 
# remove id col
bc <- bc[, -1]

for(i in 1:9) {
  bc[, i] <- as.numeric(as.character(bc[, i])) 
}

bc$Class <- ifelse(bc$Class == "malignant", 1, 0)
bc$Class <- factor(bc$Class, levels = c(0, 1))

library(caret)
'%ni%' <- Negate('%in%')
options(scipen=999)  # prevents printing scientific notations.

# Prep Training and Test data.
set.seed(100)
data_index <- createDataPartition(bc$Class, p=0.7, list = F)  # 70% training data
train_data <- bc[data_index, ]
test_data <- bc[-data_index, ]

table(train_data$Class) # class imbalance present after split (obviously)

down_train <- downSample(x = train_data[, colnames(train_data) %ni% "Class" ], y=train_data$Class) # only DownSample train_data to aid learning (thus no need to DownSample test_data)
table(down_train$Class) 
x_train <- down_train[,colnames(down_train) %ni% "Class"]
y_train <- down_train$Class
# x_train is already in the design matrix shape used in mathematicalmonk youtube derivations
x_train <- cbind(1, x_train)
# transposing so I can transpose back again in order to stick with what I see in https://www.puzzlr.org/write-your-own-logistic-regression-function-in-r/
X_mod <- t(x_train) 

sigmoid <- function(z){
  a <- (1/(1+exp(-z)))
}

w = rnorm(10, mean = 0, sd= 1)
W = matrix(w, nrow = length(x_train[1,]), ncol = 1)

alpha_fn <- function(X, W){
  #result <- sigmoid(as.matrix(X) %*% W)
  result <- sigmoid(t(as.matrix(X)) %*% W)
}

gradient <- function(X, alpha, y){
  grad <- t(X) %*% (alpha-as.numeric(as.character(y)))
}

hessian <- function(X, B){
  hess <- t(as.matrix(X)) %*% B %*% as.matrix(X)
}

newton_rhapson <- function(X, B, W, H, g, y, alpha){
  X <- as.matrix(X)
  # calc <- alpha-as.numeric(as.character(y))
  #W_update <- W - ( solve(t(X) %*% B %*% X) %*% t(X) %*% B %*% (solve(B) %*% (alpha-as.numeric(as.character(y)))))
  W_update <- W - ( solve(t(X) %*% B %*% X) %*% t(X) %*% (as.numeric(as.character(y))-alpha))
  w <- W_update
}

logistic <- function(X, y, W, max_iters){
  alpha <- alpha_fn(X, W)
  g <- gradient(X, alpha, y) # could use <<- to make g a global var but don't wanna mess with these for now
  B <- diag(as.vector(alpha*(1-alpha)))
  H <- hessian(X, B)
  newton_rhapson(X, B, W, H, g, y, alpha)
  while (i <= max_iters) {
    W <- newton_rhapson(X, B, W, H, g, y, alpha)
    i <- i+1
  }
  return("W"=W)
  #return(list("alpha" = alpha, "g" = g, "B" = B, "W" = W))
}

logistic(X_mod, y_train, W, 10)
#logistic(x_train, y_train, W, 50)
#logistic(x_train, y_train, W, 100)

# Checking with normal glm function
log_model <- glm(Class ~ ., data = down_train, family = "binomial")
summary(log_model)

