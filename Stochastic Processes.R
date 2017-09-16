#******************************* VG process **************************************
ll <- function(par){
  if(par[2]>0 & par[4]>0) {
    return( -sum(log( dvg(data, vgC=par[1], sigma=par[2], theta=par[3], nu=par[4]) ) ) ) 
  } else {
    return(Inf)
  }
}

VGprocess <- function(param, S0, d, r, nsim){
  d <- 20 
  t <- d/252 
  delta.t <- seq(0,t,length.out = d)
  
  sigma <- param.optim[1]
  theta <- param.optim[2]
  nu <- param.optim[3]
  
  w <- t/nu*log(1-theta*nu-sigma^2*nu/2)
  S <- matrix(0,ncol=d,nrow=nsim) 
  S[,1] <- S0
  for(i in 2:d){
    for(j in 1:nsim){
      g <- qgamma(runif(1,0,1), shape=delta.t[i]/nu, scale=nu)
      z <- g*theta + sigma*sqrt(g) * qnorm(runif(1,0,1))
      S[j,i] <- S[j,i-1]*exp(r*t + w + z)
    }
  }
  return(S)
}

VGprocess.faster.qgamma <- function(S0, T.t, r, nsim, nstep, param){
  dt <- T.t/nstep
  S <- matrix(0, nsim, (1+nstep)) 
  S[,1] <- rep(S0,nsim)
  
  sigma <- param[1]
  theta <- param[2]
  nu <- param[3]
  
  w <- T.t/nu*log(1 - theta*nu - sigma^2*nu/2)
  counter <- 0
  for(i in 2:(nstep+1)){
    counter <- dt + counter
    g <- qgamma(runif(nsim,0,1), scale=nu, shape=counter/nu)
    z <- g * theta + sigma * sqrt(g) * qnorm(runif(nsim,0,1))
    S[,i] <- S[,i-1]*exp(r*counter + w + z)
  }
  return(S)
}

VGprocess.faster.rgamma <- function(S0, T.t, r, nsim, nstep, param){
  dt <- T.t/nstep
  S <- matrix(0, nsim, (1+nstep)) 
  S[,1] <- rep(S0,nsim)
  
  sigma <- param[1]
  theta <- param[2]
  nu <- param[3]
  
  w <- t/nu*log(1-theta*nu-sigma^2*nu/2)
  counter <- 0
  for(i in 2:(nstep+1)){
    counter <- dt + counter
    g <- rgamma(nsim, shape=delta.t[i]/nu, scale=nu)
    z <- g*theta + sigma*sqrt(g) * rnorm(nsim,0,1)
    S[,i] <- S[,i-1]*exp(r*t + w + z)
  }
  return(S)
}


VGprocess.TimeChangedBM <- function(T.t, nstep, nsim, theta, sigma, nu, p){
  X <- matrix(0, ncol = nstep+1, nrow = nsim)
  timeStep <- seq(0, T.t, length.out = nstep)
  X[, 1] <- 0
  for(i in 1 : nstep){
    for(j in 1 : nsim){
      deltaG <- qgamma(p = 0.9, shape = timeStep[i]/nu, scale = nu)
      deltaN <- rnorm(n = 1, mean = 0, sd = 1)
      X[j, i+1] <- X[j, i] + theta * deltaG + sigma * sqrt(deltaG) * deltaN
    }
  }
  return(X)
}


#******************************** Heston process *********************************
ornstein_uhlenbeck <- function(T.t,nstep,nsim,theta1,theta2,theta3,x0){
  dt  <- T.t/nstep
  Z <- matrix(0, ncol = nstep, nrow = nsim)
  X <- matrix(0, ncol = (1+nstep), nrow = nsim)
  X[,1] <- x0
  for (i in 2:(nstep+1)){ 
    Z[,i-1] <- rnorm(nsim, 0, sqrt(dt))
    X[,i]  <-  X[,i-1] + (theta1 - theta2*X[,i-1])*dt + theta3*sqrt(X[,i-1])*Z[,i-1]
  }
  return(OU <- list(X = X,
                    Z = Z))
}

Assetpaths <- function(S0, T.t, r, nsim, nstep, V, Z.V, rho){
  dt = T.t/nstep
  S = matrix(0, nsim, (1+nstep)) 
  S[,1] = rep(S0,nsim)
  for(i in 2:(nstep+1)){ 
    eps <- rnorm(nsim)
    S[,i] <- S[,i-1]*exp((r-0.5*V[,i-1])*dt + sqrt(V[,i-1])*(rho*Z.V[,i-1] + sqrt((1-rho^2))*dt*eps))
  }
  return(S)
}

HestonSim_SDE <- function(S0, T.t, r, nsim, nstep, V, Z.V, rho){
  dt = T.t/nstep
  if(!is.null(S0)){
    S = matrix(0, nsim, (1+nstep)) 
    S[,1] = rep(S0,nsim)
    for(i in 2:(nstep+1)){ 
      eps <- rnorm(nsim)
      S[,i] <- S[,i-1]*exp((r-0.5*V[,i-1])*dt + sqrt(V[,i-1])*(rho*Z.V[,i-1] + sqrt((1-rho^2))*dt*eps))
    }
  } else {
    S = matrix(0, nsim, (1+nstep)) 
    S[,1] = rep(0,nsim)
    for(i in 2:(nstep+1)){ 
      eps <- rnorm(nsim)
      S[,i] <- (r-0.5*V[,i-1])*dt + sqrt(V[,i-1])*(rho*Z.V[,i-1] + sqrt((1-rho^2))*dt*eps)
    }
  }
  return(S)
}

expBes <- function(x,nu){
  mu <- 4*nu ^2
  A1 <- 1
  A2 <- A1 * (mu - 1) / (1 * (8*x))
  A3 <- A2 * (mu - 9) / (2 * (8*x))
  A4 <- A3 * (mu - 25) / (3 * (8*x))
  A5 <- A4 * (mu - 49) / (4 * (8*x))
  A6 <- A5 * (mu - 81) / (5 * (8*x))
  A7 <- A6 * (mu -121) / (6 * (8*x))
  1/ sqrt(2*pi*x) * (A1 - A2 + A3 - A4 + A5 - A6 + A7)
}

# fast algo
dcCIR <- function (x, t, x0 , theta , log = FALSE ){
  c <- 2* theta[2] /((1 - exp(- theta[2] *t))* theta[3]^2)
  ncp <- 2*c*x0*exp(- theta[2] *t)     # non centrality param of chi sq
  df <- 4* theta[1] / theta[3]^2       # df of condition prob of the chi sq con. prob
  u <- c*x0* exp (- theta [2] *t)
  v <- c*x
  q <- 2* theta [1] / theta [3]^2 -1
  lik <- ( log (c) - (u+v) + q/2 * log (v/u) + log ( expBes ( 2* sqrt (u*v), q))
           + 2* sqrt (u*v))
  if(!log )
    lik <- exp(lik)
  lik
}

CIR.lik <- function(theta1 , theta2 , theta3 ) {
  n <- length(X)
  dt <- deltat(X)
  -sum(dcCIR(x=X[2: n], t=dt , x0=X[1:(n -1)] ,theta=c(theta1 , theta2 , theta3 ),
             log = TRUE ))
}

# slow algo
dcCIR2 <- function(x, t, x0 , theta , log = FALSE ){
  c <- 2* theta[2] /((1 - exp(- theta[2] * t)) * theta[3]^2)
  ncp <- 2*c*x0 * exp(- theta[2] * t)
  df <- 4 * theta[1] / theta[3]^2
  lik <- (dchisq (2 * x * c, df = df , ncp = ncp , log = TRUE )
          + log (2*c))
  if(!log )
    lik <- exp( lik )
  lik
}

CIR.lik2 <- function(theta1 , theta2 , theta3 ) {
  n <- length(X)
  dt <- deltat(X)
  -sum( dcCIR2(x = X[2: n], t = dt , x0 = X[1 : (n - 1)] , theta = c( theta1 , theta2 , theta3 ),
                 log = TRUE ))
}

SimBM_AssetPrice <- function(S0, T.t, r, nsim, nstep, sd){
	dt = T.t/nstep
	S = matrix(0, nsim, (1+nstep)) 
	S[,1] = rep(S0,nsim)
	for(i in 2 : (nstep + 1)){ 
		eps <- rnorm(nsim)
		S[,i] <- S[,i-1]*exp((r-0.5*sd^2)*dt + sqrt(sd)*dt*eps)
	}
	return(S)
}
