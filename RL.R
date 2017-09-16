#********************************** reinforcement learning *****************************

mapped.market.states <- function(n, r, upper, lower, HistSeries, random.seed){
	
	if(!is.null(random.seed)){
		set.seed(random.seed)
	}
	
	k <- round((upper+lower)/2)
	m <- n*(k+1)
	M <- matrix(0, ncol = 3, nrow = m)  # ncol = 5
	
	if(is.null(HistSeries)){
		s <- seq(0, 2, by = 2 / (n-1))
		
		#  B <- runif(n = length(s), 1.6, 2)
		for(i in 1 : n){
			for(j in 1 : (k+1)){
				index <- (i-1)*(k+1)+j
				M[index,1] <- s[i]              # S/K
				M[index,2] <- runif(1,0.1,0.6)  # vol
				M[index,3] <- runif(1,0,1)      # T-t
				#      M[index,4] <- B[i] - s[i]
				#      M[index,5] <- B[i]
				if(j == 1 | j == k+1) { M[index,3] <- 0 }  
			}
		}
	}else{
		s <- sample(x = HistSeries, size = m, replace = T)/sample(x = HistSeries, size = m, replace = T)
		for(i in 1:m){
			M[i,1] <- s[i]              # S/K
			M[i,2] <- runif(1,0.1,0.6)  # vol
			M[i,3] <- runif(1,0,1)      # T-t
		}
	}
	return(M)
}

PHI <- function(mkt.state, mkt.state.predecessor, r, b){
	U.prime <- mkt.state[1] # + 0.001
	tau.prime <- mkt.state[3]
	#  dist.S.B.prime <- mkt.state[4]
	
	U <- mkt.state.predecessor[1]# + 0.001
	sigma <- mkt.state.predecessor[2] 
	tau <- mkt.state.predecessor[3] 
	# dist.S.B <- mkt.state.predecessor[4]
	
	time.lapse <- tau-tau.prime
	
	z <- log(U.prime/U-(r-0.5*sigma^2)*tau)
	z[is.na(z)] <- 0
	if(time.lapse <= 0){
		ni <- 10^20 
	}
	if (time.lapse > 0){ # & dist.S.B < 0
		ni <- 1 # ni <- 10^20 
	} else { 
		ni <- 1 }
	
	phi <- ni*abs(U-U.prime)*exp((r-0.5*sigma^2)*tau+sigma*z*sqrt(tau))     # LogNormal kernel d.bar(U,U.prime)
	output <- exp(-phi/(b^2))
	return(output)
}

mapped.states.transitions.probabilities <- function(n, r, upper, lower, b, HistSeries, random.seed){
	
	if(!is.null(random.seed)){
		set.seed(random.seed)
	}
	
	cat('Simulating market state data: ',
			n * (round((upper+lower)/2) + 1), 'x', 3,
			'\n')
	M <- mapped.market.states(n, r, upper, lower, HistSeries, random.seed)
	m <- nrow(M)
	transition.matrix <- matrix(0,m,m)
	
	cat('Computing transition probability matrix (dim: ', 
			n * (round((upper+lower)/2) + 1), 'x', 
			n * (round((upper+lower)/2) + 1)
			,' )\n')
	
	for(i in 1 : m){
		Z <- 0
		for(j in 1 : m){
			transition.matrix[i,j] <- PHI(mkt.state = M[j,], mkt.state.predecessor = M[i,], r, b = b)
			Z <- Z + transition.matrix[i,j] 
		}
		for(j in 1:m){
			transition.matrix[i,j] <- transition.matrix[i,j]/Z
		}
		cat('TPM: completed row ', i, ' of ', m,'\n')
	}
	return(list(transition.matrix=transition.matrix,
					M=M, m=m))
}

reward.barrier <- function(predecessor.state, successor.state, a, r, B.std){
	state.tau <- successor.state[3]
	successor.state.S <- successor.state[1]
	predecessor.state.tau <- predecessor.state[3]
	predecessor.state.S <- predecessor.state[1]
	predecessor.dist.S.B <- predecessor.state[4]
	
	if(a == 1 | state.tau == 0){
		return(max(predecessor.state.S-1, 0 )) # EXERCISE
	}
	if(a == 2){
		return(0)   # HOLD: no reward today my friend,sorry
	}
}

reward <- function(s, a, r){
	state.price.to.strike <- s[1]
	state.vol <- s[2]
	state.tau <- s[3]
	
	if(a == 1){
		return(exp(r*state.tau)*max(state.price.to.strike-1,0 )) # EXERCISE
	}
	if(a == 2){
		return(0)   # HOLD: no reward today my friend,sorry
	}
}



Value.iteration <- function(n,r,b,lower,upper,actions,tol, HistSeries, random.seed){
	if(!is.null(random.seed)){
		set.seed(random.seed)
	}
	
	output <- mapped.states.transitions.probabilities(n,r,upper,lower,b, HistSeries, random.seed)
	transition.matrix <- output$transition.matrix
	transition.matrix[is.na(transition.matrix)] <- 0
	
	M <- output$M
	m <- output$m
	V <- array(0,dim = m)
	A <- array(0,dim = m)
	delta <- tol+1
	
	while(delta > tol){
		delta <- 0
		for(i in 1:m){
			v <- V[i]
			if(M[i,3] == 0){
				V[i] <- reward(s = M[i,], a = actions[1], r = r)   # if T-t == 0: EXERCISE
				A[i] <- actions[1]
			} else {
				maxValue <- 0
				p <- sum(transition.matrix[i,] %*% V)    # action 2: HOLD payoff
				if(p > maxValue){
					A[i] <- 2
					maxValue <- p
				}
				V[i] <- maxValue
			}
			delta <- max(delta, abs(v-V[i]))
		}
		cat(delta,"\n")
	}
	return(list(V=V,A=A))
}

price <- function(M,V,d,b,r){
	m <- array(0,3)
	m[1] <- d[1]/d[2]
	m[2] <- d[3]
	m[3] <- d[4]
	
	P <- 0
	Z <- 0
	for(i in 1 : nrow(M)){
		phi <- PHI(mkt.state.predecessor = m, mkt.state = M[i,], 
				b = b, r = r)
		P <- P + phi*V[i]
		Z <- Z + phi
	}
	return(exp(-r * m[3]) * d[2] * P/Z)
}



