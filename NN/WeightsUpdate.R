# Set of functions called by DNNRegressor.R

WeightsUpdate <- function(Weights, method, 
		SLRParam, BackpropDerivative, 
		t, eta, LearningRate){
	
	SLR.parameters <- list()
	if(!is.null(method)){
		for(k in 1 : length(Weights)){
			if(method == 'Adam'){
				ArgumentsList <- list(m = SLRParam[[k]]$m, v = SLRParam[[k]]$v, Theta = Weights[[k]],
						Derivative = BackpropDerivative[[k]]$WeightsDerivative, 
						t = t, alpha = eta, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
			} else{
				ArgumentsList <- list(Theta = Weights[[k]],
						Derivative = BackpropDerivative[[k]]$WeightsDerivative,
						v = SLRParam[[k]]$v, 
						alpha = eta, gammaParam = 0.9)
			}
			output <- do.call(what = method, args = ArgumentsList)
			
			SLR.parameters$WeightsUpdated[[k]] <- output$ThetaUpdated
			SLR.parameters$Parameters[[k]] <- output$SLRParamList
			
		}
	} else {
		for(k in 1 : length(Weights)){
			SLR.parameters$WeightsUpdated[[k]] <- Weights[[k]] - LearningRate*BackpropDerivative[[k]]$WeightsDerivative
			
		}
	}
	return(SLR.parameters)
}




Adam <- function(m, v, Theta, Derivative, t, alpha,
				beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8){
	
	if(!is.null(dim(m))){
		mW <- m[1:(nrow(m)-1),] ; mb <- m[nrow(m),]
		vW <- v[1:(nrow(v)-1),] ; vb <- v[nrow(v),]
	}else{
		mW <- m[1:(length(m)-1)] ; mb <- m[length(m)]
		vW <- v[1:(length(v)-1)] ; vb <- v[length(v)]
	}
	
	dW <- Derivative[1 : (nrow(Derivative)-1), ]
	db <- Derivative[nrow(Derivative), ]
	
	# check dim of m and v
	mW <- beta1*mW + (1-beta1)*dW
	mb <- beta1*mb + (1-beta1)*db
	
	vW <- beta2*vW + (1-beta2)*(dW*dW)
	vb <- beta2*vb + (1-beta2)*(db*db)
	
	if(is.null(dim(mW))){
		if(nrow(m) == 2){
			numeratorM <- rbind(mW, mb)
			numeratorV <- rbind(vW, vb)
		} else {
			numeratorM <- append(mW, mb)
			numeratorV <- append(vW, vb)
		}	
	}else{
		numeratorM <- rbind(mW, mb)
		numeratorV <- rbind(vW, vb)
	}
	
	mHat <- numeratorM/(1-beta1^t)
	vHat <- numeratorV/(1-beta2^t)
	
	WeightsUpdated <- Theta - alpha * mHat/(sqrt(vHat) + epsilon)
	
	if(!is.null(dim(mW))){
		m = as.matrix(rbind(mW, mb), ncol = ncol(Theta), nrow = nrow(Theta))
		v = as.matrix(rbind(vW, vb), ncol = ncol(Theta), nrow = nrow(Theta))
	}else{
		if(nrow(m) == 2){
			m = as.matrix(rbind(mW, mb), ncol = ncol(Theta), nrow = nrow(Theta))
			v = as.matrix(rbind(vW, vb), ncol = ncol(Theta), nrow = nrow(Theta))
		} else {
			m = as.matrix(append(mW, mb))
			v = as.matrix(append(vW, vb))
		}
		
	}
	
	return(list(ThetaUpdated = as.matrix(WeightsUpdated, ncol = ncol(Theta),
							nrow = nrow(Theta)),
					SLRParamList = list(m = m, v = v )
			)
	)
}


Momentum <- function(Theta, Derivative, v, alpha, gammaParam){
	
	if(!is.null(dim(v))){
		vW <- v[1:(nrow(v)-1),] ; vb <- v[nrow(v),]
	}else{
		vW <- v[1:(length(v)-1)] ; vb <- v[length(v)]
	}
	
	dw <- Derivative[1 : (nrow(Derivative)-1), ]
	db <- Derivative[nrow(Derivative), ]
	
	
	vW <- gammaParam * vW + alpha * dw
	vb <- gammaParam * vb + alpha * db
	
	v <- if(is.null(dim(vW)) & is.null(dim(v))){append(vW, vb)}else{rbind(vW, vb)}
	
	WeightsUpdated <- Theta - v
	
	return(list(ThetaUpdated = as.matrix(WeightsUpdated, ncol = ncol(Theta), 
							nrow = nrow(Theta)),
					SLRParamList = list(v = v )
			)
	)
}

# BUG!!!!!!!! <- fix if line 135, 136
NAG <- function(Theta, Derivative, v, alpha, gammaParam){
	
	if(!is.null(dim(Theta))){
		W <- Theta[1:(nrow(Theta)-1),] ; b <- Theta[nrow(Theta),]
	}else{
		W <- Theta[1:(length(Theta)-1)] ; b <- Theta[length(Theta)]
	}
	
	if(!is.null(dim(v))){
		vW <- v[1:(nrow(v)-1),] ; vb <- v[nrow(v),]
	}else{
		vW <- v[1:(length(v)-1)] ; vb <- v[length(v)]
	}
	
	dw <- Derivative[1 : (nrow(Derivative)-1), ]
	db <- Derivative[nrow(Derivative), ]

	# vW <- gammaParam*vW + alpha * (W - gammaParam*vW) 
	# vb <- gammaParam*vb + alpha * (b - gammaParam*vb)
	
	vw_lag <- vW ; vb_lag <- vb
	
	vW <- gammaParam*vW + alpha * dw
	vb <- gammaParam*vb + alpha * db
	
	
	#v <- if(is.null(dim(vW))){append(vW, vb)}else{rbind(vW, vb)}
	v_lag <- if(is.null(dim(vw_lag))){append(vw_lag, vb_lag)}else{rbind(vw_lag, vb_lag)}
	v <- if(is.null(dim(vW))){append(vW, vb)}else{rbind(vW, vb)}	
	
	# WeightsUpdated <- Theta - v
	WeightsUpdated <- Theta - gammaParam*v_lag + (1 + gammaParam) * v
	
	return(list(ThetaUpdated = as.matrix(WeightsUpdated, ncol = ncol(Theta), 
							nrow = nrow(Theta)),
					SLRParamList = list(v = v )
			)
	)
}
