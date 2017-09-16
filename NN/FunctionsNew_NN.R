# TODO: Add comment
# DONE : new dropout(DropoutInitializer, GenerateBinaryWithProbP, DropoutGetMask, ApplyMask)
#
# Author: OF
###############################################################################

# Set of functions called by DNNRegressor.R


ModelParameterCollector <- function(x, y, omodel, traindata, 
		testdata, hidden, maxit, abstol, batchsize,
		LayerWithDropout, ProbabilityOfSamplingOne,
		lr, reg, display, 
		dropout, alpha, random.seed,
		act.fct, 
		OutputActivation,
		LossFct, generalized, tau,
		SLR,
		inputAutoencoder,
		scale.regime){
	return(list(
					x=x, y=y, omodel=omodel, traindata=traindata,
					testdata=testdata, hidden=hidden, maxit=maxit, 
					abstol=abstol, batchsize=batchsize,
					LayerWithDropout=LayerWithDropout, ProbabilityOfSamplingOne=ProbabilityOfSamplingOne,
					lr=lr, reg=reg, display=display, 
					dropout=dropout, alpha=alpha, random.seed=random.seed,
					act.fct=act.fct, 
					OutputActivation=OutputActivation,
					LossFct=LossFct, generalized=generalized, tau=tau,
					SLR=SLR,
					inputAutoencoder=inputAutoencoder,
					scale.regime=scale.regime
			)
	)
}


initSLRParam <- function(SLR, NumWeights, NetArchitecture){
	SLRParam <- list()
	if(!is.null(SLR)){
		if(SLR == 'Adam'){
			for(i in 1 : NumWeights){
				m <- matrix(0, nrow = NetArchitecture[i] + 1, ncol = NetArchitecture[i+1]) # add 1 to NetArch because of bias term
				SLRParam[[i]] <- list(m = m, v = m)
			}
		} else {
			for(i in 1 : NumWeights){
				v <- matrix(0, nrow = NetArchitecture[i] + 1, ncol = NetArchitecture[i+1])
				SLRParam[[i]] <- list(v = v)
			}
		}
	} else{
		v <- NULL ; m <- NULL
	}
	return(SLRParam)
}

initLrStandard<- function(lr, h, l){
	learningRate <- list()
	for(i in 1:l){
		learningRate[[i]] <- rep(lr, h[i+1])
	}
	return(learningRate)
}

initWeightsStandard <- function(omodel = NULL, h, inputAutoencoder, l, X, ActivationFct,
                                hidden, NumWeights, OutputActivation){
	weights <- list()
	weightsAug <- list()
	
	Param <- list()
	if(is.null(omodel)) {
		if(inputAutoencoder$pre.training.ae == TRUE){
			init.w <- initWeightsAE(X = X, hidden = hidden, NumWeights = NumWeights, 
					ActivationFct = ActivationFct, OutputActivation = OutputActivation,
					OutputActivationAE = inputAutoencoder$OutputActivationAE,
					LossFct = inputAutoencoder$LossFct,
					generalized = inputAutoencoder$generalized,
					batchsize = inputAutoencoder$batchsize,
					NoiseDistribution = inputAutoencoder$NoiseDistribution,
					RegularizationParam = inputAutoencoder$RegularizationParam,
					LearningRateParam = inputAutoencoder$LearningRateParam,
					SLR = inputAutoencoder$SLR,
					abstol = inputAutoencoder$abstol, 
					maxit = inputAutoencoder$maxit
					)
			for(i in 1 : l){
				W <- init.w[[i]]$W
				b <- init.w[[i]]$b
				wb <- rbind(W,b)
				
				Param$Weights[[i]] <- list(W = W, b = b)
				Param$WeightsAug[[i]] <- wb
			}
		} else {
			for(i in 1 : l){
				W <- matrix(runif(h[i] * h[i+1], -1 / sqrt(h[i]), 1 / sqrt(h[i]) ), 
						nrow = h[i], ncol = h[i+1])                                   									# -1/sqrt(h[i]), 1/sqrt(h[i]) OR -sqrt(6/sum(h[i]+h[i+1])), sqrt(6/sum(h[i]+h[i+1]))
				b <- matrix(runif(h[i+1], -1 / sqrt(h[i]), 1 / sqrt(h[i])), nrow = 1, ncol = h[i+1])                        # 0 OR -1/sqrt(h[i]), 1/sqrt(h[i])
				wb <- rbind(W,b)
				
				Param$Weights[[i]] <- list(W = W, b = b)
				Param$WeightsAug[[i]] <- wb
			}
		}
	} else {
#		# hidden  <- omodel$hidden 
#		H <- length(hidden)
#		h <- c(ncol(X), hidden, 1)
#		l <- length(h) - 1
#		for(i in 1 : l){             
#			wb <- rbind(omodel$weights[[i]]$W, omodel$weights[[i]]$b) 
#			
#			Param$Weights[[i]] <- list(W = W, b = b)
#			Param$WeightsAug[[i]] <- wb
#		}
	Param$WeightsAug <- omodel$Weights
	}
	return(Param)
	
}

SampleMiniBatch <- function(X, Y, batchsize){
	index <- sample(1:nrow(X), batchsize , F)
	TrainM <- X[index, ]
	TestM <- Y[index]
	return(list(TrainM = TrainM,
					TestM = TestM))
}


#******************************* Forward propagation **************************
FeedForward <- function(X, NumWeights, weightsAug, mask, N,
		OutputActivation, LayerActivation){
	
	# X <- cbind(X, rep(1, nrow(X)))
	hidden.layer <- list()
	hidden.layer[[1]] <- list(activation.fct = cbind( ApplyMask(ObjectiveMatrix = X, 
							mask, ColIndex = 1), 
					rep(1, nrow(X)))
	)	
	
	for(k in 1 : NumWeights){
		layer <- hidden.layer[[k]]$activation.fct %*% weightsAug[[k]]
		if(k == NumWeights){
			activation.fct <- activation(fct = OutputActivation, layer)
		} else {
			activation.fct <- activation(fct = LayerActivation, layer)
			activation.fct <- ApplyMask(ObjectiveMatrix = activation.fct, mask, 
					ColIndex = k + 1)
			# Initrialize vectors of 1s before while, then just bind
			activation.fct <- cbind(activation.fct, rep(1, nrow(activation.fct)))
		}
		hidden.layer[[k+1]] <- list(layer = layer, activation.fct = activation.fct)
	}
	return(hidden.layer)
}

activation <- function(fct, X){
	if(fct == 'tanh'){ return(tanh(X))
	} else if(fct == 'logistic') { return(sigmoid(X))
	} else if(fct == 'ReLU') { return(pmax(X, 0))
	} else if(fct == 'Leaky ReLU') { return(ifelse(X >= 0, X, 0.1 * X))
	} else if(fct == 'linear') { return(X)
	}
}

#******************************* Dropout **************************************
DropoutInitializer <- function(InputAndHiddenDim = c(ncol(X), hidden)){
	require(plyr)
	
	NetDepth <- length(InputAndHiddenDim)
	if( NetDepth > 1 ){
		ls <- list()
		for( i in 1 : NetDepth ){
			box <- rep(1,length.out = InputAndHiddenDim[i])
			ls[[i]] <- list(box = c(box))
		}
		mask <- t(rep(1,length.out = NetDepth))
		for( i in 1 : NetDepth ){
			mask <- rbind.fill.matrix(mask, t(unlist(ls[[i]]$box)))
		}
		mask <- t(mask[-1,])
	}else{
		mask <- rep(1,InputAndHiddenDim)
	}
	return(mask)
}

GenerateBinaryWithProbP <- function(n, p){
	return(ifelse(runif(n, 0, 1) > p, 0, 1))
}

DropoutGetMask <- function(mask, HiddenLayer, LayerWithDropout, ProbabilityOfSamplingOne){
	counter <- 0
	for( j in LayerWithDropout ){
		counter <- counter + 1
		s <- sum(GenerateBinaryWithProbP(n = HiddenLayer[j] - 1, p = ProbabilityOfSamplingOne[counter])) + 1
		i <- sample(1 : HiddenLayer[j], size = s, replace = F)
		if(!is.null(dim(mask))){ 
			mask[c(i), j] <- 1
			index <- which(!(c(1 : HiddenLayer[j]) %in% c(i)))
			mask[c(index), j] <- 0 
		} else { 
			mask[i] <- 1 ; mask[-i] <- 0 
		}
	}
	return(mask)
}

ApplyMask <- function(ObjectiveMatrix, mask, ColIndex){
	if(!is.null(dim(mask))) { 
		index <- which(mask[,ColIndex] == 0)    # mask[,ColIndex] == 0 & !is.na(mask[,ColIndex])
		ObjectiveMatrix[, index] <- 0  
	} else {
		index <- which(mask == 0)
		ObjectiveMatrix[, mask == 0 ] <- 0 
	}
	return(ObjectiveMatrix)
}



#*************************** Loss Function *************************************

ComputeLoss <- function(Yhat, Y, LossFct, tau, generalized){
	N <- length(Y)
	if(LossFct == 'MSE'){
		return( 1/N * sum((Y - Yhat)^2) )
	} else if (LossFct == 'cross entropy'){
		return( -1/N * sum( Y * log(Yhat) + 
								( 1 - Y ) * log(1 - Yhat)) ) 
	} else if (LossFct == 'Exponential Cost'){
		return(ExponentialCost(tau, Yhat, Y) ) 
	} else if (LossFct == 'Hellinger Distance'){
		return(HellingerDist(Yhat, Y)) 
	} else if (LossFct == 'Kullback Leibler'){
		return(KullbackLeiblerDivergence(Yhat, Y, generalized)) 
	} else if (LossFct == 'Itakura Saito Distance'){
		return(ItakuraSaitoDistance(Yhat, Y)) 
	}
}

L2Reg <- function(Weights, reg, NumWeights){
	sum <-  0.5 * reg * sum(Weights[[NumWeights]] * Weights[[NumWeights]])
	return(sum)
}

ComputeDerCrossEntropy <- function(Yhat, Y){
	return(-1/length(Yhat) * ( Y / Yhat -  (1 - Y) / (1 - Yhat)))
}

# https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
ExponentialCost <- function(tau, Yhat, Y){
	return(tau * exp(1/tau * sum((Y - Yhat)^2) ) )
}

HellingerDist <- function(Yhat, Y){
	return(1/(length(Y)*sqrt(2)) * sum((sqrt(Y) - sqrt(Yhat))^2 ))
}

KullbackLeiblerDivergence <- function(Yhat, Y, generalized){
	if(generalized == F){
		return(1/length(Y) * sum(Y * log(Y/Yhat + 1e-14)))
	}else{
		return(1/length(Y) * (sum(Y * log(Y/Yhat + 1e-14)) - sum(Y) + sum(Yhat)) )
	}
}

ItakuraSaitoDistance <- function(Yhat, Y){
	return(1/length(Y) * sum((Y/Yhat) - log(Y/Yhat + 1e-14) -1 ))
}

ComputeDerExpCost <- function(tau, Yhat, Y){
	return( 2/tau * (Yhat - Y) * ExponentialCost(tau, Yhat, Y))
}

ComputeDerHellingerDist <- function(Yhat, Y){
	return( 1/length(Y)* (sqrt(Yhat) - sqrt(Y))/ (sqrt(2) * sqrt(Yhat) ) )
}

ComputeDerKLDiv <- function(Yhat, Y, generalized){
	if(generalized == F){
		return(1/length(Y) * (-Y/Yhat))
	}else{
		return(1/length(Y) * (-Y + Yhat)/Yhat)
	}
}

ComputeDerISDist <- function(Yhat, Y){
	return(1/length(Y) * (Yhat + Y)/Yhat^2)
}




#**************************** Backpropagation **********************************
# modify ifelse
ComputeDerivative <- function(ActivationFct, dscores = NULL, ObjectiveMatrix){
	if(ActivationFct == 'tanh'){
		return(tanh_output_to_derivative(ObjectiveMatrix))
	} else if(ActivationFct == 'logistic'){
		return(sigmoid_output_to_derivative(ObjectiveMatrix))
	} else if(ActivationFct == 'ReLU'){
		return(ifelse(ObjectiveMatrix >= 0, 1, 0))								# check!
	} else if(ActivationFct == 'Leaky ReLU'){
		return(ifelse(ObjectiveMatrix >= 0, 1, 0.1))          					 # check!
	} else if(ActivationFct == 'linear'){
		return(dscores)
	}
}

ComputeDerivativeOutpuLayer <- function(LossFct, OutputActivation, OutputLayer,
		Y, Weights, reg, tau, generalized){
	
	RegTermDerivative <- sum(Weights[[length(Weights)]])
	
	if(LossFct == 'MSE'){
		dscores <- -2/length(Y) * (Y - OutputLayer) + reg * RegTermDerivative
		
		if(OutputActivation != 'linear'){
			dscores <- ComputeDerivative(ActivationFct = OutputActivation, 
					dscores = dscores, ObjectiveMatrix = OutputLayer) * dscores
		} else {
			dscores <- dscores
		}
		
	} else if(LossFct == 'cross entropy'){ 
		dscores <- (ComputeDerCrossEntropy(Yhat = OutputLayer, Y) + 
					reg * RegTermDerivative) * 
				ComputeDerivative(ActivationFct = OutputActivation, 
						dscores = NULL, ObjectiveMatrix = OutputLayer)
		
	} else if(LossFct == 'Exponential Cost'){ 
		dscores <- (ComputeDerExpCost(tau, Yhat = OutputLayer, Y) + 
					reg * RegTermDerivative) * 
				ComputeDerivative(ActivationFct = OutputActivation, 
						dscores = NULL, ObjectiveMatrix = OutputLayer)
		
	} else if(LossFct == 'Hellinger Distance'){
		dscores <- (ComputeDerHellingerDist(Yhat = OutputLayer, Y) + 
					reg * RegTermDerivative) * 
				ComputeDerivative(ActivationFct = OutputActivation, 
						dscores = NULL, ObjectiveMatrix = OutputLayer)
		
	}else if(LossFct == 'Kullback Leibler'){
		dscores <- (ComputeDerivative(ActivationFct = OutputActivation, 
							dscores = NULL, ObjectiveMatrix = OutputLayer) + 
					reg * RegTermDerivative) * 
				ComputeDerKLDiv(Yhat = OutputLayer, Y, generalized = T)
		
	}else if(LossFct == 'Itakura Saito Distance'){
		dscores <- (ComputeDerISDist(Yhat = OutputLayer, Y) + 
					reg * RegTermDerivative) * 
				ComputeDerivative(ActivationFct = OutputActivation, 
						dscores = NULL, ObjectiveMatrix = OutputLayer)
		
	}
	return(dscores)
}


#*************************** Predict ********************************************
Predict <- function(Weights, data, LayerActivation, OutputActivation, 
		dropout, LayerWithDropout, ProbabilityOfSamplingOne)
{
	
	l <- length(Weights)
	predict.Wb <- cbind(data, rep(1, nrow(data)))
	counter <- 1
	
	for(k in 1 : l){
		
		if(k == LayerWithDropout[counter]){
			m <- ProbabilityOfSamplingOne[counter]
			if(counter < length(LayerWithDropout)){counter <- counter + 1}else{counter <- counter}
		} else {
			m <- 1
		}
		
		predict <- predict.Wb %*% Weights[[k]]
		if(k == l){
			predict <- activation(fct = OutputActivation, X = predict) 
		} else {
			predict <- activation(fct = LayerActivation, X = predict) * m
			predict.Wb <- cbind(predict, rep(1, nrow(predict)))
		}
	}
	return(predict)
}