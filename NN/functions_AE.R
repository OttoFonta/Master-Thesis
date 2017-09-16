# Set of functions called by DenoisedAutoencoder.R


ForwardPhaseAE <- function(X, Params, ActivationFct, OutputActivationAE){
	
	ForwardList <- list(InputLayer = X)
	
	# Layer 1
	ForwardList$Neuron[[1]] <- sweep(ForwardList$InputLayer %*% Params$W[[1]], 
			2, Params$b[[1]], '+') 
	ForwardList$HiddenLayer[[1]] <- activation(fct = ActivationFct, X = ForwardList$Neuron[[1]])
	
	# Layer 2 (output)
	ForwardList$Neuron[[2]] <- sweep(ForwardList$HiddenLayer[[1]] %*% Params$W[[2]],
			2, Params$b[[2]], '+') 
	ForwardList$HiddenLayer[[2]] <- activation(fct = OutputActivationAE, X = ForwardList$Neuron[[2]])
	
	return(ForwardList)
}




BackwardPhaseAE <- function(ForwardPropagationList, LossFct,
		ActivationFct,
		OutputActivation, OutputLayer,
		Y, Weights, reg, tau, generalized){
	
	OutputLayerDerivative <- ComputeDerivativeOutpuLayer(LossFct, 
			OutputActivation, 
			OutputLayer, Y, 
			Weights = rbind(Weights$W[[2]], 
					Weights$b[[2]]), 
			reg, tau, generalized)
	
	ParamsDerivative <- list()
	
	dW2 <- crossprod(ForwardPropagationList$HiddenLayer[[1]], OutputLayerDerivative)
	db2 <- colSums(OutputLayerDerivative)
	
	dh <- tcrossprod(OutputLayerDerivative, Weights$W[[2]])
	dh <- ComputeDerivative(ObjectiveMatrix = ForwardPropagationList$Neuron[[1]], 
			ActivationFct = ActivationFct) * dh
	
	dW1 <- crossprod(ForwardPropagationList$InputLayer, dh)
	db1 <- colSums(dh)
	
	ParamsDerivative[[1]] <- list(WeightsDerivative = rbind(dW1, db1) )
	ParamsDerivative[[2]] <- list(WeightsDerivative = rbind(dW2, db2) )
	
	return(ParamsDerivative)
}



GetHiddenFeaturesAE <- function(X, Params, ActivationFct){
	X <- sweep(X %*% Params$W[[1]], 2, Params$b[[1]], '+')
	return(activation(fct = ActivationFct, X))
}