##################################################################################
#
#								Backpropagation
#
##################################################################################

Backpropagation <- function(HiddenLayer, LossFct, OutputActivation, OutputLayer, Y, 
                            Weights, reg, tau, generalized, ActivationFct, mask){
  
  l <- length(Weights)
  NumLayer <- l-1
  
  BackpropDerivative <- list()
  BackpropDerivative[[l+1]] <- list(LayerDerivative = ComputeDerivativeOutpuLayer(LossFct, OutputActivation, 
                                                                                  OutputLayer, Y, Weights, reg,
																				  tau, generalized)
																  )
  counter <- 0
  
  for(k in l : 1){
    index <- k + 1
    
    dW <- crossprod(HiddenLayer[[k]]$activation.fct[,-ncol(HiddenLayer[[k]]$activation.fct)], 
                    BackpropDerivative[[index]]$LayerDerivative)
    db <- colSums(matrix(BackpropDerivative[[index]]$LayerDerivative))
    
    if(k != 1){
      # Check Weights dim (matrix, vec)
      if(!is.null(dim(Weights[[k]]))){
        W <- Weights[[k]][-nrow(Weights[[k]]), ]
      }else{
        W <- Weights[[k]][-length(Weights[[k]])]
      }
      
      dh <- tcrossprod(BackpropDerivative[[index]]$LayerDerivative,
                       W)
      dh <- ComputeDerivative(ObjectiveMatrix = HiddenLayer[[k]]$activation.fct[,-ncol(HiddenLayer[[k]]$activation.fct)], 
			  ActivationFct = ActivationFct) * dh
      
      # Dorpout
      if(NumLayer > 1){
        indexMask <- ncol(mask) - counter
        dh <- ApplyMask(ObjectiveMatrix = dh, mask = mask, ColIndex = indexMask )
      } else {
        indexMask <- length(mask) - counter
        dh <- ApplyMask(ObjectiveMatrix = dh, mask = mask, ColIndex = indexMask )
      }
      
      BackpropDerivative[[k]] <- list(WeightsDerivative = rbind(dW, db) ,
                                      LayerDerivative = dh)
      
    } else {
      BackpropDerivative[[k]] <- list(WeightsDerivative = rbind(dW, db))
    }
    counter <- counter + 1
  }
  return(BackpropDerivative)	
}
