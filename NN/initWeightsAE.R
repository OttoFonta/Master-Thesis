# Weights initialization for DenoisedAutoencoder.R

initWeightsAE <- function(X, hidden, NumWeights,
                          ActivationFct, OutputActivation, LossFct,
                          generalized,
                          batchsize, NoiseDistribution, OutputActivationAE,
                          RegularizationParam, LearningRateParam, maxit, abstol,
                          SLR){
  
  InputMatrix <- X
  cat('Number of Hidden Layer: ', NumWeights, '\n', 
      'Number of Neurons: ', hidden, '\n')
  
  DAE <- list() ; Param <- list()
  
  for (i in 1 : NumWeights){
    
    cat('Starting training of the ', i, 
        if(i == 1){'st'}
        else if(i == 2){'nd'}
        else if(i == 3) {'rd'}
        else {'th'}, 'layer.', '\n'
    )
    
    if(i != NumWeights){		
      DAE[[i]] <- DenoisedAutoencoder(InputMatrix, NumNeurons = hidden[i], 
                                      batchsize, 
									  ActivationFct = ActivationFct,
                                      LossFct, generalized,
                                      # NoiseDistribution = 'norm' or 'unif'
                                      NoiseDistribution, 
                                      OutputActivationAE,
                                      RegularizationParam, LearningRateParam, 
                                      maxit, abstol,
                                      OptimizingWeights = i,
                                      SLR)
      
      InputMatrix <- GetHiddenFeaturesAE(X = InputMatrix, Params = DAE[[i]]$Params, 
                                         ActivationFct = ActivationFct)
    }else{
      DAE[[i]] <- DenoisedAutoencoder(InputMatrix, NumNeurons = hidden[i], 
                                      batchsize, 
                                      ActivationFct = OutputActivation, 
                                      LossFct, generalized,
                                      # NoiseDistribution = 'norm' or 'unif'
                                      NoiseDistribution, 
                                      OutputActivationAE,
                                      RegularizationParam, LearningRateParam, 
                                      maxit, abstol,
                                      OptimizingWeights = i,
                                      SLR)
      
      InputMatrix <- GetHiddenFeaturesAE(X = InputMatrix, Params = DAE[[i]]$Params, 
                                         ActivationFct = OutputActivation)
    }
    
    Param[[i]] <- list(W = DAE[[i]]$Params$W[[1]], b = DAE[[i]]$Params$b[[1]])
#    Param$WeightsAug[[i]] <- list(rbind(DAE[[i]]$Params$W[[1]], DAE[[i]]$Params$b[[1]]))
    
    cat('Finished training.', '\n', 
        if(DAE[[i]]$L[length(DAE[[i]]$L)] < abstol){
          cat('Tolerance reached:', '\n', 'loss = ', 
              DAE[[i]]$L[length(DAE[[i]]$L)], '\n')
        }
        else{ 
          cat('Tolerance level not reached:', '\n', 'loss = ', 
              DAE[[i]]$L[length(DAE[[i]]$L)], '\n')
        }
    ) 
    
    
  }
  return(Param)
}





