######################################################################
#
#							Denoised Autoencoder
#
######################################################################

DenoisedAutoencoder <- function(InputMatrix, NumNeurons, batchsize, 
                                ActivationFct, 
                                LossFct, 
                                generalized,
                                # NoiseDistribution = 'norm' or 'unif'
                                NoiseDistribution,
                                OutputActivationAE,
                                RegularizationParam, LearningRateParam, 
                                maxit, abstol,
                                OptimizingWeights,
                                SLR)
{
  # create noise and add it to InputMatrix
  RandomSamplingDist <- paste0('r', NoiseDistribution)
  
  if(RandomSamplingDist != 'runif' & RandomSamplingDist != 'runif'){
    cat('WARNING: no noise will be added to the input matrix')
    noise <- as.matrix(0, ncol = ncol(InputMatrix), nrow = nrow(InputMatrix))
    
  } else if(RandomSamplingDist == 'runif'){
    ArgList = list(
      n = ncol(InputMatrix) * nrow(InputMatrix),
      min = -1/10,
      max = 1/10
    )
  } else if(RandomSamplingDist == 'rnorm'){
    ArgList = list(
      n = ncol(InputMatrix) * nrow(InputMatrix),
      mean = 0,
      sd = 0.1
    )
  }
  
  noise <- matrix(do.call(what = RandomSamplingDist, args = ArgList), 
                  ncol = ncol(InputMatrix), nrow = nrow(InputMatrix))
  
  X <- InputMatrix + noise
  Y <- InputMatrix
  
  # create and init weights and bias
  Params <- list()
  Params$W[[1]] <- matrix( runif( ncol(X) * NumNeurons, -1/sqrt(ncol(X)), 1/sqrt(ncol(X)) ), 
                           nrow = ncol(X), ncol = NumNeurons )          # -1/sqrt(h[i]), 1/sqrt(h[i]) OR -sqrt(6/sum(h[i]+h[i+1])), sqrt(6/sum(h[i]+h[i+1]))
  Params$W[[2]] <- t(Params$W[[1]])
  
  # Invert positions to avoid listing problems
  Params$b[[2]] <- matrix( runif( ncol(X), -1/sqrt(NumNeurons), 1/sqrt(NumNeurons) ), 
                           nrow = 1, ncol = ncol(X) )                                              
  Params$b[[1]] <- matrix( runif(NumNeurons, -1/sqrt(ncol(X)), 1/sqrt(ncol(X))), 
                           nrow = 1, ncol = NumNeurons)                 # 0 OR -1/sqrt(h[i]), 1/sqrt(h[i])
  
  # Create Parameters for Stochastic Gradient Descent
  SLRParam <- initSLRParam(SLR, NumWeights = 2, 
                           NetArchitecture = c(ncol(X), 
                                               NumNeurons, ncol(X)))
  
  L <- NULL
  loss <- 1
  i <- 0
  
  while( i < maxit & loss > abstol ){
    i <- i + 1
    
    # Create new mini-batch
    if(!is.null(batchsize)){
      MiniBatch <- SampleMiniBatch(X, Y, batchsize)
      TrainBatch <- MiniBatch$TrainM
      TestBatch <- MiniBatch$TestM
    } else {
      TrainBatch <- X
      TestBatch <- Y
    }
    
    # Forward-Propagation
    ForwardPropagationList <- ForwardPhaseAE(X = TrainBatch, Params, 
                                             ActivationFct, OutputActivationAE)
    # Compute Loss Fct
    DataLoss <- ComputeLoss(Yhat = ForwardPropagationList$HiddenLayer[[2]], 
                            Y = TestBatch, LossFct, tau, generalized)
    
    loss <- DataLoss + L2Reg(Weights = rbind(Params$W[[2]], Params$b[[2]]),
                             reg = RegularizationParam, NumWeights = 2)
    L[i] <- loss
    
    # Cat info
    cat('Weights set actually optimizing:' ,OptimizingWeights, 
        '\t', 'Iteration:', i, '\t', 'Loss:', loss, '\n')
    
    # Backprop
    BackPropagationList <- BackwardPhaseAE(ForwardPropagationList, LossFct, 
                                           ActivationFct = ActivationFct,
                                           OutputActivation = OutputActivationAE, 
                                           OutputLayer = ForwardPropagationList$HiddenLayer[[2]],
                                           Y = TestBatch, Weights = Params, reg = RegularizationParam, tau, generalized)
    
    # Weights Update
    WeightsAugList <- list( 
      rbind(Params$W[[1]], Params$b[[1]] ), 
      rbind(Params$W[[2]], Params$b[[2]] ) 
    )

    
        
    WeightsUpdateList <- WeightsUpdate(Weights = WeightsAugList, method = SLR, 
                                       SLRParam = SLRParam, 
                                       BackpropDerivative = BackPropagationList,  
                                       t = i, eta = LearningRateParam, 
                                       LearningRate = LearningRateParam)
    
    # Rewrite
#    if(!is.null(WeightsUpdateList$WeightsUpdated[[1]])){
      
      Params$W[[1]] <- matrix(WeightsUpdateList$WeightsUpdated[[1]][-nrow(WeightsUpdateList$WeightsUpdated[[1]]),], 
                              nrow = nrow(WeightsUpdateList$WeightsUpdated[[1]]) - 1, 
                              ncol = ncol(WeightsUpdateList$WeightsUpdated[[1]])
                              )
      Params$b[[1]] <- matrix(WeightsUpdateList$WeightsUpdated[[1]][nrow(WeightsUpdateList$WeightsUpdated[[1]]),],
                              nrow = 1 , 
                              ncol = ncol(WeightsUpdateList$WeightsUpdated[[1]])
                              )
      
      Params$W[[2]] <- matrix(WeightsUpdateList$WeightsUpdated[[2]][-nrow(WeightsUpdateList$WeightsUpdated[[2]]),], 
                              nrow = nrow(WeightsUpdateList$WeightsUpdated[[2]]) - 1,
                              ncol = ncol(WeightsUpdateList$WeightsUpdated[[2]])
                              )
      Params$b[[2]] <- matrix(WeightsUpdateList$WeightsUpdated[[2]][nrow(WeightsUpdateList$WeightsUpdated[[2]]),], 
                              nrow = 1,
                              ncol = ncol(WeightsUpdateList$WeightsUpdated[[2]])
                              )
    
      
    # } else {
    #   Params$W[[1]] <- WeightsUpdateList$WeightsUpdated[[1]][-nrow(WeightsUpdateList$WeightsUpdated[[1]])]
    #   Params$b[[1]] <- WeightsUpdateList$WeightsUpdated[[1]][nrow(WeightsUpdateList$WeightsUpdated[[1]])]
    #   
    #   Params$W[[2]] <- WeightsUpdateList$WeightsUpdated[[2]][-nrow(WeightsUpdateList$WeightsUpdated[[2]])]
    #   Params$b[[2]] <- WeightsUpdateList$WeightsUpdated[[2]][nrow(WeightsUpdateList$WeightsUpdated[[2]])]
    # }
    
    if(!is.null(SLR)){
      SLRParam <- WeightsUpdateList$Parameters
    }
    
    model <- list(
      Params = Params,
      L = L
    )
  }
  return(model)
}




