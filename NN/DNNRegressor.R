# TODO: Add comment
# 
# Author: OF
###############################################################################


DNNRegressor <- function(x, y, omodel = NULL, traindata = data,  
                         testdata = NULL, hidden, maxit = 2000, abstol = 1e-2, 
						 batchsize = NULL,
                         LayerWithDropout, ProbabilityOfSamplingOne,
                         lr = 1e-2, reg = 1e-3, display = 100, 
                         dropout = F, alpha = 0.5, random.seed = NULL,
                         act.fct = c('logistic','tanh','ReLU','Leaky ReLU'), 
                         OutputActivation = c('logistic','tanh','ReLU','Leaky ReLU', 'linear'),
                         LossFct = c('MSE','cross entropy'), generalized = T, tau = NULL,
                         SLR = c('Adam','Momentum','NAG', NULL), 	 
                         inputAutoencoder,
                         scale.regime = scale.regime )
{
  require(sigmoid) ; require(plyr) ; require(autoencoder)
  
  ModelParameters <- ModelParameterCollector(
		  x, y, omodel, traindata,
		  testdata, hidden, maxit, abstol, batchsize,
		  LayerWithDropout, ProbabilityOfSamplingOne,
		  lr, reg, display, 
		  dropout, alpha, random.seed,
		  act.fct, 
		  OutputActivation,
		  LossFct, generalized, tau,
		  SLR,
		  inputAutoencoder,
		  scale.regime
		  )
  
  if(inputAutoencoder$pre.training.ae == TRUE & !(act.fct %in% c('logistic','tanh'))){
    stop(cat('Error: the selected activation function is not implemented in the autoencoder', '\n'))
  }
  
  if(scale.regime == 'scaled' & OutputActivation == 'linear' & LossFct == 'cross entropy'){
    stop(cat('Error: incoherent inputs - scaling regime (scaled) / linear output == TRUE', '\n'))
  }
  
  if(scale.regime == 'standardized' & OutputActivation %in% c('tanh','logistic')){
    stop(cat('Error: incoherent inputs:\nscaling regime (std);\noutput act fct = tanh or logistic.\nSet of hyperparameters not allowed.','\n'))
  }
  
  if(scale.regime == 'standardized' & LossFct == 'cross entropy'){
    stop(cat('Error: incoherent inputs:\nscaling regime (std);\nloss fct = cross entropy.\nSet of hyperparameters not allowed.','\n'))
  }
  
  set.seed(random.seed)
  X <- unname(data.matrix(traindata[,x]))
  Y <- traindata[,y]
  L <- NULL
  if(is.null(batchsize)) { N <- nrow(traindata) } else { N <- batchsize} 
  loss <- 10
  Y.hat <- NULL
  validation.error <- NULL
  optim.error <- NULL
  gen.error <- NULL
  gen.error[1] <- 0.0001
  weightsTrack <- NULL
  
  NumHiddenLayer <- length(hidden) # H, NetDepth
  NetArchitecture <- c(ncol(X), hidden, 1) # h
  NetDepth <- length(NetArchitecture)  # = l+1
  NumWeights <- length(NetArchitecture) - 1 # l
  
  # create and init weights and bias
  Weights <- initWeightsStandard(omodel, h = NetArchitecture, inputAutoencoder,
                                 l =  NumWeights, X = X, ActivationFct = act.fct,
                                 hidden = NetArchitecture[-tail(NetArchitecture, 1)], 
								 NumWeights, OutputActivation)$WeightsAug
  
  # create and init LR
  learning.rate <- initLrStandard(lr, h = NetArchitecture, l = NumWeights)
  
  # create and init parameters for SLR methods
  SLRParam <- initSLRParam(SLR, NumWeights, NetArchitecture)
  
  # create and init mask
  mask <- DropoutInitializer(InputAndHiddenDim = c(ncol(X), hidden))
  
  i <- 0
  
  # let's loop!!
  while( i < maxit && loss > abstol && gen.error[i+1] < alpha ) { 
    
    i <- i + 1
    
    # Create new mask
    if( dropout == TRUE ){
      mask <- DropoutGetMask(mask, HiddenLayer = NetArchitecture, 
                             LayerWithDropout, ProbabilityOfSamplingOne)
    }
    
    # Create new mini-batch
    if(!is.null(batchsize)){
      MiniBatch <- SampleMiniBatch(X, Y, batchsize)
      TrainBatch <- MiniBatch$TrainM
      TestBatch <- MiniBatch$TestM
    } else {
      TrainBatch <- X
      TestBatch <- Y
    }
    
    # Forward propagation
    HiddenLayer <- FeedForward(X = TrainBatch, NumWeights, weightsAug = Weights, mask, N, 
                               OutputActivation, LayerActivation = act.fct)
    
    # Check get.loss.fct and L2.regularization
    if( OutputActivation != 'linear' ){
      temp <- HiddenLayer[[NetDepth]]$activation.fct
      data.loss <- ComputeLoss(Yhat = temp, Y = TestBatch, LossFct, 
                               tau, generalized)
    } else {
      temp <- HiddenLayer[[NetDepth]]$layer
      data.loss <- ComputeLoss(Yhat = temp, Y = TestBatch, LossFct,
                               tau, generalized)
    }
    
    # Chech (probably wrong)(or maybe not!...)
    loss <- data.loss + L2Reg(Weights, reg, NumWeights)
    L[i] <- loss
    
    # display results and update model
    if(!is.null(testdata)) {
      Y.hat <- Predict(Weights, data = testdata[,x], LayerActivation = act.fct, 
                       OutputActivation, dropout, LayerWithDropout, ProbabilityOfSamplingOne)
      
      # Validation error
      validation.error[i] <- ComputeLoss(Y.hat, Y = testdata[,y], LossFct, 
                                         tau, generalized)
      
      # Optim error
      if(i == 1){  
        optim.error[i] <- validation.error[i] + 0.0001
      }
      optim.error[i+1] <- min(optim.error[i],validation.error[i])
      if(optim.error[i+1] < optim.error[i]){
        weightsTrack <- Weights
      }
      
      # Gen error
      gen.error[i+1] <- 100 * (validation.error[i] / optim.error[i+1] - 1)
      if( i %% display == 0) { cat(i, L[i], validation.error[i], optim.error[i+1], gen.error[i+1], "\n") }
    } else {
      alpha <- Inf ; gen.error[i+1] <- gen.error[1] 	
      if( i %% display == 0) { cat(i, L[i], "\n") }
    }
    
    model <- list(
      hidden = hidden,
      Weights = Weights,
      weightsTrack = weightsTrack,
      L = L,
      validation.error = validation.error,
      optim.error = optim.error,
      gen.error = gen.error
    )
    
    # backward propagation
    BackpropDerivativeList <- Backpropagation(HiddenLayer, LossFct, 
                                              OutputActivation, OutputLayer = HiddenLayer[[NetDepth]]$activation.fct, 
                                              Y = TestBatch, Weights, reg, tau, generalized, ActivationFct = act.fct,
											  mask = mask)
    
    # update parameters
    WeightsUpdateList <- WeightsUpdate(Weights, method = SLR, 
                                       SLRParam, BackpropDerivative = BackpropDerivativeList, 
                                       t = i, eta = lr, LearningRate = lr)
    Weights <- WeightsUpdateList$WeightsUpdated
    if(!is.null(SLR)){
      SLRParam <- WeightsUpdateList$Parameters
    }
  }
  
  model <- list(
    hidden = hidden,
    Weights = Weights,
    weightsTrack = weightsTrack,
    L = L,
    validation.error = validation.error,
    optim.error = optim.error,
    gen.error = gen.error,
	ModelParameters = ModelParameters
  )
  
  
  return(model)
} 








