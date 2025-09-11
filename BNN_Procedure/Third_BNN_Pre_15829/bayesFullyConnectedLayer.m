classdef bayesFullyConnectedLayer < nnet.layer.Layer ...
        & nnet.layer.Acceleratable ...
        & nnet.layer.Formattable
    % bayesFullConnectedLayer   Bayesian fully connected layer

    properties
        OutputSize
        InputSize
    end

    properties(Learnable)
        % Layer learnable parameters

        % Represent the weights and biases in a Bayesian neural network
        % by a probability distribution over a possible set
        % of weight values.
        % Use MeanWeights, RhoWeights, MeanBias, and RhoBias
        % to represent the probability distributions.
        MeanWeights
        RhoWeights
        MeanBias
        RhoBias

        % Sigma1 and Sigma2 represent the parameters of the prior
        % probabilities of the weight distributions. Sigma1 and Sigma2
        % represent the variances of the mixture of Gaussian distributions
        % that define the prior distribution.
        % These are learnable parameters that depend on the
        % problem/dataset and on the network initialization.
        Sigma1
        Sigma2
    end

    properties (State)
        % Use LogPosterior and LogPrior to approximate the
        % true Bayesian posterior on the weights.
        LogPosterior
        LogPrior
    end

    methods
        function this = bayesFullyConnectedLayer(outputSize,nvargs)
            % layer = bayesFullyConnectedLayer(outputSize) creates a Bayes
            % fully connected layer. outputSize specifies the size of the
            % output for the layer.
            %
            % layer = bayesFullyConnectedLayer(numChannels,Sigma1=2,Sigma2=1)
            % also specifies the variances of the prior distribution.
            %
            % layer = bayesFullyConnectedLayer(numChannels,Name=name) also
            % specifies the layer name.

            arguments
                outputSize
                nvargs.Name = ""
                nvargs.Sigma1 = 1.0
                nvargs.Sigma2 = 0.5
            end

            % Set output size.
            this.OutputSize = outputSize;

            % Set layer name.
            this.Name = nvargs.Name;

            % Set prior parameters.
            this.Sigma1 = nvargs.Sigma1;
            this.Sigma2 = nvargs.Sigma2;

            % Set number of layer inputs and outputs.
            this.NumInputs = 1;
            this.NumOutputs = 1;

            % Set initial values for log posterior and prior.
            this.LogPosterior = 0;
            this.LogPrior = 0;
        end

        function this = initialize(this, inputDataLayout)
            outputSize = this.OutputSize;

            sdims = finddim(inputDataLayout, 'S');
            cdim = finddim(inputDataLayout, 'C');
            inputSize = inputDataLayout.Size( [sdims cdim] );
            this.InputSize = inputSize;

            if isempty(this.MeanWeights)
%                 this.MeanWeights = initializeGlorot([outputSize inputSize], ...
                  this.MeanWeights = gpuArray(initializeGlorot([outputSize inputSize], ...
                    outputSize, prod(inputSize)));
            end

            if isempty(this.RhoWeights)
%                 this.RhoWeights = -2 + 1 .* rand([outputSize inputSize]);
                  this.RhoWeights = gpuArray(-2 + 1 .* rand([outputSize inputSize], 'single'));  % 加 'single'
            end

            if isempty(this.MeanBias)
%                 this.MeanBias = zeros(outputSize, 1);
                  this.MeanBias = gpuArray(zeros(outputSize, 1, 'single'));
            end

            if isempty(this.RhoBias)
%                 this.RhoBias = -2 .* ones(outputSize, 1);
                  this.RhoBias = gpuArray(-2 .* ones(outputSize, 1, 'single'));
            end
        end

        function [Y, logPosterior, logPrior] = predict(this, X)
            [W, b] = this.samplePosterior();
            Y = fullyconnect(X, W, b);

            logPosterior = [];
            logPrior = [];
        end

        function [Y, logPosterior, logPrior] = forward(this, X)
            [W, b] = this.samplePosterior();
            Y = fullyconnect(X, W, b);

            logPosterior = this.estimateLogPosterior(W,b);
            logPrior = this.computeLogPrior(W,b);
        end
    end

    methods(Access = private)
        function logPrior = computeLogPrior(this, sampledWeights, sampledBias)
            % This function takes as input the sampled weights and biases and returns
            % the log prior.
            %
            % The paper 'Weight Uncertainty in Neural Networks', section 3.3 defines the
            % prior as a scale mixture of gaussians with mean 0 and
            % variance sigma1 and sigma2 where
            % * sigma1 is greater than sigma2
            % * sigma2 must be less than 1.
            mixturePriorWeights = logScaleMixturePrior(sampledWeights, this.Sigma1, this.Sigma2) + eps;
            mixturePriorBias = logScaleMixturePrior(sampledBias, this.Sigma1, this.Sigma2) + eps;

            logMixturePrior = mixturePriorWeights + mixturePriorBias;
            logPrior = sum(logMixturePrior, 'all');
        end

        function logPosterior = estimateLogPosterior(this, sampledWeights, sampledBias)
            % This function takes as input the sampled weights and biases
            % and returns the log posterior.
            %
            % The posterior is the probability of having the weights and
            % bias given the dataset D.
            % This is nontrivial to compute, therefore, the value is
            % estimated.

            weightsSigma = iSigma(this.RhoWeights);
            logProbPosteriorWeights = sum(logProbabilityNormal(sampledWeights, this.MeanWeights, weightsSigma), 'all');

            biasSigma = iSigma(this.RhoBias);
            logProbPosteriorBias = sum(logProbabilityNormal(sampledBias, this.MeanBias, biasSigma));

            logPosterior = logProbPosteriorWeights + logProbPosteriorBias;
        end

        function [w,b] = samplePosterior(this)
            % This function generates sample from the posterior
            % distributions of the learnables.
            %
            % During training, the algorithm uses the posterior weights.
            % The posterior weights are conditioned on the MeanWeights and
            % MeanBias given by the formula
            % w = mu + log(1 + exp(rho)) * epsilon.

            epsilonW = randn([this.OutputSize this.InputSize], "like", this.MeanWeights);
            epsilonB = randn([this.OutputSize 1], "like", this.MeanBias);

            w = this.MeanWeights + log(1 + exp(this.RhoWeights)) .* epsilonW;
            b = this.MeanBias + log(1 + exp(this.RhoBias)) .* epsilonB;
        end
    end
end

function sample = logScaleMixturePrior(x, sigma1, sigma2)
% This function produces a mixture distribution of Gaussian distributions
% with mixing proportion set by piScalar. Each distribution has mean 0 and
% variance defined by sigma1 and sigma2.
piScaler = 0.5;
p1 = piScaler .* logProbabilityNormal(x, 0, sigma1);
p2 = (1 - piScaler) .* logProbabilityNormal(x, 0, sigma2);
sample = p1 + p2;
end

function out = iSigma(rho)
out = log(1+exp(rho));
end


