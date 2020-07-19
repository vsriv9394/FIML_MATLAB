classdef NeuralNetwork
    
    properties
        
        nLayers
        nNodes
        nNodesTotal
        nVars
        nFeatures
        
        vars
        nodes
        d_vars
        d_nodes
        
        m
        v
        b1t
        b2t
    
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function NN = NeuralNetwork(nFeatures, nNodes)
            
            NN.nLayers     = size(nNodes,1)+2;
            NN.nNodes      = [nFeatures; nNodes; 1];
            NN.nNodesTotal = sum(NN.nNodes);
            NN.nVars       = sum((NN.nNodes(1:end-1)+1).*NN.nNodes(2:end));
            NN.nFeatures   = nFeatures;
            
            NN.vars    =  rand(NN.nVars, 1);
            NN.nodes   = zeros(NN.nNodesTotal, 1);
            NN.d_vars  = zeros(NN.nVars, 1);
            NN.d_nodes = zeros(NN.nNodesTotal, 1);
            
            NN.b1t = 1.0;
            NN.b2t = 1.0;
            NN.m   = zeros(NN.nVars, 1);
            NN.v   = zeros(NN.nVars, 1);
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function NN_new = load(NN)
            
            NN.vars = dlmread("NN_weights.txt");
            NN_new  = NN;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function save(NN)
            
            dlmwrite("NN_weights.txt", NN.vars, 'precision', 10);
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function NN_new = runOptimizer(NN, sens)
            
            NN.b1t = NN.b1t * 0.9;
            NN.b2t = NN.b2t * 0.999;
            NN.m = 0.100*sens       + 0.900*NN.m;
            NN.v = 0.001*sens.*sens + 0.999*NN.v;
            NN.vars = NN.vars - 0.001 * (NN.m/(1-NN.b1t)) ./ (1e-10 + sqrt(NN.v/(1-NN.b2t)));
            
            NN_new = NN;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function sens = getSens(NN, features, dJdbeta)
            
            if size(features,2)~=NN.nFeatures
                fprintf("Inconsistent number of inputs provided to predict\n");
            end
            
            sens = zeros(NN.nVars,1);
            
            for iData = 1:size(features,1)
                
                NN = NN.forwardPropagate(features(iData,:));
                NN.d_nodes(end) = dJdbeta(iData);
                NN = NN.backwardPropagate();
                sens = sens + NN.d_vars;
                    
            end
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function beta_pred = predict(NN, features)
            
            if size(features,2)~=NN.nFeatures
                fprintf("Inconsistent number of inputs provided to predict\n");
            end
            
            beta_pred = zeros(size(features,1),1);
            
            for iData = 1:size(features,1)
                
                NN = NN.forwardPropagate(features(iData,:));
                beta_pred(iData) = NN.nodes(end);
                    
            end
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function NN_new = train(NN, features, beta, nEpochs, nBatches, trainingFraction, verbose)
            
            if size(features,2)~=NN.nFeatures
                fprintf("Inconsistent number of inputs provided to train\n");
            end
            
            indices = randperm(size(features,1));
            features = features(indices, :);
            beta     = beta(indices,:);
            
            nTrain      = fix(trainingFraction*size(features,1));
            nValid      = size(features,1)-nTrain;
            batchSize   = fix(nTrain / nBatches);
            batchExcess = mod(nTrain, nBatches);
            
            ctr = batchSize + min(batchExcess,1);
            sens = zeros(NN.nVars, 1);
            
            for iEpoch = 1:nEpochs
            
                trainLoss = 0.0;
                validLoss = 0.0;
            
                for iData = 1:nTrain
                
                    NN = NN.forwardPropagate(features(iData,:));
                    trainLoss = trainLoss + (NN.nodes(end)-beta(iData))^2;
                    NN.d_nodes(end) = 2*(NN.nodes(end) - beta(iData));
                    NN = NN.backwardPropagate();
                    sens = sens + NN.d_vars;
                    
                    ctr = ctr - 1;
                
                    if(ctr==0)
                        NN = NN.runOptimizer(sens);
                        sens(1:end) = 0.0;
                        ctr = batchSize;
                        if batchExcess > 0
                            ctr = ctr + 1;
                            batchExcess = batchExcess - 1;
                        end
                    end
                
                end
                
                for iData = nTrain+1:nTrain+nValid
                
                    NN = NN.forwardPropagate(features(iData,:));
                    validLoss = validLoss + (NN.nodes(end)-beta(iData))^2;
                    
                end
                
                if(verbose>0)
                    fprintf('Losses in epoch %06d:   Training %+.10e   Validation %+.10e', iEpoch, trainLoss/nTrain, validLoss/nValid);
                    fprintf('\n');
                end
            
            end
            
            NN_new = NN;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function NN_new = backwardPropagate(NN)
            
            cEnd = NN.nNodesTotal;
            wEnd = NN.nVars;
            
            for iLayer = NN.nLayers:-1:2
                
                cBeg = cEnd - NN.nNodes(iLayer) + 1;
                pEnd = cBeg - 1;
                pBeg = pEnd - NN.nNodes(iLayer-1) + 1;
                wBeg = wEnd - NN.nNodes(iLayer)*NN.nNodes(iLayer-1) + 1;
                bEnd = wBeg - 1;
                bBeg = bEnd - NN.nNodes(iLayer) + 1;
                
                NN.d_nodes(cBeg:cEnd) = NeuralNetwork.d_activate(NN.nodes(cBeg:cEnd), NN.d_nodes(cBeg:cEnd));
                
                NN.d_vars(bBeg:bEnd)  = NN.d_nodes(cBeg:cEnd);
                NN.d_vars(wBeg:wEnd)  = reshape(NN.d_nodes(cBeg:cEnd)*NN.nodes(pBeg:pEnd)', NN.nNodes(iLayer)*NN.nNodes(iLayer-1), 1);
                NN.d_nodes(pBeg:pEnd) = reshape(NN.vars(wBeg:wEnd), NN.nNodes(iLayer), NN.nNodes(iLayer-1))'*NN.d_nodes(cBeg:cEnd);
                
                if iLayer==NN.nLayers
                    NN.d_vars(bBeg:bEnd) = 0.0;
                end
                
                cEnd = pEnd;
                wEnd = bBeg - 1;
                
            end
            
            NN_new = NN;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function NN_new = forwardPropagate(NN, features)
            
            NN.nodes(1:NN.nFeatures) = features(1:NN.nFeatures);
            
            pBeg = 1;
            bBeg = 1;
            
            for iLayer = 2:NN.nLayers
                
                pEnd = pBeg + NN.nNodes(iLayer-1) - 1;
                cBeg = pEnd + 1;
                cEnd = cBeg + NN.nNodes(iLayer) - 1;
                bEnd = bBeg + NN.nNodes(iLayer) - 1;
                wBeg = bEnd + 1;
                wEnd = wBeg + NN.nNodes(iLayer)*NN.nNodes(iLayer-1) - 1;
                
                NN.nodes(cBeg:cEnd) = reshape(NN.vars(wBeg:wEnd), NN.nNodes(iLayer), NN.nNodes(iLayer-1))*NN.nodes(pBeg:pEnd) + NN.vars(bBeg:bEnd);
                                  
                NN.nodes(cBeg:cEnd) = NeuralNetwork.activate(NN.nodes(cBeg:cEnd));
                
                pBeg = cBeg;
                bBeg = wEnd + 1;
            
            end
            
            NN_new = NN;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    methods (Static)
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function act_nodes = activate(nodes_val)
            
            %act_nodes = tanh(nodes_val);
            act_nodes = nodes_val;
            act_nodes(nodes_val<=0.0) = 0.0;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        function d_act_nodes = d_activate(nodes_val, d_nodes_val)
            
            %d_act_nodes = d_nodes_val ./ cosh(nodes_val) ./ cosh(nodes_val);
            d_act_nodes = d_nodes_val;
            d_act_nodes(nodes_val<=0.0) = 0.0;
            
        end
        
        %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    end
    
end