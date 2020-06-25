function NN = FIML(nFIMLIters, stepSize, nHiddenLayerNodes, solver)

    % SOLVE FOR BASELINE AND TRAIN THE NEURAL NETWORK FOR BASELINE STATES ::::::::::::::::::::::::::::::::::::::::::::::
    
    fprintf("Beginning FIML\n");
    [~, ~, features, beta] = solver([]);
    NN = NeuralNetwork(size(features,2), nHiddenLayerNodes);
    
    fprintf("Training NN for baseline...\n");
    NN.train(features, beta, 100, 20, 0.8, 0);
    
    [obj, dJdbeta, ~, ~] = solver(NN);
    sens = NN.getSens(features, dJdbeta);
    
    fprintf("FIML Iteration 000000   Objective %+.10e", obj);
    fprintf('\n');
    objectives = zeros(nFIMLIters, 1);
    
    
    for iter=1:nFIMLIters
        
        NN.vars = NN.vars - stepSize * sens / max(abs(sens));
        [objectives(iter), dJdbeta, ~, ~] = solver(NN);
        sens = NN.getSens(features, dJdbeta);
        
        fprintf("FIML Iteration %06d   Objective %+.10e", iter, objectives(iter));
        fprintf('\n');
        
    end

end