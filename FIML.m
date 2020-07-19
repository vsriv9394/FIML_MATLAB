function NN = FIML(nTrainIters, nFIMLIters, stepSize, nHiddenLayerNodes,...
                   solver_dict, solver_weights, fd_step)

    % SOLVE FOR BASELINE AND TRAIN THE NEURAL NETWORK FOR BASELINE STATES
    
    fprintf("Beginning FIML\n");
    features_vec = [];
    beta_vec = [];
    NN_dummy = [];
    
    for i=1:size(solver_dict)
        [~, ~, features, beta] = solver_dict{i}(NN_dummy);
        features_vec = [features_vec; features];
        beta_vec = [beta_vec; beta];
    end
    
    NN = NeuralNetwork(size(features_vec,2), nHiddenLayerNodes);
    
    fprintf("Training NN for baseline...\n");
    NN.train(features_vec, beta_vec, nTrainIters, 20, 0.8, 1);
    
    obj = 0.0;
    sens = 0.0;
    
    %=====================================================================
    % If there is an adjoint implementation in the solver use this
    %=====================================================================
    if fd_step==0.0
        for i=1:size(solver_dict)
            [obj_temp, dJdbeta, ~, ~] = solver_dict{i}(NN);
            obj = obj + solver_weights(i) * obj_temp;
            sens_temp = NN.getSens(features, dJdbeta);
            sens = sens + solver_weights(i) * sens_temp;
        end
    end
    
    %=====================================================================
    % If there is no adjoint implementation, use finite differences
    %=====================================================================
    if fd_step>0.0
        for i=1:size(solver_dict)
            [obj_temp, ~, ~, ~] = solver_dict{i}(NN);
            obj = obj + solver_weights(i) * obj_temp;
            sens_temp = evalNNSens_FD(solver_dict{i}, NN, obj, fd_step);
            sens = sens + solver_weights(i) * sens_temp;
        end
    end
    
    fprintf("FIML Iteration 000000   Objective %+.10e", obj);
    fprintf('\n');
    objectives = zeros(nFIMLIters, 1);
    
    %=====================================================================
    % Run coupled/integrated FIML
    %=====================================================================
    
    for iter=1:nFIMLIters
        
        NN.vars = NN.vars - stepSize * sens / max(abs(sens));
        
        sens = 0.0;
        
        %=================================================================
        % If adjoint implementation
        %=================================================================
        if fd_step==0.0
            for i=1:size(solver_dict)
                [obj_temp, dJdbeta, ~, ~] = solver_dict{i}(NN);
                objectives(iter) = objectives(iter) + ...
                                   solver_weights(i) * obj_temp;
                sens_temp = NN.getSens(features, dJdbeta);
                sens = sens + solver_weights(i) * sens_temp;
            end
        end
        
        %=================================================================
        % If no adjoint implementation
        %=================================================================
        if fd_step>0.0
            for i=1:size(solver_dict)
                [obj_temp, ~, ~, ~] = solver_dict{i}(NN);
                objectives(iter) = objectives(iter) + ...
                                   solver_weights(i) * obj_temp;
                sens_temp = evalNNSens_FD(solver_dict{i}, NN, obj,...
                                          fd_step);
                sens = sens + solver_weights(i) * sens_temp;
            end
        end
        
        fprintf("FIML Iteration %06d   Objective %+.10e", iter, ...
                objectives(iter));
        fprintf('\n');
        
        NN.save();
        
    end

end

%=========================================================================
% Can be parallelized (care should be taken to ensure only one variable
% is perturbed at a time - MPI style and NOT OpenMP style)
%=========================================================================
function sens = evalNNSens_FD(solver, NN, obj, fd_step)
    sens = zeros(NN.nVars, 1);
    for i=1:NN.nVars
        fprintf("Sensitivity %3d/%3d", i, NN.nVars);
        NN.vars(i) = NN.vars(i) + fd_step;
        [obj_pert, ~, ~, ~] = solver(NN);
        sens(i) = (obj_pert - obj) / fd_step;
        NN.vars(i) = NN.vars(i) - fd_step;
        fprintf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
    end
end