%=======================================================================
% EDIT THE ORIGINAL CODE TO
% 1. OUTPUT OBJECTIVE, SENSITIVITIES, FEATURES AND AUGMENTATION
% 2. TAKE A NEURAL NETWORK AND TRUTH DATA AS ADDITIONAL INPUTS
%=======================================================================
function [obj, sens, features, beta] = RHT(T_inf, nPoints, dt, nIter,...
                                           tol, verbose, nn, data)

    %===================================================================
    % DECLARE AN AUGMENTATION FIELD "BETA"
    %===================================================================
    beta = ones(nPoints, 1);
    %===================================================================

    y   = linspace(0.,1.,nPoints);
    dy2 = (y(2)-y(1))^2;
    T   = zeros(nPoints, 1);
    res = zeros(nPoints, 1);
    
    for iter = 1:nIter
    
        %===============================================================
        % IN THE BEGINNING OF EVERY ITERATION,
        % CALCULATE FEATURES AND AUGMENTATION
        %===============================================================
        features      = zeros(nPoints, 2);
        features(:,1) = T(:,1) / T_inf;
        features(:,2) = y(1,:);
        
        if ~isempty(nn)
            beta = nn.predict(features);
        end
        %===============================================================
        
        res(1)       = -T(1);
        res(end)     = -T(end);
        res(2:end-1) = (T(1:end-2) - 2*T(2:end-1) + T(3:end))/dy2 +...
                       5E-4*beta(2:end-1).*(T_inf^4 - T(2:end-1).^4);
                   
        [jacT, jacbeta] = setJacobian(T, T_inf, beta, dy2);
        
        T = T + (eye(nPoints)/dt - jacT)\res;
        
        if verbose>0
            fprintf("%9d\t%E\n", iter, norm(res));
        end
        
        if(norm(res)<tol)
            break
        end
    
    end
    
    if ~isempty(nn)
        plot(y, T, 'r')
    else
        plot(y, T, 'b')
    end
    hold on
    
    obj  = 0;
    sens = [];
    
    %===================================================================
    % AT THE END OF THE SIMULATION:
    % IF SOME DATA IS PROVIDED, CALCULATE OBJECTIVE AND SENSITIVITIES
    % (ANY TECHNIQUE MAY BE CHOSEN FOR SENSITIVITY EVALUATION
    %  HERE WE HAVE CHOSEN ANALYTICAL GRADIENTS)
    %===================================================================
    if ~isempty(data)
        psi  = (jacT.')\(2.0*(T - data)/size(T,1));
        sens = -(jacbeta.')*psi;
        obj  = sum((T - data).^2 / size(T,1));
        plot(y, data, 'k')
    end
    %===================================================================
    
    hold off

end

function [jacT, jacbeta] = setJacobian(T, T_inf, beta, dy2)
    
    jacT    = zeros(size(T,1)); % Can be sparsified
    jacbeta = zeros(size(T,1)); % Can be sparsified

    jacT(1,1)      = -1.0;
    jacT(end, end) = -1.0;
    
    for ind = 2:size(T,1)-1
        
        jacT(ind, ind-1) =  1.0 / dy2;
        jacT(ind, ind  ) = -2E-3 * beta(ind,1) .* T(ind,1).^3 - 2.0 / dy2;
        jacT(ind, ind+1) =  1.0 / dy2;
    
        jacbeta(ind, ind) = 5E-4 * (T_inf^4 - T(ind).^4);
    
    end

end