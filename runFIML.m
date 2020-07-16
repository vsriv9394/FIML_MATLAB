NN = FIML(1000, 20, 1e-2, [7; 7], {@solver1, @solver2}, [0.5, 0.5], 0.0);

function [obj, sens, features, beta] = solver1(NN)

    T_inf = 5;

    data = dlmread(strcat("True/solution_", string(T_inf), ".txt"));
    [obj, sens, features, beta] = RHT(T_inf, 129, 1e-2, 1000, 1e-8, 0, ...
                                      NN, data);
    
end


function [obj, sens, features, beta] = solver2(NN)

    T_inf = 10;

    data = dlmread(strcat("True/solution_", string(T_inf), ".txt"));
    [obj, sens, features, beta] = RHT(T_inf, 129, 1e-2, 1000, 1e-8, 0, ...
                                      NN, data);
    
end