function RHT_genTrueSol(T_inf, nPoints, nIter, tol)

    y   = linspace(0.,1.,nPoints);
    dy2 = (y(2)-y(1))^2;
    T   = zeros(nPoints, 1);
    
    for iter=1:nIter
        Tcopy = T;
        emiss = 1e-4 * (1.0 + 5.0 * sin(3.0*pi*T/200.0) + exp(0.02 * T));
        T(2:2:end-1) = 0.5 * (T(1:2:end-2) + T(3:2:end)) +...
                       0.5 * dy2 * (emiss(2:2:end-1) .* (T_inf^4 - T(2:2:end-1).^4) +...
                                                 0.5  * (T_inf   - T(2:2:end-1)  ) );
        T(3:2:end-1) = 0.5 * (T(2:2:end-2) + T(4:2:end)) +...
                       0.5 * dy2 * (emiss(3:2:end-1) .* (T_inf^4 - T(3:2:end-1).^4) +...
                                                 0.5  * (T_inf   - T(3:2:end-1)  ) );
        T = Tcopy + 1.0*(T - Tcopy);
        res_norm = norm(T - Tcopy);
        fprintf("%9d\t%E\n", iter, res_norm);
        if(res_norm<tol)
            break
        end
    end
    
    filename = strcat("True/solution_", string(T_inf), ".txt");
    fprintf("Saving to filename %s\n", filename);
    dlmwrite(filename, T);
    
    plot(y,T)

end