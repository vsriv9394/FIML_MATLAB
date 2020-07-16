T_inf = 10;
NN = NeuralNetwork(2, [7; 7]);
NN = NN.load(); % Assignment is important to update NN
data = dlmread(strcat("True/solution_", string(T_inf), ".txt"));
[obj, sens, features, beta] = RHT(T_inf, 129, 1e-2, 1000, 1e-8, 0,...
                                  [], []);
hold on
[obj, sens, features, beta] = RHT(T_inf, 129, 1e-2, 1000, 1e-8, 0,...
                                  NN, data);
legend('Baseline','Augmented','Data')