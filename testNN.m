features = reshape(0:0.005:1, 201, 1);
beta     = reshape((0:0.005:1).*(0.005:0.005:1.005), 201, 1);

% Create a neural network
NN = NeuralNetwork(1,[7; 7]);

% Train the neural network
NN = NN.train(features, beta, 2000, 20, 0.8, 1);

% Save the neural network
NN.save();

% Load the neural network
NN.load();

% Predict using the network
beta_pred = NN.predict(features);

% Evaluate sensitivities
% sens = NN.getSens();