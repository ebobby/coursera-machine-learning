function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%%%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%%%neural network which performs classification
%%%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%%%   X, y, lambda) computes the cost and gradient of the neural network. The
%%%   parameters for the neural network are "unrolled" into the vector
%%%   nn_params and need to be converted back into the weight matrices.
%%%
%%%   The returned parameter grad should be a "unrolled" vector of the
%%%   partial derivatives of the neural network.
%%%

  %% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  %% for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  %% Setup some useful variables
  m = size(X, 1);

  %% Vectorize y
  yk = zeros(m, num_labels);
  for i = 1:m
    yk(i, y(i)) = 1;
  end

  %% Forward propagation
  a1 = [ones(m, 1) X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2), 1) a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  %% Base cost function
  J = (1/m) * sum(sum(-yk .* log(a3) - (1 - yk) .* log(1 - a3), 2));

  %% Add regularization
  J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) +
                                sum(sum(Theta2(:, 2:end) .^ 2)));

  %% Back propagation (for loop)
  %%for t = 1:m
  %%  delta_3 = (a3(t, :) - yk(t, :))(:);
  %%  delta_2 = (Theta2(:, 2:end)' * delta_3) .* sigmoidGradient(z2(t, :))(:);
  %%
  %%  Theta1_grad = Theta1_grad + delta_2 * a1(t, :);
  %%  Theta2_grad = Theta2_grad + delta_3 * a2(t, :);
  %%endfor
  %%
  %%Theta1_grad = Theta1_grad * (1/m);
  %%Theta2_grad = Theta2_grad * (1/m);

  %% Back propagation (vectorized)
  delta_3 = a3 - yk;
  delta_2 = delta_3 * Theta2(:, 2:end) .* sigmoidGradient(z2);
  Theta1_grad = delta_2' * a1 * (1/m);
  Theta2_grad = delta_3' * a2 * (1/m);

  %% Apply regularization
  Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end) * (lambda / m);
  Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end) * (lambda / m);

  %% Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
