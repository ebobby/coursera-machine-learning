function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%%%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%%%regression with multiple variables
%%%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%%%   cost of using theta as the parameter for linear regression to fit the
%%%   data points in X and y. Returns the cost in J and the gradient in grad

  %% Initialize some useful values
  m = length(y); % number of training examples

  J = (1/(2*m)) * sum((X * theta - y) .^ 2) + (lambda/(2*m)) * sum(theta(2:end) .^ 2);
  grad = (1/m) * (X' * (X * theta - y));

  %% Add regularization
  grad(2:end) = grad(2:end) + theta(2:end) * (lambda/m);

  grad = grad(:);
end
