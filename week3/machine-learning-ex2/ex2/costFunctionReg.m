function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
g = sigmoid(X*theta);
temp0 = (y.*(-1)).*log(g);
temp1 = (y.*(-1)+1).*log((g).*(-1)+1);

J = 1/m*sum(temp0-temp1)+lambda/2/m*(sum(theta.^2)-theta(1)^2);
% FIXME, needs more understanding
grad = 1/m*((g-y).'*X)+lambda/m*theta.';
grad_temp = 1/m*((g-y).'*X);
grad(1) = grad_temp(1); 




% =============================================================

end
