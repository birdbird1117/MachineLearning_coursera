function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    X_1 = [ones(m, 1) X]; %5000*401
    z2 = X_1*Theta1.';%5000*25
    a2 = sigmoid(z2);
    n = size(a2,1);
    
    a2 = [ones(n,1) a2];
    z3 = a2*Theta2.';
    a3 = sigmoid(z3);
    hx = a3;
    Y = zeros(num_labels, m);
    for i = 1:m
       rowIndex = y(i);
       temp = Y(:,i);
       temp(rowIndex,1) = 1;
       Y(:,i) = temp;
    end
    
    temp0 = (Y.'.*(-1)).*log(hx);
    temp1 = (Y.'.*(-1)+1).*log((hx).*(-1)+1);

    J = 1/m*sum(sum(temp0-temp1));
    
    Theta1_without_bias = Theta1;
    Theta2_without_bias = Theta2;
        
    % remove bias
    Theta1_without_bias(:,1) = [];
    Theta2_without_bias(:,1) = [];

    Theta1_power2 = Theta1_without_bias.^2;
    Theta2_power2 = Theta2_without_bias.^2;
    
    % add regularization
    J = J + lambda/2/m*(sum(sum(Theta1_power2)) + sum(sum(Theta2_power2)));
    
    
    Delta3 = a3 - Y.'; %5000*10 % FIXME, why a smooth value - a discrete value?
    Delta2 = Delta3*Theta2_without_bias.*sigmoidGradient(z2); % 5000*25 the same as z2
    %Delta2 = Delta2(2:end);
    %Delta1 = Theta1.'*Delta2.*sigmoidGradient(z1);
    %Delta1 = Delta1(2:end);
    DELTA1 = zeros(size(Delta2.'*X_1));
    DELTA1 = DELTA1 + Delta2.'*X_1;
    DELTA2 = zeros(size(Delta3.'*a2));
    DELTA2 = DELTA2 + Delta3.'*a2;
    
    %DELTA2(1, :) = [];
    
    Theta1_grad = 1/m*DELTA1;
    Theta2_grad = 1/m*DELTA2;
    
% -------------------------------------------------------------
    Theta1_temp = Theta1;
    Theta1_temp(:,1) = 0;
    Theta1_grad_regularization = Theta1_grad + lambda/m*Theta1_temp;
    Theta2_temp = Theta2;
    Theta2_temp(:,1) = 0;
    Theta2_grad_regularization = Theta2_grad + lambda/m*Theta2_temp;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = [Theta1_grad_regularization(:) ; Theta2_grad_regularization(:)];


end
