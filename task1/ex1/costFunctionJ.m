function J = costFunctionJ(X, y, theta)

% X is the "design matrix" with the training examples
% y is the class labels

m = size(X,1); % number of training examples
predictions = X*theta; %prediction of hypothesis in all m examples	

sqrErrors = (predictions-y).^2;  % squared Errors

J = 1/(2*m) * sum(sqrErrors) 
