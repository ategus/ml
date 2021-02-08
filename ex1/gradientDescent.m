function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_num = length(theta);
theta_temp = theta;
t0 = theta_temp(1,1);
t1 = theta_temp(2,1); 


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %
    %t0 = t0 - alpha * (1/m) * sum(sum( theta_temp' .* X - y )     );
    %t1 = t1 - alpha * (1/m) * sum(sum((theta_temp' .* X - y ).*X) );

    %theta_temp(1,1) = t0;
    %theta_temp(2,1) = t1;
    
    theta = theta - alpha * (1/m) * (X') * (X*theta-y);


    % ============================================================

    % Save the cost J in every iteration   
    J_history(iter) = computeCost(X, y, theta);
    %disp(theta_temp);

end

%theta = theta_temp;

end
