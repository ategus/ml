function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_num = length(theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %disp(computeCost(X,y, theta))
    %for theta_iter = 1:theta_num
    %    theta_temp(theta_iter)=theta(theta_iter)-alpha*(1/m)*sum((X*theta(theta_iter)-y)*)
    %end
    theta_temp = theta;
    disp(iter);
 

    t0 = theta_temp(1,1);
    t0 = t0 - alpha *(1/m)* sum(X * theta -y);
    theta(1,1) = t0;
    fprintf('t0 =  %f\n', t0);

    t1 = theta_temp(1,1);
    t1 = t1 - alpha *(1/m)* sum((X * theta -y));
    theta(2,1) = t1;
    fprintf('t1 =  %f\n', t1);
    
    fprintf('theta =  %f\n', theta);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
