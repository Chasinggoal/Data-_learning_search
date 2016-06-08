function [xopt,fopt,niter,gnorm,dx] = gradientDescent(varagin)
%A test script to demonstrate how the nonstochastic gradeint descent method
%works.

if nargin == 0
    %starting point
    x0 = [3 3]';
elseif nargin ==1
    x0 = varagin{1};
end

%Termination tolerance
tol = 1e-6;

maxiter = 1000;
dxmin = 1e-6;
%step size 
alpha = 0.01;

%initialize gradient norm, optimization vector, iteration counter,
%pertubation
gnorm = inf; x = x0; niter = 0; dx = inf;

%define the objective function:
f = @(x1,x2) x1.^2 + x1.*x2 + 3 *x2.^2;
figure(1); clf; ezcontour(f, [-5 5 -5 5]); axis equal; hold on
f2 = @(x) f(x(1),x(2));

while and (gnorm >= tol, and (niter <= maxiter, dx >= dxmin))
    %calculate gradient:
    g = grad(x);
    gnorm = norm(g);
    % take step:
    xnew = x - alpha*g;
    % check step
    if -isfinite(xnew)
        niter
       
    end 
    %plot current point
    plot([x(1) xnew(1)], [x(2) xnew(2)], 'ko-')
    refresh
    %update termination metrics
    niter = niter + 1;
    dx = norm(xnew-x);
    x = xnew;
    
end 
xopt = x;
fopt = f2(xopt);
niter = niter - 1;

%define the gradient of the objective
function g = grad(x)
g = [2*x(1) + x(2)
    x(1) + 6*x(2)];
 