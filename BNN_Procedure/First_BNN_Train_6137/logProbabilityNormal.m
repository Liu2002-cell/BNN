function logPDF = logProbabilityNormal(y,mu,sigma)
    logPDF = -0.5 .* (y - mu) .^ 2 ./ (sigma.^2) - 0.5*log(2*pi) - log(sigma);
end