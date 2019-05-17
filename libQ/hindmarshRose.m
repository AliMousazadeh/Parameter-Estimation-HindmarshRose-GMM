function F = hindmarshRose(x, I, r)
X = x(1);
Y = x(2);
Z = x(3);
F = [3*X^2 - X^3 + Y - Z + I; ...
    1 - 5*X^2 - Y; ...
    r*(4*(X + 1.6) - Z)];
end