%% Patchs
% Auteur : Julien LEROUGE (julien.lerouge@insa-rouen.fr)
% Date : 21/10/2013
%
% Centre Henri Becquerel,
% Rue d'Amiens
% 76038 Rouen Cedex 1
%
% Equipe QuantIF (LITIS EA 4108)
%
% Description : Création de bases de données jouets

%% Initialization
clear all;
close all;
clc;
rng('shuffle');

%% Chargement des textures
fg = imnormalize(double(imread('D17.gif')));
bg = imnormalize(double(imread('D77.gif')));

%% Param�tres
width = 128;
height = 128;
min_radius = min(width, height)/4;

n_train = 500;
n_valid = 500;
n_test = 500;

x_train = zeros(n_train, width*height);
y_train = zeros(n_train, width*height);
x_valid = zeros(n_valid, width*height);
y_valid = zeros(n_valid, width*height);
x_test = zeros(n_test, width*height);
y_test = zeros(n_test, width*height);

moving = false;

%% Création des données
tic;
for i=1:(n_train+n_test)
    i_center = randi([min_radius, height-min_radius]);
    j_center = randi([min_radius, width-min_radius]);
    max_radius = min([i_center, j_center, height-i_center, width-j_center]);
    radius_ext = randi([min_radius, max_radius]);
    radius_int = randi([ceil(radius_ext/4), floor(3*radius_ext/4)]);
    ydata = imcircle([height, width], i_center, j_center, radius_ext);
    ydata = ydata - imcircle([height, width], i_center, j_center, radius_int);
    if(moving)
        window_fg = [randi([1, size(fg,1)-height]), randi([1, size(fg,2)-width])];
        window_bg = [randi([1, size(bg,1)-height]), randi([1, size(bg,2)-width])];
        xdata = fg(window_fg(1):window_fg(1)+height-1, window_fg(2):window_fg(2)+width-1).*ydata + ...
                bg(window_bg(1):window_bg(1)+height-1, window_bg(2):window_bg(2)+width-1).*(1-ydata);
    else
        xdata = fg(1:height, 1:width).*ydata + bg(1:height, 1:width).*(1-ydata);
    end

    if(i <= n_train)
        x_train(i,:) = reshape(xdata, 1, width*height);
        y_train(i,:) = reshape(ydata, 1, width*height);
    elseif(i <= n_train + n_valid)
        x_valid(i-n_train,:) = reshape(xdata, 1, width*height);
        y_valid(i-n_train,:) = reshape(ydata, 1, width*height);
    else
        x_test(i-n_train-n_valid,:) = reshape(xdata, 1, width*height);
        y_test(i-n_train-n_valid,:) = reshape(ydata, 1, width*height);
    end
end
toc;

%% Enregistrement
save('train', 'x_train', 'y_train');
save('valid', 'x_valid', 'y_valid');
save('test', 'x_test', 'y_test');
