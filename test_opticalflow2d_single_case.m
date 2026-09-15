clc;
clear all;
close all;

pkg load image; % GNU Octave only

%% Load images
Iref = imread("img/other-color-twoframes/other-data/Venus/frame10.png");
Imov = imread("img/other-color-twoframes/other-data/Venus/frame11.png");

Iref = double(squeeze(Iref));
Imov = double(squeeze(Imov));

%% Convert RGB -> grayscale
if (ndims(Iref) == 3)
    Iref = rgb2gray(Iref);
endif

if (ndims(Imov) == 3)
    Imov = rgb2gray(Imov);
endif

% Normalize
Iref = (Iref - min(Iref(:))) / (max(Iref(:)) - min(Iref(:)));
Imov = (Imov - min(Imov(:))) / (max(Imov(:)) - min(Imov(:)));

%% Registration parameters
config = struct();
config.size_image   = int32(size(Iref));
config.niter        = int32([200, 200, 200, 200]);
config.alpha        = 0.7;
config.eps          = 1e-3;
config.nrefine      = 2;

%% Image registration with Horn-Schunck optical flow.
% Configure OpticalFlow object
OpticalFlow2d(config);

% Register images
tic;
OpticalFlow2d(Iref, Imov);
time = toc;

% Get the motion field
motion = OpticalFlow2d();

% Get the registered image
Ireg = OpticalFlow2d(Imov);

% Close the OpticalFlow object
OpticalFlow2d();

%% Show image alignment
figure();
subplot(231); imagesc(Iref); colormap gray; title("Reference image", "fontsize", 20); axis off;
subplot(232); imagesc(Imov); colormap gray; title("Moving image", "fontsize", 20); axis off;
subplot(233); imagesc(Ireg); colormap gray; title("Registered image", "fontsize", 20); axis off;
subplot(234); imagesc(Iref - Imov); colormap gray; caxis([-1/2 1/2]); title("Difference before", "fontsize", 20); axis off;
subplot(235); imagesc(Iref - Ireg); colormap gray; caxis([-1/2 1/2]); title("Difference after", "fontsize", 20); axis off;

%% Show motion field
figure();

% Downsample motion field
quiver_step = 15;

rows = 1:quiver_step:size(Iref,1);
cols = 1:quiver_step:size(Iref,2);
[X,Y] = meshgrid(cols, rows);

u = squeeze(motion(:,:,1));
v = squeeze(motion(:,:,2));

u_plot = u(end:-1:1,:); 
u_plot = u_plot(rows, cols);
v_plot = v(end:-1:1,:); 
v_plot = v_plot(rows, cols);

quiver(X, Y, u_plot, v_plot, 0.9);

title("Motion field", "fontsize", 20); axis off;

%% Jacobian and motion field magnitude
[dudx, dudy] = gradient(squeeze(motion(:,:,1)));
[dvdx, dvdy] = gradient(squeeze(motion(:,:,2)));

jacobian = (1.0 + dudx).*(1.0 + dvdy) - dudy.*dvdx;

magnitude = sqrt(motion(:,:,1).^2 + motion(:,:,2).^2);

figure();
subplot(121); imagesc(magnitude); colormap jet; colorbar; title("||u||", "fontsize", 20); axis off;
subplot(122); imagesc(jacobian); colormap jet; colorbar(); caxis([0.5 2.0]); title("Jacobian", "fontsize", 20); axis off;

%% Statistics
fprintf("Registration statistics:\n")
fprintf("Image size:            %d x %d\n", size(Iref,1), size(Iref,2));
fprintf("Registration time (s): %.3f s\n\n", time);

fprintf("Motion field statistics:\n")
fprintf("Mean |u|:              %.3f px\n", mean(magnitude(:)));
fprintf("Max |u|:               %.3f px\n", max(magnitude(:)));
fprintf("Jac mean (sd):         %.3f (%.3f)\n", mean(jacobian(:)), std(jacobian(:)));
idcs = jacobian(:) < 0;
fprintf("1st percentile Jac:    %.3f\n", prctile(jacobian(:), 1));
fprintf("#Jac < 0:              %.3f %%\n\n", sum(idcs(:))/prod(size(Iref))*100);

fprintf("Image alignment statistics:\n");
fprintf("MAE (before):                   %.4f\n", mean(abs(Iref(:) - Imov(:))));
fprintf("MAE (after):                    %.4f\n", mean(abs(Iref(:) - Ireg(:))));
fprintf("MSE (before):                   %.4f\n", mean((Iref(:) - Imov(:)).^2));
fprintf("MSE (after):                    %.4f\n", mean((Iref(:) - Ireg(:)).^2));


