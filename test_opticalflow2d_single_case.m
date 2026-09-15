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

%% Configure OpticalFlow object
OpticalFlow2d(config);

%% Register images
tic;
OpticalFlow2d(Iref, Imov);
time = toc;
fprintf("Registration time (s): %.3f\n", time);

%% Get the motion field
motion = OpticalFlow2d();

%% Get the registered image
Ireg = OpticalFlow2d(Imov);

%% Close the OpticalFlow object
OpticalFlow2d();

%% Show image alignment
figure();
subplot(231); imagesc(Iref); colormap gray; title("Reference image", "fontsize", 20); axis off;
subplot(232); imagesc(Imov); colormap gray; title("Moving image", "fontsize", 20); axis off;
subplot(233); imagesc(Ireg); colormap gray; title("Registered image", "fontsize", 20); axis off;
subplot(234); imagesc(Iref - Imov); colormap gray; caxis([-1/2 1/2]); title("Difference before", "fontsize", 20); axis off;
subplot(235); imagesc(Iref - Ireg); colormap gray; caxis([-1/2 1/2]); title("Difference after", "fontsize", 20); axis off;


%%
figure();
quiver(motion(end:-1:1,:,2), motion(end:-1:1,:,1), 0); title("Motion field", "fontsize", 20); axis off;

%%
[dudx, dudy] = gradient(squeeze(motion(:,:,1)));
[dvdx, dvdy] = gradient(squeeze(motion(:,:,2)));

jac = (1.0 + dudx).*(1.0 + dvdy) - dudy.*dvdx;

normu = sqrt(motion(:,:,1).^2 + motion(:,:,2).^2);

figure();
subplot(121); imagesc(normu); colormap jet; colorbar; title("||u||", "fontsize", 20); axis off;
subplot(122); imagesc(jac); colormap jet; colorbar(); caxis([0.5 2.0]); title("Jacobian", "fontsize", 20); axis off;
