clc;
clear all;
close all;

pkg load image; % GNU Octave only

%% Load images
Iref = imread("img/dirlab5_ref.tiff");
Imov = imread("img/dirlab5_mov.tiff");

Iref = double(squeeze(Iref));
Imov = double(squeeze(Imov));

Iref = (Iref - min(Iref(:))) / (max(Iref(:)) - min(Iref(:)));
Imov = (Imov - min(Imov(:))) / (max(Imov(:)) - min(Imov(:)));

Iref = padarray(Iref, [11 0], "replicate");
Imov = padarray(Imov, [11 0], "replicate");

[dimx, dimy] = size(Iref);

%% Registration paramters
niter = [100 1000 1000 1000];
nscales = 3;
nrefine = 3;
##alpha = [1.0, 0.25, 2.0, 2.0, 5, 0];
alpha = [0.8];

regularisation = 1; % Options:
                    % 0) Diffusion
                    % 1) Curvature
                    % 2) Elastic
                    % 3) Thirions demons
                    % 4) Log-Demons
                    % 5) Fluid

verbose = 1; % 0) off
             % 1) on


%% Load C++ object
OpticalFlow2d([dimx, dimy], ...
  niter, nscales, regularisation, ...
  alpha, length(alpha), nrefine, ...
  verbose);

%% Do the registration
tic;
OpticalFlow2d(Iref, Imov);
time = toc;

%% Get the motion field
motion = OpticalFlow2d();

%% Get the registered image
Ireg = OpticalFlow2d(Imov);

%% Close the C++ object
OpticalFlow2d();

%% Unpad images and motion field
Iref = Iref(1+11:end-11, 1+11:end-11);
Imov = Imov(1+11:end-11, 1+11:end-11);
Ireg = Ireg(1+11:end-11, 1+11:end-11);
motion = motion(1+11:end-11, 1+11:end-11, :);

%% Show some info
fprintf("Distribution: %.3f +/ %.3f\n", mean(motion(:)), std(motion(:)));
fprintf("Maxabs: %.3f\n", max(abs(motion(:))));

%% Show some images
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


