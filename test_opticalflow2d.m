clc;
clear all;
close all;

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
niter = [1000 1000 1000];
nscales = 2;
alpha = 0.9;

%% Load C++ object
OpticalFlow2d([dimx, dimy], niter, nscales, alpha);

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

%% Show some info
fprintf("Distribution: %.3f +/ %.3f\n", mean(motion(:)), std(motion(:)));
fprintf("Maxabs: %.3f\n", max(abs(motion(:))));

%% Show some images
figure();
subplot(231); imagesc(Iref); colormap gray;
subplot(232); imagesc(Imov); colormap gray;
subplot(233); imagesc(Ireg); colormap gray;
subplot(234); imagesc(Iref - Imov); colormap gray; caxis([-1/2 1/2]);
subplot(235); imagesc(Iref - Ireg); colormap gray; caxis([-1/2 1/2]);


%%
figure();
quiver(motion(end:-1:1,:,2), motion(end:-1:1,:,1), 0);

%%
function jac = jacobian(motion)
  [dudx, dudy] = gradient(squeeze(motion(:,:,1)));
  [dvdx, dvdy] = gradient(squeeze(motion(:,:,2)));

  jac = (1.0 + dudx).*(1.0 + dvdy) - dudy.*dvdx;
endfunction

normu = sqrt(motion(:,:,1).^2 + motion(:,:,2).^2);
jac = jacobian(motion);

figure();
subplot(121); imagesc(normu); colormap jet; colorbar;
subplot(122); imagesc(jac); colormap jet; colorbar;


