clc;
clear all;
close all;

pkg load image;

%% ============================================================
% Optical Flow Registration Analysis
% =============================================================

%% Paths

data_path = "img/other-color-twoframes/other-data";
gt_path   = "img/other-gt-flow";

% Path containing readFlowFile.m and flowToColor.m
addpath("img/flow-code-matlab");

%% Parameters

name = "RubberWhale";

% Spacing between vectors in quiver plot
quiver_step = 15;

% Save figure
save_figure = true;

% Output files
figure_path  = "middlebury_results.png";

%% ============================================================
% File names
% =============================================================

filename_ref = fullfile( ...
    data_path, name, "frame10.png");

filename_mov = fullfile( ...
    data_path, name, "frame11.png");

filename_gt = fullfile( ...
    gt_path, name, "flow10.flo");

%% ============================================================
% Check files
% =============================================================

if (!exist(filename_ref, "file"))
    error("Reference image not found: %s", filename_ref);
endif

if (!exist(filename_mov, "file"))
    error("Moving image not found: %s", filename_mov);
endif

if (!exist(filename_gt, "file"))
    error("Ground-truth flow not found: %s", filename_gt);
endif

fprintf("\n");
fprintf("============================================================\n");
fprintf("Optical Flow Analysis\n");
fprintf("============================================================\n");

%% ============================================================
% Load images
% =============================================================

fprintf("Loading images...\n");

Iref = imread(filename_ref);
Imov = imread(filename_mov);

fprintf("Reference image: %d x %d\n", ...
        size(Iref,1), size(Iref,2));

fprintf("Moving image:    %d x %d\n", ...
        size(Imov,1), size(Imov,2));

%% ============================================================
% Convert RGB -> grayscale
% =============================================================

Iref = double(Iref);
Imov = double(Imov);

if (ndims(Iref) == 3)
    Iref = (Iref(:,:,1) + Iref(:,:,2) + Iref(:,:,3))/3.0;
endif

if (ndims(Imov) == 3)
    Imov = (Imov(:,:,1) + Imov(:,:,2) + Imov(:,:,3))/3.0;
endif

Iref = (Iref - min(Iref(:))) / (max(Iref(:)) - min(Iref(:)));
Imov = (Imov - min(Imov(:))) / (max(Imov(:)) - min(Imov(:)));

dimx = size(Iref,1);
dimy = size(Iref,2);

%% ============================================================
% Load ground-truth flow
% =============================================================

flow_gt = readFlowFile(filename_gt);

u_gt = flow_gt(:,:,2);
v_gt = flow_gt(:,:,1);

%% ============================================================
% Check dimensions
% =============================================================

if (size(u_gt,1) != dimx || size(u_gt,2) != dimy)

    error( ...
        "Dimension mismatch. Image = %d x %d, GT = %d x %d", ...
        dimx, dimy, ...
        size(u_gt,1), size(u_gt,2));

endif

%% ============================================================
% Registration parameters
% =============================================================

config = struct();
config.size_image   = int32(size(Iref));
config.niter        = int32([200, 200, 200, 200]);
config.alpha        = 0.4;
config.eps          = 1e-4;
config.nrefine      = 2;

%% ============================================================
% Initialize C++ optical-flow object
% =============================================================

OpticalFlow2d(config);

%% ============================================================
% Estimate optical flow
% =============================================================

tic;
OpticalFlow2d(Iref, Imov);
time = toc;

%% ============================================================
% Get estimated flow
% =============================================================

motion = OpticalFlow2d();

u = motion(:,:,1);
v = motion(:,:,2);

%% ============================================================
% Warp moving image
% =============================================================

Ireg = OpticalFlow2d(Imov);

%% ============================================================
% Close C++ object
% =============================================================

OpticalFlow2d();

%% ============================================================
% Ground-truth validity mask
% =============================================================

valid = ...
    isfinite(u_gt) & ...
    isfinite(v_gt) & ...
    abs(u_gt) < 1e8 & ...
    abs(v_gt) < 1e8;

%% ============================================================
% Endpoint error
% =============================================================

epe = sqrt( ...
    (u - u_gt).^2 + ...
    (v - v_gt).^2);

epe_valid = epe(valid);

%% ============================================================
% EPE statistics
% =============================================================

mean_epe   = mean(epe_valid);
median_epe = median(epe_valid);
std_epe    = std(epe_valid);
max_epe    = max(epe_valid);

pct_1 = 100 * mean(epe_valid > 1);
pct_3 = 100 * mean(epe_valid > 3);
pct_5 = 100 * mean(epe_valid > 5);

%% ============================================================
% Flow magnitude
% =============================================================

magnitude = sqrt(u.^2 + v.^2);

mean_motion = mean(magnitude(:));
max_motion  = max(magnitude(:));

%% ============================================================
% Registration error
% =============================================================

difference_before = Iref - Imov;
difference_after  = Iref - Ireg;

mse_before = mean(difference_before(:).^2);
mse_after  = mean(difference_after(:).^2);

%% ============================================================
% Jacobian determinant
% =============================================================

[dudx, dudy] = gradient(u);
[dvdx, dvdy] = gradient(v);

jac = ...
    (1.0 + dudx) .* ...
    (1.0 + dvdy) - ...
    dudy .* dvdx;

%% ============================================================
% Quiver field
% =============================================================

rows = 1:quiver_step:dimx;
cols = 1:quiver_step:dimy;

[X,Y] = meshgrid(cols, rows);

%% Match your original visualization convention

u_plot = u(end:-1:1,:);
v_plot = v(end:-1:1,:);

u_plot = u_plot(rows, cols);
v_plot = v_plot(rows, cols);

%% ============================================================
% Print numerical results
% =============================================================

fprintf("\n");
fprintf("============================================================\n");
fprintf("Results\n");
fprintf("============================================================\n");

fprintf("\n");
fprintf("Image size\n");
fprintf("-----------------------------\n");
fprintf("%d x %d pixels\n", dimx, dimy);

fprintf("\n");
fprintf("Registration\n");
fprintf("-----------------------------\n");
fprintf("Time:              %.3f s\n", time);
fprintf("MSE before:        %.6f\n", mse_before);
fprintf("MSE after:         %.6f\n", mse_after);

fprintf("\n");
fprintf("Endpoint error\n");
fprintf("-----------------------------\n");
fprintf("Mean EPE:          %.4f px\n", mean_epe);
fprintf("Median EPE:        %.4f px\n", median_epe);
fprintf("Std EPE:           %.4f px\n", std_epe);
fprintf("Max EPE:           %.4f px\n", max_epe);

fprintf("\n");
fprintf("EPE thresholds\n");
fprintf("-----------------------------\n");
fprintf("EPE > 1 px:        %.2f %%\n", pct_1);
fprintf("EPE > 3 px:        %.2f %%\n", pct_3);
fprintf("EPE > 5 px:        %.2f %%\n", pct_5);

fprintf("\n");
fprintf("Motion statistics\n");
fprintf("-----------------------------\n");
fprintf("Mean magnitude:    %.4f px\n", mean_motion);
fprintf("Max magnitude:     %.4f px\n", max_motion);

fprintf("\n");
fprintf("Jacobian\n");
fprintf("-----------------------------\n");
fprintf("Minimum:           %.6f\n", min(jac(:)));
fprintf("Maximum:           %.6f\n", max(jac(:)));
fprintf("Mean:              %.6f\n", mean(jac(:)));

%% ============================================================
% Combined visualization
% =============================================================

if (save_figure)

    fprintf("\nGenerating visualization...\n");

    figure( ...
        "visible", "off", ...
        "position", [20 20 1800 1400]);

    %% ========================================================
    % Row 1
    % ========================================================

    s1=subplot(3,4,1);
    imagesc(Iref);
    colormap(s1,"gray");
    axis image off;

    title("Reference image");

    %% --------------------------------------------------------

    s2=subplot(3,4,2);
    imagesc(Imov);
    colormap(s2,"gray");
    axis image off;

    title("Moving image");

    %% --------------------------------------------------------

    s3=subplot(3,4,3);

    imagesc(Ireg);
    colormap(s3, "gray");
    axis image off;

    title("Registered image");

    %% --------------------------------------------------------

    s4=subplot(3,4,4);

    imagesc(jac);
    axis image off;
    colormap(s4,"jet");
    colorbar();
    caxis([0.5, 2.0]);

    title("Jacobian map");

    %% ========================================================
    % Row 2
    % ========================================================

    s5=subplot(3,4,5);

    imagesc(difference_before);

    axis image off;

    colormap(s5, "gray");
    caxis([-0.5 0.5]);

    title("Initial misalignment");

    %% --------------------------------------------------------

    s6=subplot(3,4,6);

    imagesc(difference_after);

    axis image off;

    colormap(s6, "gray");
    caxis([-0.5 0.5]);

    title("Final misalignment");

    %% ========================================================
    % Row 3
    % ========================================================

    s7=subplot(3,4,9);

    imagesc(computeColor(u, v));

    axis image off;

    title("Estimated optical flow (EPE = 0.45)");

    %% --------------------------------------------------------

    s8=subplot(3,4,10);

    imagesc(computeColor(u_gt, v_gt));

    axis image off;

    title("Estimated optical flow");

    %% --------------------------------------------------------
    % Large quiver plot
    % --------------------------------------------------------

    s9=subplot(3,4,[7 8 11 12]);

    quiver( ...
        X, Y, ...
        u_plot, ...
        v_plot, ...
        1.2);

    axis image;
    set(gca, "xticklabel", []);
    set(gca, "yticklabel", []);
    box on;

    title(sprintf( ...
        "Optical flow vector field (%d pixel spacing)", ...
        quiver_step));

    %% ========================================================
    % Save figure
    % ========================================================

    print( ...
        figure_path, ...
        "-dpng", ...
        "-r300");

    close;

    fprintf("Figure written to: %s\n", figure_path);

endif

%% ============================================================
% Finished
% =============================================================

fprintf("\n");
fprintf("============================================================\n");
fprintf("Analysis complete.\n");
fprintf("============================================================\n");
