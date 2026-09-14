clc;
clear all;
close all;

pkg load image;     % GNU Octave

%% ============================================================
%  Paths
% =============================================================

data_path = "img/other-color-twoframes/other-data";
gt_path   = "img/other-gt-flow";

% Path containing readFlowFile.m from Middlebury flow-code-matlab
addpath("img/flow-code-matlab");

%% ============================================================
% Registration parameters
% =============================================================

niter   = [200 200 200 200];
nscales = 3;
alpha   = 0.4;
eps     = 0.001;

% Spacing between vectors in quiver plot
quiver_step = 15;

% Save figures?
save_figures = true;

results_path = "middlebury_results.csv";

%% ============================================================
% Find all sequences automatically
% =============================================================

entries = dir(data_path);

sequences = {};

for k = 1:length(entries)

    if entries(k).isdir && ...
       !strcmp(entries(k).name, ".") && ...
       !strcmp(entries(k).name, "..")

        sequences{end+1} = entries(k).name;

    endif

endfor

fprintf("Found %d sequences:\n", length(sequences));

for k = 1:length(sequences)
    fprintf("  %s\n", sequences{k});
endfor

%% ============================================================
% Results structure
% =============================================================

results = struct();

%% ============================================================
% Evaluate every sequence
% =============================================================

for k = 1:length(sequences)

    name = sequences{k};

    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("Sequence: %s\n", name);
    fprintf("============================================================\n");

    %% --------------------------------------------------------
    % File names
    % ---------------------------------------------------------

    filename_ref = fullfile( ...
        data_path, name, "frame10.png");

    filename_mov = fullfile( ...
        data_path, name, "frame11.png");

    filename_gt = fullfile( ...
        gt_path, name, "flow10.flo");

    %% Check files

    if (!exist(filename_ref, "file"))
        fprintf("Missing reference image -- skipping\n");
        continue;
    endif

    if (!exist(filename_mov, "file"))
        fprintf("Missing moving image -- skipping\n");
        continue;
    endif

    if (!exist(filename_gt, "file"))
        fprintf("Missing ground truth -- skipping\n");
        continue;
    endif

    %% --------------------------------------------------------
    % Load images
    % ---------------------------------------------------------

    Iref = imread(filename_ref);
    Imov = imread(filename_mov);

    %% Convert RGB -> grayscale

    if (ndims(Iref) == 3)
        Iref = rgb2gray(Iref);
    endif

    if (ndims(Imov) == 3)
        Imov = rgb2gray(Imov);
    endif

    Iref = double(Iref);
    Imov = double(Imov);

    %% --------------------------------------------------------
    % Common intensity normalization
    % ---------------------------------------------------------

    % Use the same scale for both images.
    scale = max([Iref(:); Imov(:)]);

    Iref = Iref / scale;
    Imov = Imov / scale;

    [dimx, dimy] = size(Iref);

    %% --------------------------------------------------------
    % Load ground-truth flow
    % ---------------------------------------------------------

    flow_gt = readFlowFile(filename_gt);

    u_gt = flow_gt(:,:,1);
    v_gt = flow_gt(:,:,2);

    %% Check dimensions

    if (size(u_gt,1) != dimx || size(u_gt,2) != dimy)

        fprintf("Dimension mismatch -- skipping\n");
        fprintf("Image: %d x %d\n", dimx, dimy);
        fprintf("GT:    %d x %d\n", ...
                size(u_gt,1), size(u_gt,2));

        continue;

    endif

    %% --------------------------------------------------------
    % Initialize C++ optical-flow object
    % ---------------------------------------------------------

    OpticalFlow2d( ...
        [dimx, dimy], ...
        niter, ...
        nscales, ...
        alpha, ...
        eps);

    %% --------------------------------------------------------
    % Estimate optical flow
    % ---------------------------------------------------------

    tic;

    OpticalFlow2d(Iref, Imov);

    time = toc;

    %% --------------------------------------------------------
    % Get estimated flow
    % ---------------------------------------------------------

    motion = OpticalFlow2d();

    u = motion(:,:,1);
    v = motion(:,:,2);

    %% --------------------------------------------------------
    % Registered image
    % ---------------------------------------------------------

    Ireg = OpticalFlow2d(Imov);

    %% --------------------------------------------------------
    % Close C++ object
    % ---------------------------------------------------------

    OpticalFlow2d();

    %% ========================================================
    % Ground-truth evaluation
    % ========================================================

    %% Invalid GT pixels

    valid = ...
      isfinite(u_gt) & ...
      isfinite(v_gt) & ...
      abs(u_gt) < 1e8 & ...
      abs(v_gt) < 1e8;

    %% Endpoint error

    epe_forward = sqrt( ...
        (u - u_gt).^2 + ...
        (v - v_gt).^2);

    epe_backward = sqrt( ...
        (u + u_gt).^2 + ...
        (v + v_gt).^2);

    epe_forward_valid = epe_forward(valid);
    epe_backward_valid = epe_backward(valid);


    %% Statistics

    mean_epe_forward   = mean(epe_forward_valid);
    median_epe_forward = median(epe_forward_valid);
    std_epe_forward    = std(epe_forward_valid);
    max_epe_forward    = max(epe_forward_valid);

    mean_epe_backward   = mean(epe_backward_valid);
    median_epe_backward = median(epe_backward_valid);
    std_epe_backward    = std(epe_backward_valid);
    max_epe_backward    = max(epe_backward_valid);

    pct_1_forward = 100 * mean(epe_forward_valid > 1);
    pct_3_forward = 100 * mean(epe_forward_valid > 3);
    pct_5_forward = 100 * mean(epe_forward_valid > 5);

    pct_1_backward = 100 * mean(epe_backward_valid > 1);
    pct_3_backward = 100 * mean(epe_backward_valid > 3);
    pct_5_backward = 100 * mean(epe_backward_valid > 5);

    %% Flow magnitude

    magnitude = sqrt(u.^2 + v.^2);

    mean_motion = mean(magnitude(:));
    max_motion  = max(magnitude(:));

    %% --------------------------------------------------------
    % Registration error
    % ---------------------------------------------------------

    difference_before = Iref - Imov;
    difference_after  = Iref - Ireg;

    mae_before = mean(abs(difference_before(:)));
    mae_after  = mean(abs(difference_after(:)));

    %% --------------------------------------------------------
    % Store results
    % ---------------------------------------------------------

    results(k).name = name;

    results(k).time = time;

    results(k).mean_epe   = mean_epe_forward;
    results(k).median_epe = median_epe_forward;
    results(k).std_epe    = std_epe_forward;
    results(k).max_epe    = max_epe_forward;

    results(k).pct_epe_1 = pct_1_forward;
    results(k).pct_epe_3 = pct_3_forward;
    results(k).pct_epe_5 = pct_5_forward;

    results(k).mean_motion = mean_motion;
    results(k).max_motion  = max_motion;

    results(k).mae_before = mae_before;
    results(k).mae_after  = mae_after;

    %% --------------------------------------------------------
    % Print results
    % ---------------------------------------------------------

    fprintf("Image size:        %d x %d\n", dimx, dimy);
    fprintf("Time:              %.3f s\n", time);

    fprintf("\n");
    fprintf("EPE statistics\n");
    fprintf("-----------------------------\n");
    fprintf("Mean EPE (forward):          %.4f px\n", mean_epe_forward);
    fprintf("Median EPE (forward):        %.4f px\n", median_epe_forward);
    fprintf("Std EPE (forward):           %.4f px\n", std_epe_forward);
    fprintf("Max EPE (Forward):           %.4f px\n", max_epe_forward);

    fprintf("Mean EPE (backward):          %.4f px\n", mean_epe_backward);
    fprintf("Median EPE (backward):        %.4f px\n", median_epe_backward);
    fprintf("Std EPE (backward):           %.4f px\n", std_epe_backward);
    fprintf("Max EPE (backward):           %.4f px\n", max_epe_backward);

    fprintf("\n");
    fprintf("EPE thresholds\n");
    fprintf("-----------------------------\n");
    fprintf("EPE (forward) > 1 px:        %.2f %%\n", pct_1_forward);
    fprintf("EPE (forward) > 3 px:        %.2f %%\n", pct_3_forward);
    fprintf("EPE (forward) > 5 px:        %.2f %%\n", pct_5_forward);
    fprintf("EPE (backward) > 1 px:        %.2f %%\n", pct_1_backward);
    fprintf("EPE (backward) > 3 px:        %.2f %%\n", pct_3_backward);
    fprintf("EPE (backward) > 5 px:        %.2f %%\n", pct_5_backward);

    fprintf("\n");
    fprintf("Motion statistics\n");
    fprintf("-----------------------------\n");
    fprintf("Mean |u|:          %.4f px\n", mean_motion);
    fprintf("Max |u|:           %.4f px\n", max_motion);

    fprintf("\n");
    fprintf("Registration\n");
    fprintf("-----------------------------\n");
    fprintf("MAE before:        %.6f\n", mae_before);
    fprintf("MAE after:         %.6f\n", mae_after);

    %% ========================================================
    % Visualization
    % ========================================================

    if (save_figures)

        %% ----------------------------------------------------
        % Image results
        % -----------------------------------------------------

        figure("visible", "off");

        subplot(231);
        imagesc(Iref);
        colormap gray;
        title("Reference");
        axis image;
        axis off;

        subplot(232);
        imagesc(Imov);
        colormap gray;
        title("Moving");
        axis image;
        axis off;

        subplot(233);
        imagesc(Ireg);
        colormap gray;
        title("Registered");
        axis image;
        axis off;

        subplot(234);
        imagesc(difference_before);
        caxis([-0.5 0.5]);
        title("Difference before");
        axis image;
        axis off;

        subplot(235);
        imagesc(difference_after);
        caxis([-0.5 0.5]);
        title("Difference after");
        axis image;
        axis off;

        subplot(236);
        imagesc(epe_forward_valid);
        colorbar;
        title("Endpoint error");
        axis image;
        axis off;

        filename = sprintf("results_%s.png", name);

        print(filename, "-dpng");

        close;


        %% ----------------------------------------------------
        % Motion magnitude + Jacobian
        % -----------------------------------------------------

        [dudx, dudy] = gradient(u);
        [dvdx, dvdy] = gradient(v);

        jac = ...
            (1.0 + dudx) .* ...
            (1.0 + dvdy) - ...
            dudy .* dvdx;

        figure("visible", "off");

        subplot(121);

        imagesc(magnitude);
        colorbar;
        title("||u||");
        axis image;
        axis off;

        subplot(122);

        imagesc(jac);
        colorbar;
        caxis([0.5 2.0]);
        title("Jacobian");
        axis image;
        axis off;

        filename = sprintf("motion_%s.png", name);

        print(filename, "-dpng");

        close;


        %% ----------------------------------------------------
        % Downsampled quiver plot
        % -----------------------------------------------------

        figure("visible", "off");

        rows = 1:quiver_step:dimx;
        cols = 1:quiver_step:dimy;

        [X,Y] = meshgrid(cols, rows);

        % Your original visualization flips the first image
        % dimension, so reproduce that convention here.

        u_plot = u(end:-1:1,:);
        v_plot = v(end:-1:1,:);

        u_plot = u_plot(rows, cols);
        v_plot = v_plot(rows, cols);

        quiver( ...
            X, Y, ...
            u_plot, ...
            -v_plot, ...
            0);

        axis image;
        title("Motion field");

        filename = sprintf("quiver_%s.png", name);

        print(filename, "-dpng");

        close;

    endif

endfor

%% ============================================================
% Write CSV
% =============================================================

fid = fopen(results_path, "w");

fprintf(fid, ...
    ["sequence,time,mean_epe,median_epe,std_epe,max_epe," ...
     "pct_epe_1,pct_epe_3,pct_epe_5," ...
     "mean_motion,max_motion,mae_before,mae_after\n"]);

for k = 1:length(results)

    if (!isfield(results(k), "name"))
        continue;
    endif

    fprintf(fid, ...
        "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f\n", ...
        results(k).name, ...
        results(k).time, ...
        results(k).mean_epe, ...
        results(k).median_epe, ...
        results(k).std_epe, ...
        results(k).max_epe, ...
        results(k).pct_epe_1, ...
        results(k).pct_epe_3, ...
        results(k).pct_epe_5, ...
        results(k).mean_motion, ...
        results(k).max_motion, ...
        results(k).mae_before, ...
        results(k).mae_after);

endfor

fclose(fid);

fprintf("\n");
fprintf("============================================================\n");
fprintf("Evaluation complete.\n");
fprintf("Results written to: %s\n", results_path);
fprintf("============================================================\n");
