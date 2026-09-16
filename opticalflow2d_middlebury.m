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

    u_gt = flow_gt(:,:,2);
    v_gt = flow_gt(:,:,1); % .flo file has swapped orientation compared to my implementation.

    %% Check dimensions

    if (size(u_gt,1) != dimx || size(u_gt,2) != dimy)

        fprintf("Dimension mismatch -- skipping\n");
        fprintf("Image: %d x %d\n", dimx, dimy);
        fprintf("GT:    %d x %d\n", ...
                size(u_gt,1), size(u_gt,2));

        continue;

    endif

    %% ============================================================
    % Registration parameters
    % =============================================================

    %% Registration parameters
    config = struct();
    config.size_image   = int32(size(Iref));
    config.niter        = int32([200, 200, 200, 200]);
    config.alpha        = 0.3;
    config.eps          = 1e-4;
    config.nrefine      = 2;

    %% --------------------------------------------------------
    % Initialize C++ optical-flow object
    % ---------------------------------------------------------

    OpticalFlow2d(config);

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

    epe = sqrt( ...
        (u - u_gt).^2 + ...
        (v - v_gt).^2);

    epe_valid = epe(valid);


    %% Statistics

    mean_epe   = mean(epe_valid);
    median_epe = median(epe_valid);
    std_epe    = std(epe_valid);
    max_epe    = max(epe_valid);

    pct_1 = 100 * mean(epe_valid > 1);
    pct_3 = 100 * mean(epe_valid > 3);
    pct_5 = 100 * mean(epe_valid > 5);

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

    results(k).mean_epe   = mean_epe;
    results(k).median_epe = median_epe;
    results(k).std_epe    = std_epe;
    results(k).max_epe    = max_epe;

    results(k).pct_epe_1 = pct_1;
    results(k).pct_epe_3 = pct_3;
    results(k).pct_epe_5 = pct_5;

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
        imagesc(epe_valid);
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
