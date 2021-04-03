clc; clear; close all;
path = genpath('../');
addpath(path)


%% Collect and plot SBD results
categories = categories();
% Original GT (Thin)
result_dir = {'../../../exps/sbd/result/evaluation/test/cls/gt_orig_thin/dff'};
plot_pr(result_dir, {'DFF'}, '../../../exps/sbd/result/evaluation/test/cls/gt_orig_thin/pr_curve', categories, false);
% Original GT (Raw)
result_dir = {'../../../exps/sbd/result/evaluation/test/cls/gt_orig_raw/dff'};
plot_pr(result_dir, {'DFF'}, '../../../exps/sbd/result/evaluation/test/cls/gt_orig_raw/pr_curve', categories, false);