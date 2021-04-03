clc; clear; close all;
path = genpath('../../matlab');
addpath(path)



%% Evaluation on SBD
categories = categories();
% Original GT (Thin)
eval_dir = {'../../../exps/sbd/dff/dff_val/fuse'};
result_dir = {'../../../exps/sbd/result/evaluation/test/cls/gt_orig_thin/dff'};
evaluation('../../../data/sbd-preprocess/gt_eval/gt_orig_thin/test.mat', '../../../data/sbd-preprocess/gt_eval/gt_orig_thin/cls',...
           eval_dir, result_dir, categories, 0, 99, true, 0.1)
% % Original GT (Raw)
eval_dir = {'../../../exps/sbd/dff/dff_val/fuse'};
result_dir = {'../../../exps/sbd/result/evaluation/test/cls/gt_orig_raw/dff'};
evaluation('../../../data/sbd-preprocess/gt_eval/gt_orig_raw/test.mat', '../../../data/sbd-preprocess/gt_eval/gt_orig_raw/cls',...
           eval_dir, result_dir, categories, 0, 99, false, 0.1)                                                                                                                                                             