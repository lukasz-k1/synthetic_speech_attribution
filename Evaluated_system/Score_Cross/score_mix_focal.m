clearvars;

% import data in following format:

% -scores sheet - numeric matrix, shape: num_files x  num_classes
% scores used for training

% -filenames sheet - table, headers: 'Filename', shape: num_files x  1
% 'Filename' - filename
% filenames used for training corresponding to scores sheet

% -labels train sheet - table, headers: 'track', 'algorithm', shape: num_files x 2
% 'track' - filename, 'algorithm'- true class in range <0,N_classes-1>
% labels used for training

% -labels eval sheet - table, headers: 'track', shape: num_files x 1
% 'track' - filename
% labels to evaluate

% ===============================================
% SYSTEM 1 - Resnet18
% train
load("scores\system_1\filenames_system_1_resnet_train.mat")
load("scores\system_1\scores_system_1_resnet_train.mat")

% eval
load("scores\system_1\filenames_system_1_resnet_eval.mat")
load("scores\system_1\scores_system_1_resnet_eval.mat")

% ===============================================

% SYSTEM 2 - Rawnet2
% train
load("scores\system_2\filenames_system_2_rawnet_train.mat")
load("scores\system_2\scores_system_2_rawnet_train.mat")

% eval
load("scores\system_2\filenames_system_2_rawnet_eval.mat")
load("scores\system_2\scores_system_2_rawnet_eval.mat")

% ===============================================
% LABELS

% train
load("scores\labels\labels_train.mat")

% eval
load("scores\labels\labels_eval.mat")

% Sort all scores by filename to get corresponding results

[labels_train, ~] = sortrows(labels_train,"track",'descend');
[labels_eval, ~] = sortrows(labels_eval,"track",'descend');

% SYSTEM 1 train
[filenames_system_1_resnet_train, idx] = sortrows(filenames_system_1_resnet_train,"filename",'descend');
scores_system_1_resnet_train = scores_system_1_resnet_train(idx,:);

% SYSTEM 1 eval
[filenames_system_1_resnet_eval, idx] = sortrows(filenames_system_1_resnet_eval,"filename",'descend');
scores_system_1_resnet_eval = scores_system_1_resnet_eval(idx,:);

% SYSTEM 2 train
[filenames_system_2_rawnet_train, idx] = sortrows(filenames_system_2_rawnet_train,"filename",'descend');
scores_system_2_rawnet_train = scores_system_2_rawnet_train(idx,:);

% SYSTEM 2 eval
[filenames_system_2_rawnet_eval, idx] = sortrows(filenames_system_2_rawnet_eval,"filename",'descend');
scores_system_2_rawnet_eval = scores_system_2_rawnet_eval(idx,:);


% TRAIN - find mixing coefficients (alfa, beta)

labels = labels_train{:,2}';
labels = labels + 1; % Focal needs labels in <1:num_classes> format

scores_system_1 = scores_system_1_resnet_train;
scores_system_2 = scores_system_2_rawnet_train;

% data split
split_ratio = 1/24;
split_idx = round(length(labels)*split_ratio);

scores_test = {scores_system_1(1:split_idx,:)', scores_system_2(1:split_idx,:)'};
scores_train = {scores_system_1(split_idx+1:end,:)', scores_system_2(split_idx+1:end,:)'};

% train
[alpha,beta] = train_nary_llr_fusion(scores_train,labels(split_idx+1:end)); 
score_matrix = apply_nary_lin_fusion(scores_test,alpha,beta)';


% system accuracy

[~, idx] = max(score_matrix(1:split_idx, :)', [], 1);

accuracy = mean(labels(1:split_idx)==idx)


% Generate predictions

eval_scores_system_1 = scores_system_1_resnet_eval;
eval_scores_system_2 = scores_system_2_rawnet_eval;

scores_eval = {eval_scores_system_1', eval_scores_system_2'};
 
score_matrix = apply_nary_lin_fusion(scores_eval,alpha,beta);

[~, pred_labels] = max(score_matrix, [], 1);

pred_labels = pred_labels - 1; % return lebels in range <0, num_classes-1>

predictions_sheet = [labels_eval(:,1),table(pred_labels')];
predictions_sheet.Properties.VariableNames = {'track', 'algorithm'};


writetable(predictions_sheet, "Predictions.csv")

