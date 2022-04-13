clearvars;clc;close all;

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


% SYSTEM 1 - SYSTEM 2 label differences

system_1 = scores_system_1_resnet_train;
system_2 = scores_system_2_rawnet_train;

vnames = {'index','system_1_label','system_2_label','true_label', 'track'};
error_table = cell2table({0,0,0,0,''});
error_table.Properties.VariableNames=vnames;

tab_idx = 1;
table_size = size(labels_train);
for i=1:table_size(1)
    label_system_1 = find(system_1(i,:)==max(system_1(i,:)))-1;
    label_system_2 = find(system_2(i,:)==max(system_2(i,:)))-1;
    if (label_system_1~=labels_train{i,'algorithm'}) || (label_system_2~=labels_train{i,"algorithm"})
        row = {i, label_system_1, label_system_2, labels_train{i,"algorithm"}, labels_train{i,"track"}};
        error_table = [error_table;row];
    end
end

error_table(1,:) = [];

figure
histogram(error_table{:,"true_label"})
title('confusions: system 1 - system 2')

figure
confusionchart(error_table{:,"system_1_label"},error_table{:,"system_2_label"})
ylabel("system 1 pred")
xlabel("system 2 pred")
title("System 1 - System 2 (sys1 - sys2 confusions)")

figure
confusionchart(error_table{:,"true_label"},error_table{:,"system_1_label"})
ylabel("true label")
xlabel("system 1 pred")
title("True - System 1 (sys1 - sys2 confusions)")

figure
confusionchart(error_table{:,"true_label"},error_table{:,"system_2_label"})
ylabel("true label")
xlabel("system 2 pred")
title("True - System 2 (sys1 - sys2 confusions)")

% SYSTEM 1 - TRUE LABELS label differences

vnames = {'index','system_1_label','system_2_label','true_label', 'track'};
error_table = cell2table({0,0,0,0,''});
error_table.Properties.VariableNames=vnames;

tab_idx = 1;
table_size = size(labels_train);
for i=1:table_size(1)
    label_system_1 = find(system_1(i,:)==max(system_1(i,:)))-1;
    label_system_2 = find(system_2(i,:)==max(system_2(i,:)))-1;
    if (label_system_1~=labels_train{i,'algorithm'})
        row = {i, label_system_1, label_system_2, labels_train{i,"algorithm"}, labels_train{i,"track"}};
        error_table = [error_table;row];
    end
end

error_table(1,:) = [];

figure
histogram(error_table{:,"true_label"})
title('confusions: system 1 - True labels')

figure
confusionchart(error_table{:,"system_1_label"},error_table{:,"system_2_label"})
ylabel("system 1 pred")
xlabel("system 2 pred")
title("System 1 - System 2 (System 1 confusions)")

figure
confusionchart(error_table{:,"true_label"},error_table{:,"system_2_label"})
ylabel("true label")
xlabel("system 2 pred")
title("True - System 2 (System 1 confusions)")

% SYSTEM 2 - TRUE LABELS label differences
vnames = {'index','system_1_label','system_2_label','true_label', 'track'};
error_table = cell2table({0,0,0,0,''});
error_table.Properties.VariableNames=vnames;

tab_idx = 1;
table_size = size(labels_train);
for i=1:table_size(1)
    label_system_1 = find(system_1(i,:)==max(system_1(i,:)))-1;
    label_system_2 = find(system_2(i,:)==max(system_2(i,:)))-1;
    if (label_system_2~=labels_train{i,"algorithm"})
        row = {i, label_system_1, label_system_2, labels_train{i,"algorithm"}, labels_train{i,"track"}};
        error_table = [error_table;row];
    end
end

error_table(1,:) = [];

figure
histogram(error_table{:,"true_label"})
title('confusions: system 2 - True labels')

figure
confusionchart(error_table{:,"system_1_label"},error_table{:,"system_2_label"})
ylabel("system 1 pred")
xlabel("system 2 pred")
title("System 1 - System 2 (System 2 confusions)")

figure
confusionchart(error_table{:,"true_label"},error_table{:,"system_1_label"})
ylabel("true label")
xlabel("system 1 pred")
title("True - System 1 (System 2 confusions)")

