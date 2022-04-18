clearvars;

% ==================================================================
% data_path - path with files from 5 classes
% unseen_path - path with unseen files
% new_data_path - path for new folder with data
% ffmpeg_path - path to ffmpeg binaries

settings = ReadYaml('data_settings.yml');

data_path = settings.data_path;
new_data_path = settings.new_data_path;
unseen_path = settings.unseen_path;
ffmpeg_path = settings.ffmpeg_path;

% ==================================================================
% get data from 5class csv
csv_path = sprintf('%s/labels.csv', data_path);
csv_labels = readtable(csv_path);

% ==================================================================
% get filenames for 6th class
unseen_filenames = dir(fullfile(unseen_path, '*.wav'));
unseen_wav_filenames = cell2table({unseen_filenames.name}');
unseen_wav_filenames.Properties.VariableNames = {'track'};

% ==================================================================
% create 6th class labels
labels(1:1000) = 5;
unseen_labels = [unseen_wav_filenames,table(labels','VariableNames',{'algorithm'})];

csv_labels = [csv_labels;unseen_labels];

% ==================================================================
% save csv with all labels and copy 6th class files to the 5 class folder
copyfile(unseen_path, data_path);
writetable(csv_labels,sprintf('%s/labels.csv', data_path))

% ==================================================================
% create new folder for training data and copy all data with csv file into
new_data_path = sprintf('%s/train_data',new_data_path);
mkdir(new_data_path);

copyfile(data_path, new_data_path);

% ==================================================================
% data augumentation and saving in training folder
file_list = csv_labels{:,"track"};

for i=1:length(file_list)
    % readpath
    filepath = sprintf('%s/%s', data_path, file_list{i});
    [audioIn, fs] = audioread(filepath);
    
    %get label and append csv list
    algorithm = csv_labels{strcmp(csv_labels{:,"track"},file_list{i}),'algorithm'};
    [~, name, ext] = fileparts(filepath);
    csv_labels(end+1,:) = {sprintf('%s_compressed%s', name, ext),algorithm};
    csv_labels(end+1,:) = {sprintf('%s_reverb%s', name, ext),algorithm};
    csv_labels(end+1,:) = {sprintf('%s_noise%s', name, ext),algorithm};
    
    % Compression
    bitrate=randi([2,16]);
    add_compression(filepath, new_data_path, ffmpeg_path, bitrate);
    
    % Reverb
    predelay = rand()*2;
    high_cf = randi([8000,20000]);
    diffusion = rand();
    decay = rand();
    hifreq_damp = rand();
    wetdry_mix = rand();
    fsamps = [2000, 4000, 8000, 16000, 32000, 48000, 96000];
    fsamp = fsamps(randi([1,length(fsamps)]));
    add_reverberation(filepath, new_data_path, audioIn, fs, predelay, high_cf, diffusion, decay,...
        hifreq_damp, wetdry_mix, fsamp);
    
    % Noise
    noise_probability = rand();
    SNR_value = rand()*37+3;
    add_noise(filepath, new_data_path, audioIn, fs, noise_probability, SNR_value);
end

% save new csv
writetable(csv_labels,sprintf('%s/labels.csv', new_data_path))



% ==================================================================
% Functions
% ==================================================================
function add_compression(filepath, dest_path, ffmpeg_path, bitrate)

    [~, name, ~] = fileparts(filepath);

    output_path = sprintf('%s/%s_compressed.mp3', dest_path, name);
    
    cmd = sprintf('%sffmpeg -y -i %s -b:a %dk %s', ffmpeg_path, filepath, bitrate, output_path);
    system(cmd);
end

function add_reverberation(filepath, dest_path, audioIn, fs, predelay, high_cf, diffusion, decay,...
    hifreq_damp, wetdry_mix, fsamp)
    
    reverb = reverberator( ...
        "PreDelay", predelay, ...
        "HighCutFrequency", high_cf, ...
        "Diffusion", diffusion, ...
        "DecayFactor", decay, ...
        "HighFrequencyDamping", hifreq_damp, ...
        "WetDryMix", wetdry_mix, ...
        "SampleRate", fsamp);
    
    audioRev = reverb(audioIn);
    % Stereo to mono
    audioRev = .5*(audioRev(:,1) + audioRev(:,2));
    
    [~, name, ext] = fileparts(filepath);

    output_path = sprintf('%s/%s_reverb%s', dest_path, name, ext);
    
    audiowrite(output_path, audioRev, fs);

end

function add_noise(filepath, dest_path, audioIn, fs, noise_probability, SNR_value)
    
    augmenter = audioDataAugmenter( ...
        "AugmentationParameterSource","specify", ...
        "AddNoiseProbability", noise_probability, ...
        "SNR", SNR_value, ...
        "ApplyTimeStretch", false,...
        "ApplyVolumeControl", false, ...
        "ApplyPitchShift", false, ...
        "ApplyTimeStretch", false, ...
        "ApplyTimeShift", false);
    
    
    data = augment(augmenter, audioIn, fs);
    audioAug = data.Audio{1};
    
    [~, name, ext] = fileparts(filepath);

    output_path = sprintf('%s/%s_noise%s', dest_path, name, ext);
    
    audiowrite(output_path, audioAug, fs);
end
