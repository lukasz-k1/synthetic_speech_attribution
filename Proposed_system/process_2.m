clearvars;

% ==================================================================
% data_path - path with files from 6 classes
% new_data_path - path for new folder with data
% ffmpeg_path - path to ffmpeg binaries

% files will be saved with coder which informs about used effects
% c - compression, r - reverb, n - noise

settings = ReadYaml('data_settings.yml');

data_path = settings.data_path;
new_data_path = settings.new_data_path;
ffmpeg_path = settings.ffmpeg_path;

output_path = sprintf('%s/train_data',new_data_path);
tmp_path = sprintf('%s/tmp',new_data_path); %tmp dir used for mp3 save-read-delete operation
mkdir(tmp_path);

% ==================================================================
% get data from 6class csv
csv_path = sprintf('%s/labels.csv', data_path);
csv_labels = readtable(csv_path);

% get data from augumented 6class csv
csv_aug_path = sprintf('%s/labels.csv', output_path);
csv_aug_labels = readtable(csv_aug_path);
% ==================================================================
% data augumentation and saving in training folder
file_list = csv_labels{:,"track"};

for i=1:length(file_list)
    
    filepath = sprintf('%s/%s', data_path, file_list{i});

    % random choice of effects to be used
    coder = '___';
    aug = 'crn';
    aug = aug(randperm(3));
    aug = aug(1:randi(2,1)+1);

    [audioIn, fs] = audioread(filepath);

    % Compression
    if contains(aug,'c')
        bitrate=randi([2,16]);
        audioIn = add_compression(filepath, tmp_path, ffmpeg_path, bitrate);
        coder(1)='c';
    end

    % Reverb
    if contains(aug,'r')
        predelay = rand()*2;
        high_cf = randi([8000,20000]);
        diffusion = rand();
        decay = rand();
        hifreq_damp = rand();
        wetdry_mix = rand();
        fsamps = [2000, 4000, 8000, 16000, 32000, 48000, 96000];
        fsamp = fsamps(randi([1,length(fsamps)]));
        audioIn = add_reverberation(audioIn, fs, predelay, high_cf, diffusion, decay,...
            hifreq_damp, wetdry_mix, fsamp);
        coder(2)='r';
    end

    % Noise
    if contains(aug,'n')
        noise_probability = rand();
        SNR_value = rand()*37+3;
        audioIn = add_noise(audioIn, fs, noise_probability, SNR_value);
        coder(3)='n';
    end
    
    [~, name, ext] = fileparts(filepath);
    save_path = sprintf('%s/%s%s%s', output_path, name, coder,ext);
    
    audiowrite(save_path, audioIn, fs);

    %get label and append csv list
    algorithm = csv_labels{strcmp(csv_labels{:,"track"},file_list{i}),'algorithm'};
    csv_aug_labels(end+1,:) = {sprintf('%s%s%s', name, coder, ext),algorithm};
end

% save new csv
writetable(csv_aug_labels,csv_aug_path)

function data_out = add_compression(filepath, dest_path, ffmpeg_path, bitrate)

    [~, name, ~] = fileparts(filepath);

    output_path = sprintf('%s/%s_compressed.mp3', dest_path, name);
    
    cmd = sprintf('%sffmpeg -y -i %s -b:a %dk %s', ffmpeg_path, filepath, bitrate, output_path);
    system(cmd);

    [data_out, ~] = audioread(output_path);
    delete(output_path);
end

function data_out = add_reverberation(audioIn, fs, predelay, high_cf, diffusion, decay,...
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
    
    data_out = audioRev;

end

function data_out = add_noise(audioIn, fs, noise_probability, SNR_value)
    
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
    data_out = audioAug;
end

