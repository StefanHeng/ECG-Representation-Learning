config = util.config();

% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
fls = util.get_rec_files(dnm);
f = fls(78)
dp = data_preprocessor;

%for i = 1:numel(fls)
%    f = fls(i);
    fnm = fullfile(f.folder, f.name)
    sigs = readmatrix(fnm).';
    size(sigs)
    sig = sigs(1, :);
    sig(1:10)
%    size(sig)
    sig = dp.butterworth_low_pass(sig);
%    size(sig)
    'before rloess'
    sig_smoothed = dp.rloess(sig, 500);
    size(sig), size(sig_smoothed)

    dp.est_noise_std(sig - sig_smoothed)

    quit(1)
%end
