% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];

addpath('..')

%try_resample()
denoise_acc_check()

%Util.config
dnm = 'CHAP_SHAO';
de = DataExport;
%de.run(dnm, 'ori')
%de.run(dnm, 'data')

function try_resample()
    % Python counterpart is faster, probably for concurrency
    dl = DataLoader;
    dp = DataPreprocessor;

    dnm = 'CHAP_SHAO';
    sigs = dl.run('CHAP_SHAO');
    sz = size(sigs)
    fqs = 500;
    fqs_tgt = 250;

    resampler = dp.resampler(fqs, fqs_tgt);
    l = numel(resampler(1:size(sigs, 3)));
    assert(strcmp(class(sigs), 'double'));
    sigs_resam = zeros([size(sigs, [1, 2]), l]);

    disp([Util.now() '| Resampling... '])
    for i = 1:size(sigs, 1)
        sigs_ = squeeze(sigs(i, :, :));
        for j = 1:size(sigs_, 1)
            sigs_resam(i, j, :) = resampler(sigs_(j, :));
        end
    %    disp(util.now())
    %    quit(1)
    end
    disp(Util.now())
end

function denoise_acc_check()
    dp = DataPreprocessor;
    dl = DataLoader;
    dnm = 'CHAP_SHAO';

    [sigs, attr] = dl.run(dnm);
%    size(sigs)
    sig = sigs(:, 1, 78);
%    size(sig)

%    sig_den = dp.zheng(sig, 500)
    sig_den = dp.zheng(sig, 250)
    quit(1)
end