% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];

addpath('..')

%check_resample()
%try_resample()
%denoise_acc_check()

%Util.config
dnm = 'CHAP_SHAO';
de = DataExport;
%de.run(dnm, 'ori')
de.run(dnm, 'data')


function check_resample()
    dnm = 'CHAP_SHAO';
    dl = DataLoader;
    sigs = dl.run('CHAP_SHAO');
    sig = squeeze(sigs(1, 1, :));
    sz = size(sig)
    fqs_ori = 500;
    fqs_new = 250;
    t_ori = (1:numel(sig)) / fqs_ori;

%    [numer, denom] = rat(fqs_new / fqs_ori)
%    assert(fqs_new == fqs_ori * numer / denom)
%    sig_ori = sig;
%    sig_new = resample(sig_ori, numer, denom);

    dp = DataPreprocessor;
    sig_ori = sig;
%    sig_new = dp.resample(sig_ori, fqs_ori, fqs_new);
    resampler = dp.resampler(fqs_ori, fqs_new);
    sig_new = resampler(sig_ori);
    t_new = (0:numel(sig_new)-1)/fqs_new;

    figure('units','inch','position',[7, 7, 16, 9]);
    %set(gca,'LooseInset', get(gca, 'TightInset'));
    plot(t_ori, sig_ori, '-.', t_new, sig_new, '-o')
    legend('Original', 'Resampled')
    title([dnm ' signal 1 lead I, resample ' num2str(fqs_ori) 'Hz => ' num2str(fqs_new) 'Hz'])
%    saveas(gcf,'plot/explore_resample.png')
end

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