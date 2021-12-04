% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];

%try_resample()

de = DataExport;
dnm = 'CHAP_SHAO';
de.run(dnm)


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