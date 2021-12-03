config = util.config();
DIR_DSET = util.config.meta.dir_dset;

% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
d_dset = config.(DIR_DSET).(dnm)
fls = util.get_rec_files(dnm);
f = fls(78)

%for i = 1:numel(fls)
%    f = fls(i);
    fnm = fullfile(f.folder, f.name)
    sigs = readmatrix(fnm).';
    sig = sigs(1, :);
%    sig = dp.butterworth_low_pass(sig);
%    sig = sig - dp.rloess(sig, 500);
%    dp.est_noise_std(sig)
    sig = dp.zheng(sig, d_dset.fqs);
    size(sig)

    quit(1)
%end

classdef data_export
    properties (Constant)
        config = util.get_config();
        DIR_DSET = util.config.meta.dir_dset;
        dp = data_preprocessor;
    end

    methods (Static)
        function ret = _(dnm)
            d_dset = data_export.config.(data_export.DIR_DSET).(dnm)
            fls = util.get_rec_files(dnm);


        end

        function ret = single(fnm)
            d_dset = data_export.config.(data_export.DIR_DSET).(dnm)
            fls = util.get_rec_files(dnm);


        end
    end
end
