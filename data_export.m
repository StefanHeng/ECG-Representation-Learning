classdef data_export
    properties (Constant)
        DIR_DSET = util.config.meta.dir_dset;
        dp = data_preprocessor;
    end

    methods (Static)
        function ret = run(dnm)
            disp(['Denosing dataset [' dnm ']... '])
            d_dset = util.config.(data_export.DIR_DSET).(dnm);
            fls = util.get_rec_files(dnm);
            disp(['... of [' num2str(numel(fls)) '] elements '])

            fqs = d_dset.fqs;
            denoiser = @(sig) data_export.dp.zheng(sig, fqs);
            for i = 1:numel(fls)
                f = fls(i);
                fnm = fullfile(f.folder, f.name);
                disp(['Denosing file ' f.name])
                disp(fix(clock))
                disp(util.now())
                sigs = data_export.single(fnm, denoiser);
                size(sigs)
                now = fix(clock)
                quit(1)
            end
        end

        function ret = single(fnm, denoiser)
            ret = readmatrix(fnm).';
            size(ret, 2)
            for i = 1:size(ret, 1)
                ret(i, :) = denoiser(ret(i, :));
            end
        end
    end
end
