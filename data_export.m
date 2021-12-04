classdef data_export
    properties (Constant)
        DIR_DSET = util.config.meta.dir_dset;
        dp = data_preprocessor;
        dl = data_loader;
    end

    methods
        function ret = run(self, dnm)
            disp(['Denosing dataset [' dnm ']... '])
%            d_dset = util.config.(data_export.DIR_DSET).(dnm);
%            fls = util.get_rec_files(dnm);
            [sigs, attr] = self.dl.run(dnm);
            size(sigs)
            n_rec = size(sigs, 1);
            disp(['... of [' num2str(n_rec) '] elements '])

%            fqs = d_dset.fqs;
            fqs = attr.fqs
            denoiser = @(sig) data_export.dp.zheng(sig, fqs);
            for i = 1:n_rec
                sigs_ = sigs(i);
                size(sigs_)
                fnm = fullfile(f.folder, f.name);
                disp([util.now() '| Denosing file ' num2str(i) ': ' f.name])
                sigs = data_export.single(fnm, denoiser);
                size(sigs)
                disp(util.now())
                quit(1)
            end
        end

        function ret = single(fnm, denoiser)
            ret = readmatrix(fnm).';
            for i = 1:size(ret, 1)
                ret(i, :) = denoiser(ret(i, :));
            end
        end
    end
end
