classdef data_export
    properties (Constant)
        DIR_DSET = util.config.meta.dir_dset;
        dp = DataPreprocessor;
        dl = DataLoader;
    end

    methods
        function ret = run(self, dnm)
            disp(['Denosing dataset [' dnm ']... '])
%            d_dset = util.config.(data_export.DIR_DSET).(dnm);
%            fls = util.get_rec_files(dnm);
            [sigs, attr] = self.dl.run(dnm);
%            size(sigs)
            n_rec = size(sigs, 1);
            disp(['... of [' num2str(n_rec) '] elements '])

            fqs = attr.fqs
            denoiser = @(sig) data_export.dp.zheng(sig, fqs);
            for i = 1:n_rec
                sigs_ = squeeze(sigs(i, :, :));
                disp([util.now() '| Denosing file #' num2str(i) '... '])
                sigs = self.apply_1d(sigs_, denoiser);
                disp(util.now())
                quit(1)
            end
        end

        function sigs = apply_1d(self, sigs, fn)
            % Apply function along the last dimension of 2D array
            for i = 1:size(sigs, 1)
                sigs(i, :) = fn(sigs(i, :));
            end
        end
    end
end
