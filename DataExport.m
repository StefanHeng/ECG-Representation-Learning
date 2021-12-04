classdef DataExport
    % Export denoised verion of dataset
    properties (Constant)
        PATH_BASE = Util.config.meta.path_base;
        DIR_DSET = Util.config.meta.dir_dset;
        dp = DataPreprocessor;
        dl = DataLoader;
    end

    methods
        function ret = run(self, dnm)
            disp(['Denosing dataset [' dnm ']... '])
            [sigs, attr] = self.dl.run(dnm);
            n_rec = size(sigs, 1);
            disp(['    ... of [' num2str(n_rec) '] elements '])
            pad = @(n) Util.zero_pad(n, numel(num2str(n_rec)));

            fqs = attr.fqs;
            denoiser = @(sig) self.dp.zheng(sig, fqs);
%            for i = 1:n_rec
            for i = 1:2
                sigs_ = squeeze(sigs(i, :, :));
%                disp([Util.now() '| Denosing file #' num2str(i) '... '])
                disp([Util.now() '| Denosing file #' pad(i) '... '])
                sigs(i, :, :) = self.apply_1d(sigs_, denoiser);
%                disp(util.now())
%                quit(1)
            end

%            fnm = sprintf(self.D_DSET.rec_fmt, dnm);
%            fnm = fullfile(self.PATH_BASE, self.DIR_DSET, self.D_DSET.dir_nm, fnm);
            % Write to hdf5
            fnm = Util.get_dset_combined_fnm(dnm, 'denoised')
            delete(fnm)  % Essentially overwrite the file
            dir_nm = '/data';
            h5create(fnm, dir_nm, flip(size(sigs)))  % `sz` parameter inspired by `h5py`
%            hdf5write(fnm, dir_nm, sigs)
            h5writeatt(fnm, '/', 'meta', jsonencode(attr));
            h5disp(fnm)
        end

        function sigs = apply_1d(self, sigs, fn)
            % Apply function along the last dimension of 2D array
            for i = 1:size(sigs, 1)
                sigs(i, :) = fn(sigs(i, :));
            end
        end
    end
end
