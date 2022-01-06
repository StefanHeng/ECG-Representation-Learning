classdef DataExport
    % Export denoised verion of dataset
    properties (Constant)
        Util
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
                disp([Util.now() '| Denosing file #' pad(i) '... '])
                sigs(i, :, :) = self.apply_1d(sigs_, denoiser);
            end

            % Write to hdf5
            fnm = Util.get_dset_combined_fnm(dnm, 'denoised')
            delete(fnm)  % Equivalently, file is overwriten
            dir_nm = '/data';
%            flip(size(sigs))
            dims = flip(size(sigs));
            h5create(fnm, dir_nm, dims)  % `sz` parameter inspired by `h5py`
            h5write(fnm, dir_nm, reshape(sigs, dims));
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
