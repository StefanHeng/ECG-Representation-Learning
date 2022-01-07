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
            n_rec = size(sigs, 3);  % Column-major memory per MATLAB
            disp(['    ... of [' num2str(n_rec) '] elements '])
            pad = @(n) Util.zero_pad(n, numel(num2str(n_rec)));

            fqs = attr.fqs;
            denoiser = @(sig) self.dp.zheng(sig, fqs);

            fnm = Util.get_dset_combined_fnm(dnm, 'denoised')
            if isfile(fnm)  % Take advantage of prior processing results
%                'file found'
                sigs_den = h5read(fnm, '/data');
%                sigs_den = permute(h5read(fnm, '/data'), [3 2 1]);
            else
                sigs_den = zeros(size(sigs));
            end
%            size(sigs)
%            size(sigs_den)
%            for i = 1:n_rec
            for i = 1:4
                if ~any(sigs_den(:, :, i), 'all')
                    sigs_ = squeeze(sigs(:, :, i));
                    disp([Util.now() '| Denosing file #' pad(i) '... '])
                    sigs_den(:, :, i) = self.apply_1d(sigs_, denoiser);
                else
                    disp([Util.now() '| File #' pad(i) ' was denoised - ignored '])
                end
            end

            % Write to hdf5
            if isfile(fnm)
                delete(fnm)  % Equivalently, overwrite prior file
            end
            dir_nm = '/data';
%            dims = flip(size(sigs_den));
%            dims = size(sigs_den);
            h5create(fnm, dir_nm, size(sigs_den))  % `sz` parameter inspired by `h5py`
%            h5write(fnm, dir_nm, reshape(sigs_den, dims));
            h5write(fnm, dir_nm, sigs_den);
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
