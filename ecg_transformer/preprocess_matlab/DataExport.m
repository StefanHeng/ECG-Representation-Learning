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
        function ret = run(self, dnm, key)
            arguments
                self
                dnm
                key {mustBeText} = 'data'
            end
            disp(['Denosing dataset [' dnm ']... '])
            [sigs, attr] = self.dl.run(dnm, key);
            n_rec = size(sigs, 3);  % Column-major memory per MATLAB
            disp(['    ... of [' num2str(n_rec) '] elements '])
            pad = @(n) Util.zero_pad(n, numel(num2str(n_rec)));

            fqs = attr.fqs;
            denoiser = @(sig) self.dp.zheng(sig, fqs);

            fnm = Util.get_dset_combined_fnm(dnm, 'denoised')
            if isfile(fnm)  % Take advantage of prior processing results
                sigs_den = h5read(fnm, '/data');
            else
                sigs_den = zeros(size(sigs));
            end


%            for i = 1:n_rec
%            for i = 1:3328
%             for i = 1:20608+768+128
            for i = 1:21837
                if ~any(sigs_den(:, :, i), 'all')
                    sigs_ = squeeze(sigs(:, :, i));
                    disp([Util.now() '| Denosing file #' pad(i) '... '])
                    sigs_den(:, :, i) = self.apply_1d(sigs_, denoiser);
                else
                    disp([Util.now() '| File #' pad(i) ' was denoised - ignored '])
                end
                if i == 12722
                    % PTB-XL/records500/12000/12722_hr channel 11, has all 0's => nan's after denoising
                    % => explicitly set it to all 0's
                    sig_broken = sigs(:, 11, 12722)
                    assert(all(sig_broken(:)==0));  % all 0's
                    sig_broken_den = denoiser(sig_broken);
                    assert(sum(isnan(sig_broken_den(:))));  % has nan
                    sigs_den(:, 11, i) = zeros(size(sigs_den(:, 11, i)));
                end
            end

            % Write to hdf5
            if isfile(fnm)
                delete(fnm)  % Equivalently, overwrite prior file
            end
            dir_nm = '/data';
            h5create(fnm, dir_nm, size(sigs_den))  % `sz` parameter inspired by `h5py`
            h5write(fnm, dir_nm, sigs_den);
            h5writeatt(fnm, '/', 'meta', jsonencode(attr));
            h5disp(fnm)
        end

        function sigs = apply_1d(self, sigs, fn)
            % Apply function along the closest-tied axis of 2D array
            for i = 1:size(sigs, 2)
                sigs(:, i) = fn(sigs(:, i));
            end
        end
    end
end
