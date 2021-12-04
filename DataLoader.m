classdef DataLoader
    % Given dataset name, returns N x 12 x l signal array, per the hdf5 file
    properties (Constant)
        PATH_BASE = Util.config.meta.path_base;
        DIR_DSET = Util.config.meta.dir_dset;
        DNM = 'my';
%        K_ORI = '/ori'
%        K_RSAM = '/resampled'
        D_DSET = Util.config.(DataLoader.DIR_DSET).(DataLoader.DNM);
    end

    methods
        function [sigs, attr] = run(self, dnm)
            arguments
                self
                dnm
            end
%            fnm = sprintf(self.D_DSET.rec_fmt, dnm);
%            fnm = fullfile(self.PATH_BASE, self.DIR_DSET, self.D_DSET.dir_nm, fnm);
            fnm = Util.get_dset_combined_fnm(dnm)
%            h5disp(fnm, '/')
%            assert(strcmp(which, self.K_ORI) || strcmp(which, self.K_RSAM))
            sigs = permute(h5read(fnm, '/data'), [3 2 1]);
            attr = jsondecode(h5readatt(fnm, '/', 'meta'));
        end
    end
end
