classdef DataLoader
    % Given dataset name, returns N x 12 x l signal array, per the hdf5 file
    properties (Constant)
        PATH_BASE = Util.config.meta.path_base;
        DIR_DSET = Util.config.meta.dir_dset;
        DNM = 'my';
        D_DSET = Util.config.(DataLoader.DIR_DSET).(DataLoader.DNM);
    end

    methods
        function [sigs, attr] = run(self, dnm)
            arguments
                self
                dnm
            end
            fnm = Util.get_dset_combined_fnm(dnm);
%            h5disp(fnm, '/')
            sigs = permute(h5read(fnm, '/data'), [3 2 1]);
            attr = jsondecode(h5readatt(fnm, '/', 'meta'));
        end
    end
end
