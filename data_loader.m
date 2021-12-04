classdef data_loader
    % Given dataset name, returns N x 12 x l signal array, per the hdf5 file
    properties (Constant)
        DIR_DSET = util.config.meta.dir_dset;
        PATH_BASE = util.config.meta.path_base;
        DNM = 'my'
        K_ORI = '/ori'
        K_RSAM = '/resampled'
        D_DSET = util.config.(data_loader.DIR_DSET).(data_loader.DNM)
    end

    methods
        function [sigs, attr] = run(self, dnm, which)
            arguments
                self
                dnm
%                sig (1, :) {mustBeNumeric}
%                opn.scale {mustBeNumeric} = data_preprocessor.C_ZHENG.nlm.smooth_factor
%                opn.sch_wd {mustBeNumeric} = NaN
%                opn.patch_wd {mustBeNumeric} = data_preprocessor.C_ZHENG.nlm.window_size
                which string = self.K_ORI
            end
%            d_dset = util.config.(self.DIR_DSET).(dnm)
            fnm = sprintf(self.D_DSET.rec_fmt, dnm);
            fnm = fullfile(self.PATH_BASE, self.DIR_DSET, self.D_DSET.dir_nm, fnm);
            h5disp(fnm, '/')
%            h5read(fnm, '/')
            assert(strcmp(which, self.K_ORI) || strcmp(which, self.K_RSAM))
            if strcmp(which, self.K_ORI)
                sigs = permute(h5read(fnm, self.K_ORI), [3 2 1]);
            elseif strcmp(which, self.K_ORI)
                sigs = permute(h5read(fnm, self.K_RSAM), [3 2 1]);
            end
            attr = jsondecode(h5readatt(fnm, '/', 'meta'));
%            sigs = h5read(fnm, self.K_ORI);
%            size(sigs)
%            sigs = permute(sigs, [3 2 1]);
%            size(sigs)
        end
    end
end