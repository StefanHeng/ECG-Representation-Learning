classdef data_loader
    % Given dataset name, returns N x 12 x l signal array, per the hdf5 file
    properties (Constant)
        DIR_DSET = util.config.meta.dir_dset;
        PATH_BASE = util.config.meta.path_base;
        DNM = 'my'
        D_DSET = util.config.(data_loader.DIR_DSET).(data_loader.DNM)
    end

    methods
        function ret = run(self, dnm)
            d_dset = util.config.(self.DIR_DSET).(dnm);

            fnm = sprintf(self.D_DSET.rec_fmt, dnm);
            fnm = fullfile(self.PATH_BASE, self.DIR_DSET, self.DNM, fnm)
            h5read(fnm)
        end
    end
end
