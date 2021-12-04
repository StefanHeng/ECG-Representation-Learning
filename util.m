classdef util
    properties (Constant)
        config = util.get_config();
        PATH_BASE = util.config.meta.path_base;
        DIR_DSET = util.config.meta.dir_dset;
    end

    methods (Static)
        function ret = get_config()
            fid = fopen('config.json');
            ret = jsondecode(char(fread(fid, inf)'));
            fclose(fid);
        end

        function ret = get_rec_files(dnm)
            % Get record file structs given dataset name, sorted by path
            d_dset = util.config.datasets.(dnm);
            ret = dir(fullfile(util.PATH_BASE, util.DIR_DSET, d_dset.dir_nm, d_dset.rec_fmt));
            [~, idx] = sort(strcat({ret.folder}, {ret.name}));
            ret = ret(idx);
        end

        function ret = now()
            % String representation of current time, as in `t.strftime('%Y-%m-%d %H:%M:%S')`
            c = fix(clock);
            [y, mo, d, h, mi, s] = deal(c(1), c(2), c(3), c(4), c(5), c(6));
            pad = @(n, n_) num2str(n, ['%0' num2str(n_) '.f']);
            ret = [pad(y, 4) '-' pad(mo, 2) '-' pad(d, 2) ' ' pad(h, 2) ':' pad(mi, 2) ':' pad(s, 2) ];
        end
    end
end
