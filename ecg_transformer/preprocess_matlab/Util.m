classdef Util
    properties (Constant)
        config = Util.get_config();
        PATH_BASE = Util.config.meta.path_base;
        DIR_DSET = Util.config.meta.dir_dset;

        DNM = 'my';
        D_DSET = Util.config.(Util.DIR_DSET).(Util.DNM);
    end

    methods (Static)
        function ret = get_config()
            fid = fopen('../util/config.json');
            ret = jsondecode(char(fread(fid, inf)'));
            fclose(fid);
        end

        function ret = get_rec_files(dnm)
            % Get record file structs given dataset name, sorted by path
            d_dset = Util.config.datasets.(dnm);
            ret = dir(fullfile(Util.PATH_BASE, Util.DIR_DSET, d_dset.dir_nm, d_dset.rec_fmt));
            [~, idx] = sort(strcat({ret.folder}, {ret.name}));
            ret = ret(idx);
        end

        function ret = zero_pad(n, n_d)
            % :param n_: Number of decimals in total
            ret = num2str(n, ['%0' num2str(n_d) '.f']);
        end

        function ret = now()
            % String representation of current time, as in `t.strftime('%Y-%m-%d %H:%M:%S')`
            c = fix(clock);
            [y, mo, d, h, mi, s] = deal(c(1), c(2), c(3), c(4), c(5), c(6));
            pad = @(n, n_) Util.zero_pad(n, n_);
            ret = [pad(y, 4) '-' pad(mo, 2) '-' pad(d, 2) ' ' pad(h, 2) ':' pad(mi, 2) ':' pad(s, 2) ];
        end

        function ret = get_dset_combined_fnm(dnm, which)
            arguments
                dnm
                which string = 'combined'
            end
            assert(strcmp(which, 'combined') || strcmp(which, 'denoised'))
            if strcmp(which, 'combined')
                k_fmt = 'rec_fmt';
            else
                k_fmt = 'rec_fmt_denoised';
            end
            ret = sprintf(Util.D_DSET.(k_fmt), dnm);
            ret = fullfile(Util.PATH_BASE, Util.DIR_DSET, Util.D_DSET.dir_nm, ret);
        end
    end
end
