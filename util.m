%c = config();
%c

%function c = config()
%    fnm = 'config.json';
%    fid = fopen(fnm);
%    c = jsondecode(char(fread(fid, inf)'));
%    fclose(fid);
%end

classdef util
    properties (Constant)
%        config_fnm='config.json';
        config = util.get_config();
%        path = config.meta.path_base
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
            % Get file structs given dataset name, sorted by path
            d_dset = util.config.datasets.(dnm);
            ret = dir(fullfile(util.PATH_BASE, util.DIR_DSET, d_dset.dir_nm, d_dset.rec_fmt));
            [~, idx] = sort(strcat({ret.folder}, {ret.name}));
            ret = ret(idx);
        end
    end
end
