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
        config_fnm='config.json';
    end

    methods (Static)
        function ret = config()
            fid = fopen(util.config_fnm);
            ret = jsondecode(char(fread(fid, inf)'));
            fclose(fid);
        end
    end
end
