%c = config();
%c

function c = config()
    fnm = 'config.json';
    fid = fopen(fnm);
%    raw = fread(fid, inf);
%    str = char(raw');
%    str = ;
    c = jsondecode(char(fread(fid, inf)'));
    fclose(fid);
end
