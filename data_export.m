config = util.config();

% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
fls = util.get_rec_files(dnm);
fls(78)

for i = 1:numel(fls)
    f = fls(i);
    fnm = fullfile(f.folder, f.name);
    sigs = readmatrix(fnm).';
    size(sigs)

    quit(1)
end
