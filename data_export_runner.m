% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
dl = DataLoader;
dp = DataPreprocessor
sigs = dl.run('CHAP_SHAO');

fqs = 500;
fqs_tgt = 250;
resampler = dp.resampler(fqs, fqs_tgt);
l = numel(resampler(1:size(sigs, 3)))
assert(strcmp(class(sigs), 'double'))
%sz = size(sigs)
%sz(3) = l
sigs_resam = zeros([size(sigs, [1, 2]), l]);
size(sigs_resam)


disp([util.now() '| Resampling... '])
for i = 1:size(sigs, 1)
    sigs_ = squeeze(sigs(i, :, :));
    for j = 1:size(sigs_, 1)
        sigs_resam(i, j, :) = resampler(sigs_(j, :));
    end
    squeeze(sigs_resam(i, :, :))
    squeeze(sigs_resam(2, :, :))
    quit(1)
end
disp(util.now())

%de = data_export;
%de.run(dnm)
