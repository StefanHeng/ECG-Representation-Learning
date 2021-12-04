% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
%dl = data_loader;
%dl.run('CHAP_SHAO');
de = data_export;
de.run(dnm)
