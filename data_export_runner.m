% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
dl = DataLoader;
sigs = dl.run('CHAP_SHAO');
disp([util.now() '| Resampling... '])

fqs_ori = 48000;
fqs_new = 44100;

[numer, denom] = rat(fqs_new / fqs_ori)
fqs_ori * numer / denom
tEnd = 0.01;
t_ori = 0:1/fqs_ori:tEnd;
f = 500;
sig_ori = sin(2*pi*f*t_ori);

sig_new = resample(sig_ori,numer,denom);
t_new = (0:numel(sig_new)-1)/fqs_new;

fig = figure('units','inch','position',[7, 7, 16, 9]);
%tightfig;
%set(gca, 'LooseInset', [0.025, 0.025, 0.025, 0.025]);
set(gca,'LooseInset', get(gca, 'TightInset'));
%plot(Tx,x,'. ')
%hold on
%plot(Ty,y,'o ')
%hold off
%legend('Original','Resampled')
plot(t_ori, sig_ori, '.', t_new, sig_new, 'o')
legend('Original','Resampled')

%for i = 1:size(sigs, 1)
%    sigs_ = squeeze(sigs(i, :, :));
%    for j = 1:size(sigs_, 1)
%        sig = sigs_(j, :);
%        size(sig)
%        quit(1)
%    end
%    quit(1)
%end
%disp(util.now())

%de = data_export;
%de.run(dnm)
