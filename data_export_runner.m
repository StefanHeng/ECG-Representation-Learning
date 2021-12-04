% ld_nms = ["I";"II";"III";"aVR";"aVL";"aVF";"V1";"V2";"V3";"V4";"V5";"V6"];
dnm = 'CHAP_SHAO';
dl = data_loader;
sigs = dl.run('CHAP_SHAO');
disp([util.now() '| Resampling... '])

originalFs = 48000;
desiredFs = 44100;

[p,q] = rat(desiredFs / originalFs)
originalFs * p / q
tEnd = 0.01;
Tx = 0:1/originalFs:tEnd;
f = 500;
x = sin(2*pi*f*Tx);

y = resample(x,p,q);
Ty = (0:numel(y)-1)/desiredFs;

fig = figure('units','inch','position',[7, 7, 16, 9]);
%tightfig;
%set(gca, 'LooseInset', [0.025, 0.025, 0.025, 0.025]);
set(gca,'LooseInset', get(gca, 'TightInset'));
plot(Tx,x,'. ')
hold on
plot(Ty,y,'o ')
hold off
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
