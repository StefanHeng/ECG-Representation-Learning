check_resample()

function check_resample()
    dnm = 'CHAP_SHAO';
    dl = DataLoader;
    sigs = dl.run('CHAP_SHAO');
    sig = squeeze(sigs(1, 1, :));
    size(sig)

    fqs_ori = 500;
    fqs_new = 250;
    [numer, denom] = rat(fqs_new / fqs_ori)
    assert(fqs_new == fqs_ori * numer / denom)
    t_ori = (1:numel(sig)) / fqs_ori;
    sig_ori = sig;
    sig_new = resample(sig_ori, numer, denom);
    t_new = (0:numel(sig_new)-1)/fqs_new;

    figure('units','inch','position',[7, 7, 16, 9]);
    %set(gca,'LooseInset', get(gca, 'TightInset'));
    plot(t_ori, sig_ori, '-.', t_new, sig_new, '-o')
    legend('Original', 'Resampled')
    title([dnm ' signal 1 lead I, resample ' num2str(fqs_ori) 'Hz => ' num2str(fqs_new) 'Hz'])
%    saveas(gcf,'plot/explore_resample.png')
end
