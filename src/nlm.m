% Modified from https://github.com/zheng120/ECGDenoisingToolx

function sig_den = NLM_1dDarbon(sig, scale, sch_wd, patch_wd)
    % Implements fast NLM method of Darbon et al, for a 1-D signal
    % INPUTS:
    % signal: input signal (vector)
    % scale: Gaussian scale factor
    % sch_wd: max search distance
    % patch_wd: patch half-width
    % OUTPUTS:
    % sig_den: the NLM-denoised signal
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % =========================================================================
    % PAPER INFO:
    %       Brian Tracey and Eric Miller, "Nonlocal means denoising of ECG signals",
    %       IEEE Transactions on Biomedical Engineering, Vol 59, No 9, Sept
    %       2012, pages 2383-2386
    % -------------------------------------------------------------------------
    %       PLEASE CITE THIS PAPER, IF YOU USE THIS CODE FOR ACADEMIC PURPOSES
    % -------------------------------------------------------------------------
    %       For all inquiries, please contact author Brian Tracey(btracey[at]alum.mit.edu)
    %
    %       Last Update 05/09/2013
    % =========================================================================
    if length(sch_wd)==1,  % scalar has been entered; expand into patch sample index vector
        sch_wd_ = -sch_wd:sch_wd;
    else
        sch_wd_ = sch_wd;  % use the vector that has been input
    end
    l_s = length(sig);

    sig_den = NaN * ones(size(sig));

    % to simpify, don't bother denoising edges
    i_strt=1 + patch_wd + 1;
    i_end = l_s - patch_wd;
    sig_den(i_strt:i_end) = 0;

    % initialize weight normalization
    z = zeros(size(sig));
    cnt = zeros(size(sig));

    % convert lambda value to 'h', denominator, as in original Buades papers
    n_patch = 2*patch_wd + 1;
    h = 2*n_patch * scale^2;

    for idx = sch_wd_  % loop over all possible differences: s-t
        % do summation over p  - Eq. 3 in Darbon
        k=1:l_s;
        kplus = k + idx;
        igood = find(kplus>0 & kplus<=l_s);  % ignore OOB data; we could also handle it
        ssd = zeros(size(k));
        ssd(igood) = (sig(k(igood)) - sig(kplus(igood))).^2;
        sdx = cumsum(ssd);

        for ii = i_strt:i_end  % loop over all points 's'
            distance = sdx(ii+patch_wd) - sdx(ii-patch_wd-1); % Eq 4; this is in place of point-by-point MSE
            % but note the -1; we want to include the point ii-iPatchHW

            w = exp(-distance / h);  %Eq 2 in Darbon
            t = ii+idx;  % in the papers, this is not made explicit

            if t>1 && t<=l_s
                sig_den(ii) = sig_den(ii) + w*sig(t);
                z(ii) = z(ii) + w;
                cnt(ii) = cnt(ii)+1;
            end
        end
    end % loop over shifts

    % now apply normalization
    sig_den = sig_den./(z+eps);
    sig_den(1 : patch_wd+1) = sig(1 : patch_wd+1);
    sig_den(end-patch_wd+1 : end) =sig(end-patch_wd+1 : end);
end
