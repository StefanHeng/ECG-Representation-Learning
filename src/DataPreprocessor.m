% Modified from https://github.com/zheng120/ECGDenoisingTool

classdef DataPreprocessor
    properties (Constant)
        C_ZHENG = Util.config.pre_processing.zheng;
    end

    methods
        function ret = zheng(self, sig, fqs)
            % Modified from:
            %
            % ***************************************************************************
            % Copyright 2017-2019, Jianwei Zheng, Chapman University,
            % zheng120@mail.chapman.edu
            %
            % Licensed under the Apache License, Version 2.0 (the "License");
            % you may not use this file except in compliance with the License.
            % You may obtain a copy of the License at
            %
            % 	http://www.apache.org/licenses/LICENSE-2.0
            %
            % Unless required by applicable law or agreed to in writing, software
            % distributed under the License is distributed on an "AS IS" BASIS,
            % WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            % See the License for the specific language governing permissions and
            % limitations under the License.
            %
            % Written by Jianwei Zheng.
            arguments
                self
                sig (1, :) {mustBeNumeric}
                fqs {mustBeNumeric}
        end
            opn.fqs = fqs;
            opn.w_pass = self.C_ZHENG.low_pass.passband;
            opn.w_stop = self.C_ZHENG.low_pass.stopband;
            opn.r_pass = self.C_ZHENG.low_pass.passband_ripple;
            opn.r_stop = self.C_ZHENG.low_pass.stopband_attenuation;
            ret = self.butterworth_low_pass(sig, opn);
%            quit(1)
            ret = ret - self.rloess(ret, fqs);
            ret = self.nlm(ret);
        end

        function ret = resampler(self, fqs, fqs_tgt)
            arguments
                self
                fqs {mustBeNumeric}
                fqs_tgt {mustBeNumeric}
            end
            [numer, denom] = rat(fqs_tgt / fqs);
            assert(fqs_tgt == fqs * numer / denom);
            ret = @(sig) resample(sig, numer, denom);
        end

        function ret = butterworth_low_pass(self, sig, opn)
            arguments
                self
                sig (1, :) {mustBeNumeric}
                opn
%                opn.fqs {mustBeNumeric} = 500
%                opn.w_pass {mustBeNumeric} = self.C_ZHENG.low_pass.passband
%                opn.w_stop {mustBeNumeric} = self.C_ZHENG.low_pass.stopband
%                opn.r_pass {mustBeNumeric} = self.C_ZHENG.low_pass.passband_ripple
%                opn.r_stop {mustBeNumeric} = self.C_ZHENG.low_pass.stopband_attenuation
            end
            nyq = 0.5 * opn.fqs;
            [n, wn] = buttord(opn.w_pass / nyq, opn.w_stop / nyq, opn.r_pass, opn.r_stop);
            [bz, az] = butter(n, wn);
            ret = filtfilt(bz, az, sig);
        end

        function ret = rloess(self, sig, n)
            arguments
                self
                sig (1, :) {mustBeNumeric}
                n {mustBeNumeric}
            end
            ret = smooth(1:length(sig), sig, n, 'rloess').';  % For shape 1 x numel(sig)
        end

        function ret = est_noise_std(self, sig)
            arguments
                self
                sig (1, :) {mustBeNumeric}
            end
            res = sig;
            sq = sqrt(6);
            for i = 2:length(res)-1
                res(i) = (2*res(i) - res(i-1) - res(i+1)) / sq;
            end
            ret = 1.4826 * median(abs(res - median(res)));
        end

        function ret = nlm(self, sig, opn)
            arguments
                self
                sig (1, :) {mustBeNumeric}
                opn.scale {mustBeNumeric} = self.C_ZHENG.nlm.smooth_factor
                opn.sch_wd {mustBeNumeric} = NaN
                opn.patch_wd {mustBeNumeric} = self.C_ZHENG.nlm.window_size
            end
            if ~isnan(opn.sch_wd) sch_wd = opn.sch_wd; else sch_wd = numel(sig); end
            ret = nlm(sig, opn.scale * self.est_noise_std(sig), sch_wd, opn.patch_wd);
        end
    end
end
