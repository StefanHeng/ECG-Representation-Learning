% Modified from https://github.com/zheng120/ECGDenoisingTool

%        DataFileName = strcat(PATH,'/MUSE_20180111_163412_52000.csv')
%        DataFile = table2array(readtable(DataFileName,'ReadVariableNames',true));
%        [rows,~] = size(DataFile);
%        DenoisingData= zeros(rows,12);
%
%        for j=1:12
%            fix(clock)
%            OrigECG  = DataFile(:,j);
%            Fs=500;
%            fp=50;fs=60;
%            rp=1;rs=2.5;
%            wp=fp/(Fs/2);ws=fs/(Fs/2);
%            [n,wn]=buttord(wp,ws,rp,rs);
%            [bz,az] = butter(n,wn);
%            LPassDataFile=filtfilt(bz,az,OrigECG);
%
%            t = 1:length(LPassDataFile);
%            yy2 = smooth(t,LPassDataFile,0.1,'rloess');
%
%            BWRemoveDataFile = (LPassDataFile-yy2);
%
%            Dl1=BWRemoveDataFile;
%            for k=2:length(Dl1)-1
%%                Dl1(k), Dl1(k-1), Dl1(k+1)
%                Dl1(k)=(2*Dl1(k)-Dl1(k-1)-Dl1(k+1))/sqrt(6);
%            end
%
%            NoisSTD = 1.4826*median(abs(Dl1-median(Dl1)))
%            DenoisingData(:,j)= nlm(BWRemoveDataFile,(1.5)*(NoisSTD),5000,10);
%            fix(clock)
%            quit(1);
%        end
%
%        OutputfileName =strcat(OutPutFilePath, FileTable.Patient_ID{i},'.csv');
%        csvwrite(OutputfileName,DenoisingData);
%        fprintf('Finished File: %s\n',FileTable.Patient_ID{i});


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
%            opn.fqs, opn.w_pass
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
