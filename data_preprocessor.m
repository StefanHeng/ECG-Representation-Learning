% Modified from https://github.com/zheng120/ECGDenoisingTool

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


classdef data_preprocessor
    properties (Constant)
        config = util.get_config();
        C_ZHENG = util.config.pre_processing.zheng;
    end

    methods (Static)
        function ret = butterworth_low_pass(sig, opn)
            arguments
                sig (1, :) {mustBeNumeric}
                opn.fqs {mustBeNumeric} = 500
                opn.w_pass {mustBeNumeric} = data_preprocessor.C_ZHENG.low_pass.passband
                opn.w_stop {mustBeNumeric} = data_preprocessor.C_ZHENG.low_pass.stopband
                opn.r_pass {mustBeNumeric} = data_preprocessor.C_ZHENG.low_pass.passband_ripple
                opn.r_stop {mustBeNumeric} = data_preprocessor.C_ZHENG.low_pass.stopband_attenuation
            end
%            opn.fqs
%            opn.w_pass
            opn
            size(sig)
        end
    end
end
