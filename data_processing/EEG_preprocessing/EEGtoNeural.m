

clear
sets=dir('*subject*set');
s=21
%%
for s=1:length(sets)
  sets(s).name
    EEG = pop_loadset(sets(s).name);

  %% Filter (2 - 4Hz) Delta phase
    low_freq = 2;
    high_freq = 4;

    for i=1:EEG.trials
        for j=1:EEG.nbchan
            maxDelta = 0 ; minDelta = 0;
            EEG_delta(j,:,i) = eegfiltfft(EEG.data(j,:,i),EEG.srate,low_freq,high_freq);
            EEG_delta_inv(j,:,i) = - EEG_delta(j,:,i);
            maxDelta = max(EEG_delta_inv(j,:,i));
            minDelta = min(EEG_delta_inv(j,:,i));
            phaseDelta(j,:,i) = ((EEG_delta_inv(j,:,i) - minDelta)./(maxDelta - minDelta));
        end
    end

% figure, 
% hold on;plot(EEG.times,EEG_delta(4,:,4),'green'); plot(EEG.times,phaseDelta(4,:,4)); plot(EEG.times,EEG_delta_inv(4,:,4),'red'); hold off;


%% Filter (30 - 45Hz) Gamma power
    low_freq = 30;
    high_freq = 45;

    for i=1:EEG.trials
        for j=1:EEG.nbchan
            EEG_gamma(j,:,i) = eegfiltfft(EEG.data(j,:,i),EEG.srate,low_freq,high_freq);
            powerGamma(j,:,i) = abs(hilbert(EEG_gamma(j,:,i)));
        end
    end

    Original = EEG;
    EEG.data = (0.5 * phaseDelta) + (0.5 * powerGamma);
    sets(s).name
    EEG = pop_saveset( EEG );
end

%% plot
n_trial = 20;
n_channel=10
figure, 
subplot(5,1,1); plot(Original.times,Original.data(n_channel,:,n_trial)); title('EEG signal');
subplot(5,1,2);hold on; plot(Original.times,EEG_gamma(n_channel,:,n_trial),'red'); plot(Original.times,EEG_delta(n_channel,:,n_trial)); title('Gamma & Delta'); hold off;
subplot(5,1,3); plot(Original.times,powerGamma(n_channel,:,n_trial)); title('Gamma power');
subplot(5,1,4); plot(Original.times,phaseDelta(n_channel,:,n_trial)); title('Delta phase');
subplot(5,1,5); hold on; plot(EEG.times,EEG.data(n_channel,:,n_trial),'red'); 
plot(Original.times,powerGamma(n_channel,:,n_trial));title('Modeled spike vs. gamma power and delta phase');

% figure, 
% for i = 1:4
%     subplot(4,1,i);
%     hold on; title('Modeled spike : a, i & u ');
%     plot(GLM_a.times,GLM_a.data(5,:,i),'green'); 
%     plot(GLM_i.times,GLM_i.data(5,:,i),'red'); 
%     plot(GLM_u.times,GLM_u.data(5,:,i),'blue'); 
%     legend('a clean','i clean','u clean','Location','northwest')
%     hold off;
% end
 
