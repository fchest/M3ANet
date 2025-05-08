%% 1. Put EEG recordings and mastoid recordings together in a matrix


clc
clear
subject='33'
for i=1:30
    
    run=num2str(i)
    load_path=['F:\dataset\doi_10.5061_dryad.070jc__v3\Cocktail Party\EEG\Subject' ...
        subject '\Subject' subject '_Run' run '.mat'];
    load(load_path)
    eegData=[eegData,mastoids];
    save_path=['F:\dataset\doi_10.5061_dryad.070jc__v3\Cocktail Party\'...
    'EEG\Subject' subject '\Subject' subject '_Run' run '_data.mat'];

    save(save_path,'eegData')
end
%% 2. Create each file individually and save them as a .set file

%eeglab
clear
clc
subject='33'

for i=[1:30]
   
    run=num2str(i)
    
    trial=num2str(i)
    data_path=['F:\dataset\doi_10.5061_dryad.070jc__v3\Cocktail Party\'...
'EEG\Subject' subject '\Subject' subject '_Run' run '_data.mat'];

    EEG=pop_importdata('dataformat','matlab','setname',trial,'data',...
        data_path,'srate',128)
    if i<10   %this is just to keep the data in the correct order of the merging later
        EEG = pop_saveset( EEG,['trial' trial] );
    else
        EEG = pop_saveset( EEG,['trial9' trial] );   
    end
end

%% 3. merge all the data together and filter 

sets=dir('*trial*set'); %for this to work you have to be in the folder of the data with the .set files
%end_times= 7681* ones(1,30);%end_record=[181.4375,184.7969,183.3906,185.1328,184.7813,206.5,169.7734,175.0703,180.0781,197.5234,179.2734...
%,183.7188,185.3594,182.1641,187.0234,187.6016,191.3281,184.9766,179.6406,191.0703]
for i =1:length(sets)
  sets(i).name

    i
    EEG = pop_loadset(sets(i).name);
    %EEG=eeg_eegrej(EEG, [end_times(i) length(EEG.data)]);
    if i==1  
        merged = EEG ; 
    else
        merged = pop_mergeset(merged,EEG) ;
    end   
end
%filter the merged data
EEG=pop_eegfiltnew(merged, 0.1, 45)
EEG = pop_saveset( EEG,'filename','merged_BP' );

%% 4. load channel locations

EEG.chanlocs=readlocs('locations.ced')
EEG.chanlocs(18).type = [];
EEG.chanlocs(31).type = [];
EEG = pop_saveset( EEG,'filename','merged_BP' );

%% 5. find bad channels, this is specific to each subject 

for i=1:130
    i
if isempty(find(EEG.data(i,:)>100 | EEG.data(i,:)<-100))  %gives channels with excessive noise
z(i)=0;
else
z(i)=1;
end
end
for i=1:130     %gives the standard deviation of each channel
    i
s(i)=std(EEG.data(i,:));
end

%list of bad channels for each subject:
%subject1:[48]
%subject2: [1 2 5 48 71]
%subject5: [44,49,125]
%subject8: [14]
%subject13:[48]
%subject15: none
%subject17: [32,34,48]
%subject19: [46,48]
% subject20: none
% subject21:[1 2 3 5 48 71]
% subject22: [65 102 110]
% subject23: [48]
% subject24:[1 2 5 9 31 32 37 48 63 71 78]
% subject25:[2 23 26 27 28 48]
% subject26: [14 17 48 84 102 104]
% subject28: [11 14 16 21 22 26 29 38 40 48 50 65 67 100]
% subject29:[39 40 46 47 71 80 96 97]
% subject30:[2 43 48 54 66 70 71 127]
% subject31: [29 48 65 66 75 84 87 97 120 121 122]
info.subject_numbers=[1, 2, 5, 7, 8, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31];
info.subject1.bad_Channels=[48]
info.subject2.bad_channels=[1 2 5 48 71];
info.subject5.bad_channels=[44,49,125];
info.subject8.bad_channels=[14];
info.subject13.bad_Channels=[48];
info.subject15.bad_Channels=[];
info.subject17.bad_Channels= [32,34,48];
info.subject19.bad_Channels= [46,48];
info.subject20.bad_Channels=[];
info.subject21.bad_Channels=[1 2 3 5 48 71];
info.subject22.bad_Channels=[65 102 110];
info.subject23.bad_Channels=[48];
info.subject24.bad_Channels=[1 2 5 9 31 32 37 48 63 71 78];
info.subject25.bad_Channels=[2 23 26 27 28 48];
info.subject26.bad_Channels=[14 17 48 84 102 104];
info.subject28.bad_Channels=[11 14 16 21 22 26 29 38 40 48 50 65 67 100];
info.subject29.bad_Channels=[39 40 46 47 71 80 96 97];
info.subject30.bad_Channels=[2 43 48 54 66 70 71 127];
info.subject31.bad_Channels=[29 48 65 66 75 84 87 97 120 121 122];

EEG=pop_interp(EEG,info.subject1.bad_Channels)
EEG = pop_saveset( EEG,'filename','merged_BP_chremoved' );

%% 6. rereference the data to the average of the mastoid channels
EEG=pop_reref(EEG,{ 'M1' 'M2' })
EEG = pop_saveset( EEG,'filename','merged_BP_chremoved_reref' );



%% 7. add the events 

EEG.event=events_EEGDATA()
%you have to check the event values to be at the correct times. they have
%to be every 60 seconds for the cocktail party data
ALLEEG=[]
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % store changes

%% 8.epoch the data
EEG=pop_epoch(EEG)

%after epoching the data, check the event values again to make sure they
%are correct. For some reason it adds events at 60 seconds which you have
%to delete
EEG = pop_saveset( EEG,'filename','merged_BP_chremoved_reref_epoched' );




%% 9. run ICA
%for the ICA, for each subject, you have to remove the interpolated channel in the step 5,
%because it is not independent
EEG=pop_runica(EEG,'chanind',[1:47 49:128])
EEG = pop_saveset( EEG,'filename','merged_BP_chremoved_reref_epoched_ICA' );
%% 10. remove components
%list of bad components per subject:
% subject1:[1 4 27 34 38 50 67]
% subject2:[1 4 7 15 16 30 62 65]
% subject5:[1 3 4 6 18 21 42 52 68 83 103]
% subject8:[7 12 14 31 32 35 79]
% subject13:[3 30 47 64 70 81 87 115]
% subject15:[1 2 3 4 7 11 14 18 40 44 64]
% subject17:[1 2 11 19 26 31 40 42 46 86 120 ]
% subject19:[2 3 4 10 16 21 24 34 38 40 50 63 126]
% subject20:[1 2 3 4 5 6 7 13 19 23 28 46 63 75]
% subject21:[1 2 4 5 6 8 10 13 17 32 45]
% subject22:[6 25 29 34 60 115]
% subject23:[1 2 3 6 8 17 28 33 40 66 74 76 80 91 97 114 122 125 127]
% subject24:[1 2 10 11 12 24 42 61 82 101]
% subject25:[1 4 10 23 36 46 62 71 82 83 93 94 109 103 117]
% subject26:[1 2 5 6 8 12 16 17 22 23 51 52 79 93 110 117 122]
% subject28:[1 2 3 4 5 37 45 47 58 80 86 103]
% subject29:[1 2 3 4 8 11 12 14 22 28 39 51 54 63 74 110 116 120]
% subject30:[1 4 15 16 18 36 53 75 83 104 107]
% subject31:[1 5 7 9 30 34 39 66 112 114]
%here at the second argument put the list of bad components for each
%subject
EEG=pop_subcomp(EEG,[1 4 27 34 38 50 67])
EEG = pop_saveset( EEG,'filename','merged_BP_chremoved_reref_epoched_ICA_removed' );

%% 11. interpolate the bad channels again
EEG=pop_interp(EEG,info.subject1.bad_Channels)
EEG = pop_saveset( EEG,'filename',['subject' subject] );