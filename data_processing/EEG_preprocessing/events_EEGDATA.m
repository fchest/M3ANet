function event=events_EEGDATA()
event=[]
begin_times=  repmat(60,1,29);

begin_times=[0,begin_times];
events_begin=zeros(1,30);
events_begin_seconds=zeros(1,30);

events_begin(1)=128*begin_times(1);
for i=2:length(begin_times)
    i
    events_begin(i)=128*(begin_times(i))+events_begin(i-1);
end

events_begin_seconds(1)=begin_times(1);
for i=2:length(begin_times)
    i
    events_begin_seconds(i)=(begin_times(i))+events_begin_seconds(i-1);
end


%event(1).latency=events_end(1);
event(1).latency=0;
event(1).type='b';
for i=2:length(events_begin)
%n_events=length(events_times);
event(i).type='b';
event(i).latency=events_begin(i);
event(i).latency/128
end
