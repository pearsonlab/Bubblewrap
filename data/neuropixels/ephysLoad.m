% Neuropixels data
% Download all the files at https://figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750
% Slightly modified from https://figshare.com/ndownloader/files/14570105

ephysroot = '.'
% mouse names
% Original: mstr = {'Krebs','Waksman','Robbins'};
% doing only Waksman
mstr = {'Waksman'};
% start of spontaneous activity in each mouse
tstart = [3811 3633 3323];

% probe location in brain information
% borders tells you region based on depth
% ccfCoords tells you position in microns relative to allen CCF
load(fullfile(ephysroot, 'probeLocations.mat'));

% possible brain regions (as strings)
areaLabels = {'FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP',...
'CP','LS','LH','HPF','TH','SC','MB'};

%%
% which mouse
brainLocAll=[];
for imouse = [1:length(mstr)]
  mouse_name = mstr{imouse};

  % load spikes
  load(fullfile(ephysroot, sprintf('spks/spks%s_Feb18.mat',mouse_name)));
  % spks(k).st = spike times (in seconds)
  % spks(k).clu = cluster identity of each spike in st (which neuron does spike belong to)
  % spks(k).Wheights = height of each cluster on the probe

  % load behavior file
  beh = load(fullfile(ephysroot, sprintf('faces/%s_face_proc.mat',mouse_name)));
  % movie of mouse face during recording was dimensionality reduced using SVD
  % beh.motionSVD = timepoints x components
  % beh.motionMask = pixels x pixels x components
  % beh.times = times of behavioral frames (in seconds) in same timeframe as spikes
  motSVD = beh.motionSVD;
  tVid = beh.times; % times of movie frames in spike reference frame

  %% extract spike times and create a matrix neurons x time
  stall = zeros(5e3,5500,'uint8');
  ij = 0;
  maxt=0;
  Wh = [];
  iprobe=[];
  brainLoc=[];
  srate = 30; % sampling rate in Hz (how much to bin matrix)
  % loop over probes
  for k = 1:numel(spks)
    clu = spks(k).clu; % cluster ids
    st  = spks(k).st; % spike times in seconds
    st = round(st*srate); % spike times in 30Hz

    % transform spike times into a matrix
    % any duplicates of spike times are added together
    S = sparse(st, clu, ones(1, numel(clu)));
    S = uint8(full(S))';
    % there might be missing numbers (bad spike clusters)
    S = S(sort(unique(clu)),:);

    % add these to the big matrix with all probes
    stall(ij+[1:size(S,1)],1:size(S,2)) = S;
    ij = ij + size(S,1);
    maxt = max(maxt, size(S,2));

    % height of clusters on the probe
    % we will use these to determine the brain area
    whp = spks(k).Wheights(sort(unique(clu)));

    % borders of brain areas wrt the probe
    lowerBorder = probeLocations(imouse).probe(k).borders.lowerBorder;
    upperBorder = probeLocations(imouse).probe(k).borders.upperBorder;
    acronym     = probeLocations(imouse).probe(k).borders.acronym;
    loc = zeros(numel(whp),1);
    % determine brain area for each cluster based on whp
    for j = 1:numel(acronym)
      whichArea = find(strcmp(areaLabels, acronym{j}));
      loc(whp >= lowerBorder(j) & whp < upperBorder(j)) = whichArea;
    end

    % concatenate for all probes
    Wh = [Wh; whp];
    brainLoc = [brainLoc; loc];
    iprobe=[iprobe; k * ones(size(S,1),1)];
  end
  %%
  stall = stall(1:ij, 1:maxt);

  % only use spontaneous time frames
  tspont = tstart(imouse)*srate : min(floor(tVid(end)*srate), size(stall,2)-4);
  stall = stall(:,tspont);
  tspont = tspont / srate; % put back in seconds

  % to put the behavioral data into the spike frames (no time delays)
  x = interp1(tVid, motSVD, tspont);

  %%% save the extracted spike matrix and brain locations and faces %%%
  save(fullfile('.', sprintf('%swithFaces_KS2.mat',mouse_name)), 'stall','Wh','iprobe',...
  'motSVD','tspont','tVid','srate','brainLoc','areaLabels');

  brainLocAll=[brainLocAll;brainLoc];
  
end