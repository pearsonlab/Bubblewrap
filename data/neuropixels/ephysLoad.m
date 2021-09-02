ephysroot = '/home/carsen/dm11/data/Spikes/eightprobes/'
% mouse names
mstr = {'Krebs','Waksman','Robbins'};
% start of spontaneous activity in each mouse
tstart = [3811 3633 3323];

% probe location in brain information
% borders tells you region based on depth
% ccfCoords tells you position in microns relative to allen CCF
load(fullfile(ephysroot, 'probeLocations.mat'));

% possible brain regions (as strings)
areaLabels = {'FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP',...
'CP','LS','LH','HPF','TH','SC','MB'};

%% to plot probe in wire brain, download https://github.com/cortex-lab/allenCCF and npy-matlab
% note that this plot includes every site plotted as a dot - zoom in to see.
addpath(genpath('/github/allenCCF'));
addpath(genpath('/github/npy-matlab'));
plotBrainGrid([], [], [], 0);
hold all;
co = get(gca, 'ColorOrder');
for imouse = 1:3
  probeColor = co(imouse,:);
  for pidx = 1:numel(probeLocations(imouse).probe)
    ccfCoords = probeLocations(imouse).probe(pidx).ccfCoords;

    % here we divide by 10 to convert to units of voxels (this atlas is 10um
    % voxels, but coordinates are in um) and we swap 3rd with 2nd dimension
    % because Allen atlas dimensions are AP/DV/LR, but the wiremesh brain has
    % DV as Z, the third dimension (better for view/rotation in matlab).
    % So both of these are ultimately quirks of the plotBrainGrid function,
    % not of the ccfCoords data
    plot3(ccfCoords(:,1)/10, ccfCoords(:,3)/10, ccfCoords(:,2)/10, '.', 'Color', probeColor,'markersize',4)
  end
end

for q = 1:2:360
  view(q, 25); drawnow;
end

%%
% which mouse
brainLocAll=[];
for imouse = [1:3]
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
  %save(fullfile(matroot, sprintf('%swithFaces_KS2.mat',mouse_name)), 'stall','Wh','iprobe',...
  %'motSVD','tspont','tVid','srate','brainLoc','areaLabels');

  brainLocAll=[brainLocAll;brainLoc];
  
end
