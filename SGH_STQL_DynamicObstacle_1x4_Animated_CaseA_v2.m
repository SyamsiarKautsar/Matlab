%% SGH_STQL_DynamicObstacle_1x4_Animated_CaseA_v2.m
% Standalone animated implementation of standard tabular Q-learning (STQL)
% for a 1x4 smart-greenhouse configuration.
%
% IMPORTANT
% ---------
% This file implements the DYNAMIC-TRAINING standard Q-learning baseline:
%
%   Q(x,y,a) : one context-blind Q-table shared by all obstacle contexts.
%
% During training, each episode randomly selects C0 (no moving obstacle)
% or one of C1...C5 (one moving obstacle in that corridor). All experience
% updates the SAME Q-table. No context dimension, policy library, or
% warm-start transfer is used.
%
% During deployment, the same Q-table is used for every context, while the
% instantaneous obstacle position is handled by the same safe-action mask
% used in the CMQL implementation.
%
% Greenhouse Case A:
%   Grid       : 13 x 20
%   Start      : [1 10]
%   Goal       : [10 10]
%   Cell size  : 0.5 m
%   Dynamic scenarios : C0 + 5 corridor contexts
%
% Outputs:
%   STQL_Dynamic_1x4_CaseA_Model.mat
%   STQL_Dynamic_1x4_CaseA_RawEvaluation.csv
%   STQL_Dynamic_1x4_CaseA_ContextSummary.csv
%   STQL_Dynamic_1x4_CaseA_OverallSummary.csv
%
% Keyboard control during animated mission:
%   SPACE = pause / resume animation
%
% MATLAB R2016b+ (script with local functions).

clc;
close all;
clear;

%% ========================================================================
% 1. USER SETTINGS
% =========================================================================

rngSeed = 11;

% Total real training episodes for the single Q-table.
% Set to 5000 for a faster demonstration if desired.
totalEpisodes = 10000;

% Training parameters
alpha = 0.25;
gamma = 0.95;

epsilon0 = 0.95;
epsilonMin = 0.03;

randomStartProb = 0.50;
maxStepsTrain = 300;
maxStepsTest = 450;

% Reward
goalReward = 150;
invalidPenalty = -30;
waitPenalty = -1.20;
progressWeight = 0.60;
nearObstacleRange = 1.5;
nearObstaclePenalty = -0.60;

% Dynamic obstacle
% 1 = obstacle moves every robot decision step.
dynamicObstacleMoveEvery = 1;

% Repeated evaluation per context
nEvalPerContext = 20;

% Animated mission
% C1-C5 = corridors x=[1 4 7 10 13], y=3:18
demoContext = 2;
demoSeed = 2026;
animationDelay = 0.08;

% Save outputs
saveModel = true;
saveCSV = true;

%% ========================================================================
% 2. GREENHOUSE MAP - CASE A (1x4)
% =========================================================================

env.cellSize_m = 0.5;
env.caseID = 'A';
env.name = 'Case A - 1x4 beds';

env.gridSize = [13 20];
env.start = [1 10];
env.goal = [10 10];

% Four long planting-bed blocks for the 1x4 configuration.
% Bed columns: x = 2:3, 5:6, 8:9, 11:12
% Longitudinal extent: y = 3:18
bedObs = [];

for x = [2 5 8 11]
    for xx = x:(x+1)
        for y = 3:18
            bedObs = [bedObs; xx y]; %#ok<AGROW>
        end
    end
end

env.bedObs = unique(bedObs,'rows');

% Fixed objects from the benchmark Case A configuration.
env.objObs = [ ...
     1  9;
    10 11;
     7 18];

env.allStaticObs = unique([env.bedObs; env.objObs],'rows');

env.staticMap = zeros(env.gridSize);

for i = 1:size(env.allStaticObs,1)
    p = env.allStaticObs(i,:);
    env.staticMap(p(1),p(2)) = 1;
end

assert(env.staticMap(env.start(1),env.start(2))==0, ...
    'Start is blocked.');
assert(env.staticMap(env.goal(1),env.goal(2))==0, ...
    'Goal is blocked.');

%% ========================================================================
% 3. ACTION SET: 8-CONNECTED + WAIT
% =========================================================================

env.actions = [ ...
    -1  0; ... % 1: -x
     1  0; ... % 2: +x
     0 -1; ... % 3: -y
     0  1; ... % 4: +y
    -1 -1; ... % 5
    -1  1; ... % 6
     1 -1; ... % 7
     1  1; ... % 8
     0  0];    % 9: WAIT

env.waitID = 9;
nActions = size(env.actions,1);

%% ========================================================================
% 4. DYNAMIC OBSTACLE CORRIDOR CONTEXTS
% =========================================================================

corridorX = [1 4 7 10 13];

defs = cell(5,1);

for k = 1:5
    defs{k} = struct( ...
        'label',sprintf('Corridor %d',k), ...
        'half','SINGLE', ...
        'laneNumber',k, ...
        'x',corridorX(k), ...
        'yRange',3:18);
end

env.corridors = buildCorridors( ...
    env.staticMap,defs,env.start,env.goal);

env.nCorridors = numel(env.corridors);

fprintf('\n============================================================\n');
fprintf(' STANDARD Q-LEARNING (STQL) - DYNAMIC 1x4 SMART GREENHOUSE\n');
fprintf('============================================================\n');
fprintf('Grid             : %d x %d\n', ...
    env.gridSize(1),env.gridSize(2));
fprintf('Start -> Goal    : [%d %d] -> [%d %d]\n', ...
    env.start(1),env.start(2),env.goal(1),env.goal(2));
fprintf('Dynamic scenarios: C0 + %d corridor contexts\n', ...
    env.nCorridors);

for c = 1:env.nCorridors
    fprintf('  C%02d: %-18s | x=%2d | y=%d:%d\n', ...
        c,env.corridors(c).label,env.corridors(c).x, ...
        env.corridors(c).yRange(1),env.corridors(c).yRange(end));
end

%% ========================================================================
% 5. TRAINING CONFIGURATION
% =========================================================================

cfg.alpha = alpha;
cfg.gamma = gamma;

cfg.epsilon0 = epsilon0;
cfg.epsilonMin = epsilonMin;

cfg.randomStartProb = randomStartProb;

cfg.maxStepsTrain = maxStepsTrain;
cfg.maxStepsTest = maxStepsTest;

cfg.goalReward = goalReward;
cfg.invalidPenalty = invalidPenalty;
cfg.waitPenalty = waitPenalty;
cfg.progressWeight = progressWeight;
cfg.nearObstacleRange = nearObstacleRange;
cfg.nearObstaclePenalty = nearObstaclePenalty;

cfg.dynamicObstacleMoveEvery = dynamicObstacleMoveEvery;

rng(rngSeed,'twister');

%% ========================================================================
% 6. STANDARD Q-LEARNING TRAINING
% =========================================================================
%
% One Q-table is used for ALL contexts:
%
%       Q(x,y,a)
%
% At the start of every episode:
%   C0      -> no moving obstacle
%   C1-C5   -> one moving obstacle in the selected corridor
%
% The context is NOT included in the Q-table. This is the standard,
% context-blind baseline used to compare against CA-QL and proposed CMQL.

nR = env.gridSize(1);
nC = env.gridSize(2);

Q = zeros(nR,nC,nActions);

trainSuccess = false(totalEpisodes,1);
trainSteps = zeros(totalEpisodes,1);
trainReturn = zeros(totalEpisodes,1);
trainContext = zeros(totalEpisodes,1);

realTrainingSteps = 0;

fprintf('\nTraining one standard Q-table Q(x,y,a)...\n');
fprintf('Total training episodes : %d\n',totalEpisodes);
fprintf('Episode contexts        : random C0-C%d\n',env.nCorridors);

tTrain = tic;

for ep = 1:totalEpisodes

    % Randomly interleave nominal and dynamic-obstacle contexts.
    context = randi(env.nCorridors+1)-1;
    trainContext(ep) = context;

    if context==0
        dyn = [];
        dynPos = [];
        forbidden = [];
    else
        dyn = initializeDynamicObstacle( ...
            env.corridors(context), ...
            cfg.dynamicObstacleMoveEvery);

        dynPos = dyn.position;
        forbidden = dynPos;
    end

    state = chooseEpisodeStart( ...
        env,cfg.randomStartProb,forbidden);

    epsilon = epsilonSchedule( ...
        ep,totalEpisodes,cfg.epsilon0,cfg.epsilonMin);

    G = 0;

    for step = 1:cfg.maxStepsTrain

        % Same temporal convention as the dynamic CMQL code:
        % obstacle moves before the robot selects its action.
        if context>0
            dyn = moveDynamicObstacle(dyn,state);
            dynPos = dyn.position;
        else
            dynPos = [];
        end

        safeIDs = getSafeActions( ...
            state,env,dynPos,true);

        if isempty(safeIDs)
            actionID = env.waitID;
        else
            actionID = epsilonGreedy( ...
                Q,state,safeIDs,epsilon);
        end

        [nextState,reward,terminal] = ...
            realTransition( ...
            state,actionID,env,dynPos,cfg);

        safeNext = getSafeActions( ...
            nextState,env,dynPos,true);

        if terminal || isempty(safeNext)
            target = reward;
        else
            qNext = squeeze(Q( ...
                nextState(1),nextState(2),safeNext));

            target = reward + cfg.gamma*max(qNext);
        end

        qOld = Q(state(1),state(2),actionID);

        Q(state(1),state(2),actionID) = ...
            qOld + cfg.alpha*(target-qOld);

        realTrainingSteps = realTrainingSteps+1;

        G = G+reward;
        state = nextState;

        if terminal
            trainSuccess(ep)=true;
            break;
        end
    end

    trainSteps(ep)=step;
    trainReturn(ep)=G;

    if mod(ep,max(1,round(totalEpisodes/5)))==0
        recent = max(1,ep-499):ep;

        fprintf(['  STQL %5d/%5d | eps=%.3f | ' ...
            'recent success=%.1f%% | mean steps=%.1f\n'], ...
            ep,totalEpisodes,epsilon, ...
            100*mean(trainSuccess(recent)), ...
            mean(trainSteps(recent)));
    end
end

trainingTime_s = toc(tTrain);

fprintf('\nSTQL training completed in %.3f s.\n',trainingTime_s);
fprintf('Real Q-updates / environment steps = %d\n',realTrainingSteps);

%% ========================================================================
% 7. TRAINING VISUALIZATION
% =========================================================================

figure('Name','STQL Dynamic Training - Case A 1x4 Greenhouse', ...
    'Color','w','Position',[100 100 1000 760]);

subplot(2,1,1);

w = min(300,numel(trainSuccess));

plot(movmean(double(trainSuccess),w)*100,'LineWidth',1.5);
ylim([0 100]);
grid on;

xlabel('Training episode');
ylabel('Moving success rate (%)');
title(sprintf('Standard Q-learning training | moving window=%d',w));

subplot(2,1,2);

contextTrainingSuccess = nan(env.nCorridors+1,1);

for c = 0:env.nCorridors

    rr = trainContext==c;

    if any(rr)
        contextTrainingSuccess(c+1) = ...
            100*mean(trainSuccess(rr));
    end
end

bar(0:env.nCorridors,contextTrainingSuccess);

ylim([0 100]);
grid on;

xlabel('Training context');
ylabel('Training success rate (%)');
title('Observed training success by randomly sampled context');

%% ========================================================================
% 10. REPEATED DYNAMIC EVALUATION
% =========================================================================

fprintf('\nRepeated dynamic evaluation...\n');

nContexts = env.nCorridors+1;
N = nContexts*nEvalPerContext;

Context = zeros(N,1);
Run = zeros(N,1);
Success = false(N,1);
PathLength_m = nan(N,1);
Smoothness_deg = nan(N,1);
WaitActions = nan(N,1);
AvoidanceEvents = nan(N,1);
Steps = nan(N,1);
NavigationTime_ms = nan(N,1);

idx = 0;

for c = 0:env.nCorridors

    fprintf('  Evaluating context C%d...\n',c);

    for r = 1:nEvalPerContext

        idx = idx+1;
        seed = 50000 + 10000*c + r;

        tNav = tic;

        result = runSTQLMission( ...
            Q,env,c,cfg,seed,false,0);

        NavigationTime_ms(idx) = 1000*toc(tNav);

        Context(idx)=c;
        Run(idx)=r;
        Success(idx)=result.success;
        Steps(idx)=result.steps;
        AvoidanceEvents(idx)=result.avoidanceEvents;

        if result.success
            PathLength_m(idx)=result.pathLength_m;
            Smoothness_deg(idx)=result.smoothness_deg;
            WaitActions(idx)=result.waitCount;
        end
    end
end

RawEvaluation = table( ...
    Context,Run,Success,PathLength_m,Smoothness_deg, ...
    WaitActions,AvoidanceEvents,Steps,NavigationTime_ms);

%% ========================================================================
% 11. CONTEXT SUMMARY
% =========================================================================

ContextID = (0:env.nCorridors)';
ContextLabel = strings(nContexts,1);
SuccessRate_pct = zeros(nContexts,1);
MeanPathLength_m = nan(nContexts,1);
StdPathLength_m = nan(nContexts,1);
MeanSmoothness_deg = nan(nContexts,1);
MeanWaitActions = nan(nContexts,1);
MeanAvoidanceEvents = nan(nContexts,1);
MeanSteps = nan(nContexts,1);
MeanNavigationTime_ms = nan(nContexts,1);

for c = 0:env.nCorridors

    rr = RawEvaluation.Context==c;
    T = RawEvaluation(rr,:);

    if c==0
        ContextLabel(c+1)="C0 - nominal/no dynamic obstacle";
    else
        ContextLabel(c+1)= ...
            "C"+c+" - "+string(env.corridors(c).label);
    end

    SuccessRate_pct(c+1)=100*mean(T.Success);
    MeanAvoidanceEvents(c+1)=mean(T.AvoidanceEvents,'omitnan');
    MeanSteps(c+1)=mean(T.Steps,'omitnan');
    MeanNavigationTime_ms(c+1)= ...
        mean(T.NavigationTime_ms,'omitnan');

    ok = T.Success;

    if any(ok)
        MeanPathLength_m(c+1)= ...
            mean(T.PathLength_m(ok),'omitnan');
        StdPathLength_m(c+1)= ...
            std(T.PathLength_m(ok),'omitnan');
        MeanSmoothness_deg(c+1)= ...
            mean(T.Smoothness_deg(ok),'omitnan');
        MeanWaitActions(c+1)= ...
            mean(T.WaitActions(ok),'omitnan');
    end
end

ContextSummary = table( ...
    ContextID,ContextLabel,SuccessRate_pct, ...
    MeanPathLength_m,StdPathLength_m,MeanSmoothness_deg, ...
    MeanWaitActions,MeanAvoidanceEvents,MeanSteps, ...
    MeanNavigationTime_ms);

disp(ContextSummary);

%% ========================================================================
% 12. OVERALL SUMMARY
% =========================================================================

overallSuccess_pct = 100*mean(RawEvaluation.Success);
ok = RawEvaluation.Success;

overallMeanPath_m = mean( ...
    RawEvaluation.PathLength_m(ok),'omitnan');

overallStdPath_m = std( ...
    RawEvaluation.PathLength_m(ok),'omitnan');

overallSmooth_deg = mean( ...
    RawEvaluation.Smoothness_deg(ok),'omitnan');

overallWait = mean( ...
    RawEvaluation.WaitActions(ok),'omitnan');

overallAvoid = mean( ...
    RawEvaluation.AvoidanceEvents,'omitnan');

overallSteps = mean( ...
    RawEvaluation.Steps,'omitnan');

overallNav_ms = mean( ...
    RawEvaluation.NavigationTime_ms,'omitnan');

QValueCount = numel(Q);

OverallSummary = table( ...
    totalEpisodes,trainingTime_s, ...
    realTrainingSteps,QValueCount,nEvalPerContext, ...
    overallSuccess_pct,overallMeanPath_m,overallStdPath_m, ...
    overallSmooth_deg,overallWait,overallAvoid, ...
    overallSteps,overallNav_ms);

fprintf('\n============================================================\n');
fprintf(' STQL DYNAMIC CASE A 1x4 - SUMMARY\n');
fprintf('============================================================\n');
fprintf('Training episodes             : %d\n',totalEpisodes);
fprintf('Training time                 : %.3f s\n',trainingTime_s);
fprintf('Real training steps           : %d\n',realTrainingSteps);
fprintf('Q-value count                 : %d\n',QValueCount);
fprintf('Evaluation runs/context       : %d\n',nEvalPerContext);
fprintf('Overall success rate          : %.2f %%\n',overallSuccess_pct);
fprintf('Mean successful path length   : %.3f +/- %.3f m\n', ...
    overallMeanPath_m,overallStdPath_m);
fprintf('Mean successful smoothness    : %.3f deg\n',overallSmooth_deg);
fprintf('Mean successful WAIT actions  : %.3f\n',overallWait);
fprintf('Mean avoidance events         : %.3f\n',overallAvoid);
fprintf('Mean mission steps            : %.3f\n',overallSteps);
fprintf('Mean navigation CPU time      : %.4f ms\n',overallNav_ms);
fprintf('============================================================\n');

%% ========================================================================
% 13. SAVE RESULTS
% =========================================================================

if saveCSV
    writetable(RawEvaluation, ...
        'STQL_Dynamic_1x4_CaseA_RawEvaluation.csv');

    writetable(ContextSummary, ...
        'STQL_Dynamic_1x4_CaseA_ContextSummary.csv');

    writetable(OverallSummary, ...
        'STQL_Dynamic_1x4_CaseA_OverallSummary.csv');
end

if saveModel
    model.method = 'STQL';
    model.trainingMode = 'DYNAMIC_CONTEXT_MIXED';
    model.Q = Q;
    model.trainingTime_s = trainingTime_s;
    model.realTrainingEpisodes = totalEpisodes;
    model.realTrainingSteps = realTrainingSteps;
    model.QValueCount = QValueCount;
    model.history.success = trainSuccess;
    model.history.steps = trainSteps;
    model.history.return = trainReturn;
    model.history.context = trainContext;

    save('STQL_Dynamic_1x4_CaseA_Model.mat', ...
        'model','env','cfg','-v7.3');
end

%% ========================================================================
% 14. ANIMATED STANDARD Q-LEARNING MISSION
% =========================================================================

assert(demoContext>=1 && demoContext<=env.nCorridors, ...
    'demoContext must be 1...5.');

fprintf('\n============================================================\n');
fprintf(' RUNNING ANIMATED STQL MISSION\n');
fprintf('============================================================\n');
fprintf('Active context : C%d (%s)\n', ...
    demoContext,env.corridors(demoContext).label);
fprintf('Policy used    : single Q(x,y,a) for all contexts\n');
fprintf('Obstacle moves : every %d robot decision(s)\n', ...
    cfg.dynamicObstacleMoveEvery);

demo = runSTQLMission( ...
    Q,env,demoContext,cfg,demoSeed, ...
    true,animationDelay);

if demo.success
    fprintf(['Animated STQL mission SUCCESS | ' ...
        'path=%.3f m | steps=%d | WAIT=%d | avoid=%d\n'], ...
        demo.pathLength_m,demo.steps, ...
        demo.waitCount,demo.avoidanceEvents);
else
    fprintf(['Animated STQL mission FAILED to reach goal ' ...
        'within %d steps | WAIT=%d | avoid=%d\n'], ...
        cfg.maxStepsTest,demo.waitCount, ...
        demo.avoidanceEvents);
end

%% ========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function corridors = buildCorridors( ...
        staticMap,defs,startPoint,goalPoint)

    n = numel(defs);

    corridors = repmat(struct( ...
        'id',0, ...
        'label','', ...
        'half','', ...
        'laneNumber',0, ...
        'x',0, ...
        'yRange',[], ...
        'allCells',[], ...
        'segments',{{}}),n,1);

    for i = 1:n

        d = defs{i};

        cells = [ ...
            d.x*ones(numel(d.yRange),1), ...
            d.yRange(:)];

        corridors(i).id = i;
        corridors(i).label = d.label;
        corridors(i).half = d.half;
        corridors(i).laneNumber = d.laneNumber;
        corridors(i).x = d.x;
        corridors(i).yRange = d.yRange;
        corridors(i).allCells = cells;

        corridors(i).segments = ...
            contiguousFreeSegments( ...
            cells,staticMap,startPoint,goalPoint);

        if isempty(corridors(i).segments)
            error('Corridor %d has no usable segment.',i);
        end
    end
end

function segments = contiguousFreeSegments( ...
        cells,staticMap,startPoint,goalPoint)

    usable = false(size(cells,1),1);

    for i = 1:size(cells,1)

        p = cells(i,:);

        usable(i) = ...
            staticMap(p(1),p(2))==0 && ...
            ~isequal(p,startPoint) && ...
            ~isequal(p,goalPoint);
    end

    segments = {};

    i = 1;

    while i<=numel(usable)

        if usable(i)

            j = i;

            while j<numel(usable) && usable(j+1)
                j = j+1;
            end

            segments{end+1} = cells(i:j,:); %#ok<AGROW>
            i = j+1;

        else
            i = i+1;
        end
    end
end

function eps = epsilonSchedule( ...
        ep,totalEpisodes,eps0,epsMin)

    if totalEpisodes<=1
        eps=epsMin;
        return;
    end

    horizon=max(1,0.85*totalEpisodes);

    eps = eps0*(epsMin/eps0)^(ep/horizon);
    eps = max(epsMin,eps);
end

function state = chooseEpisodeStart( ...
        env,pRandom,forbidden)

    if nargin<3
        forbidden=[];
    end

    if rand>pRandom

        state=env.start;

        if isempty(forbidden) || ...
                ~isequal(state,forbidden)
            return;
        end
    end

    freeIdx=find(env.staticMap==0);

    while true

        pick=freeIdx(randi(numel(freeIdx)));

        [x,y]=ind2sub( ...
            size(env.staticMap),pick);

        state=[x y];

        if isequal(state,env.goal)
            continue;
        end

        if ~isempty(forbidden) && ...
                isequal(state,forbidden)
            continue;
        end

        return;
    end
end

function dyn = initializeDynamicObstacle( ...
        corridor,moveEvery)

    segs=corridor.segments;
    lens=cellfun(@(s)size(s,1),segs);

    p=lens/sum(lens);
    u=rand;

    segID=find(u<=cumsum(p),1,'first');

    path=segs{segID};
    idx=randi(size(path,1));

    if rand<0.5
        direction=-1;
    else
        direction=1;
    end

    dyn.active=true;
    dyn.corridorID=corridor.id;
    dyn.path=path;
    dyn.index=idx;
    dyn.direction=direction;
    dyn.position=path(idx,:);
    dyn.moveEvery=max(1,round(moveEvery));
    dyn.tick=0;
end

function dyn = moveDynamicObstacle( ...
        dyn,robotState)

    if isempty(dyn) || ~dyn.active
        return;
    end

    dyn.tick=dyn.tick+1;

    if mod(dyn.tick-1,dyn.moveEvery)~=0
        return;
    end

    n=size(dyn.path,1);

    if n<=1
        dyn.position=dyn.path(1,:);
        return;
    end

    nextIdx=dyn.index+dyn.direction;

    if nextIdx<1 || nextIdx>n
        dyn.direction=-dyn.direction;
        nextIdx=dyn.index+dyn.direction;
    end

    candidate=dyn.path(nextIdx,:);

    % The moving obstacle is not allowed to enter the robot's current cell.
    if isequal(candidate,robotState)

        dyn.direction=-dyn.direction;

        altIdx=dyn.index+dyn.direction;

        if altIdx>=1 && altIdx<=n

            altCandidate=dyn.path(altIdx,:);

            if ~isequal(altCandidate,robotState)
                dyn.index=altIdx;
                dyn.position=altCandidate;
            end
        end

    else
        dyn.index=nextIdx;
        dyn.position=candidate;
    end
end

function safeIDs = getSafeActions( ...
        state,env,dynPos,includeWait)

    if nargin<4
        includeWait=true;
    end

    safe=false(size(env.actions,1),1);

    for a=1:size(env.actions,1)

        move=env.actions(a,:);

        if all(move==0)
            safe(a)=includeWait;
            continue;
        end

        ns=state+move;

        if ~isStateFree(ns,env.staticMap)
            continue;
        end

        if ~isempty(dynPos) && ...
                isequal(ns,dynPos)
            continue;
        end

        safe(a)=true;
    end

    safeIDs=find(safe);
end

function tf = isStateFree(state,staticMap)

    x=state(1);
    y=state(2);

    tf = ...
        x>=1 && x<=size(staticMap,1) && ...
        y>=1 && y<=size(staticMap,2);

    if tf
        tf=staticMap(x,y)==0;
    end
end

function actionID = epsilonGreedy( ...
        Q,state,safeIDs,epsilon)

    if isempty(safeIDs)
        actionID=[];
        return;
    end

    if rand<epsilon
        actionID=safeIDs( ...
            randi(numel(safeIDs)));
        return;
    end

    q=squeeze(Q( ...
        state(1),state(2),safeIDs));

    q=q(:);

    mx=max(q);

    candidates=safeIDs( ...
        abs(q-mx)<1e-12);

    actionID=candidates( ...
        randi(numel(candidates)));
end

function actionID = greedyAction( ...
        Q,state,safeIDs)

    if isempty(safeIDs)
        actionID=[];
        return;
    end

    q=squeeze(Q( ...
        state(1),state(2),safeIDs));

    q=q(:);

    mx=max(q);

    candidates=safeIDs( ...
        abs(q-mx)<1e-12);

    actionID=candidates( ...
        randi(numel(candidates)));
end

function [reward,terminal] = rewardFunction( ...
        state,nextState,actionID,env,dynPos,cfg)

    terminal=isequal(nextState,env.goal);

    if terminal
        reward=cfg.goalReward;
        return;
    end

    move=env.actions(actionID,:);

    if all(move==0)
        stepCost=cfg.waitPenalty;
    else
        stepCost=-norm(move);
    end

    d0=norm(state-env.goal);
    d1=norm(nextState-env.goal);

    progress=cfg.progressWeight*(d0-d1);

    reward=stepCost+progress;

    if ~isempty(dynPos)

        dObs=norm(nextState-dynPos);

        if dObs<=cfg.nearObstacleRange
            reward=reward+ ...
                cfg.nearObstaclePenalty;
        end
    end
end

function [nextState,reward,terminal] = ...
        realTransition( ...
        state,actionID,env,dynPos,cfg)

    nextState=state+ ...
        env.actions(actionID,:);

    if ~isStateFree( ...
            nextState,env.staticMap) || ...
       (~isempty(dynPos) && ...
        isequal(nextState,dynPos) && ...
        actionID~=env.waitID)

        nextState=state;
        reward=cfg.invalidPenalty;
        terminal=false;
        return;
    end

    [reward,terminal]=rewardFunction( ...
        state,nextState,actionID, ...
        env,dynPos,cfg);
end

function result = runSTQLMission( ...
        Q,env,context,cfg,seed, ...
        animate,animationDelay)

    rng(seed,'twister');

    if context==0
        dyn=[];
        dynPos=[];
    else
        dyn=initializeDynamicObstacle( ...
            env.corridors(context), ...
            cfg.dynamicObstacleMoveEvery);
        dynPos=dyn.position;
    end

    % Standard Q-learning is context-blind:
    % the SAME Q(x,y,a) table is used for C0 and all C1...C5.

    robot=env.start;
    path=robot;

    waitCount=0;
    avoidanceEvents=0;

    if animate
        % Figure size follows the tall 13 x 30 greenhouse aspect ratio.
        nX = env.gridSize(1);
        nY = env.gridSize(2);
        pxPerCell = 22;
        figW = (nX+2)*pxPerCell + 120;
        figH = (nY+2)*pxPerCell + 120;

        fig=figure( ...
            'Name','Standard Q-Learning - Dynamic Case A 1x4 Greenhouse', ...
            'Color','w', ...
            'Position',[100 40 figW figH], ...
            'WindowKeyPressFcn',@togglePauseSTQL);

        % Animation state used by the SPACE-key callback.
        setappdata(fig,'STQLPaused',false);

        ax=axes('Parent',fig);
        set(ax,'Position',[0.11 0.08 0.82 0.86]);

        % Give the user a clear keyboard hint.
        annotation(fig,'textbox',[0.12 0.01 0.76 0.045], ...
            'String','SPACE: pause / resume animation', ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'EdgeColor','none', ...
            'FontWeight','bold', ...
            'FontSize',10);
    else
        fig=[];
        ax=[];
    end

    for step=1:cfg.maxStepsTest

        % Same temporal convention used during STQL dynamic training.
        if context>0
            dyn=moveDynamicObstacle(dyn,robot);
            dynPos=dyn.position;
        else
            dynPos=[];
        end

        % Detect whether the Q-preferred action, ignoring the moving
        % obstacle, is temporarily blocked by its current position.
        staticSafe=getSafeActions( ...
            robot,env,[],true);

        dynamicSafe=getSafeActions( ...
            robot,env,dynPos,true);

        if ~isempty(dynPos) && ...
                ~isempty(staticSafe)

            preferredStatic= ...
                greedyAction( ...
                Q,robot,staticSafe);

            if ~ismember( ...
                    preferredStatic,dynamicSafe)
                avoidanceEvents= ...
                    avoidanceEvents+1;
            end
        end

        if isempty(dynamicSafe)
            actionID=env.waitID;
        else
            actionID=greedyAction( ...
                Q,robot,dynamicSafe);
        end

        if isempty(actionID)
            actionID=env.waitID;
        end

        if actionID==env.waitID
            waitCount=waitCount+1;
        end

        nextRobot= ...
            robot+env.actions(actionID,:);

        % The action mask should already prevent collision.
        if ~isStateFree( ...
                nextRobot,env.staticMap) || ...
           (~isempty(dynPos) && ...
            isequal(nextRobot,dynPos))

            nextRobot=robot;
        end

        robot=nextRobot;
        path=[path;robot]; %#ok<AGROW>

        if animate

            % SPACE toggles appdata(fig,'STQLPaused').
            % While paused, keep processing GUI events so SPACE can resume.
            waitWhilePaused(fig);

            if ~isgraphics(fig)
                warning('Animation figure was closed. Mission continues without animation.');
                animate = false;
            else
                drawSTQLScene( ...
                    ax,env,context,dynPos, ...
                    robot,path,step, ...
                    waitCount,avoidanceEvents,fig);

                drawnow;

                if animationDelay>0
                    pause(animationDelay);
                end
            end
        end

        if isequal(robot,env.goal)
            success=true;
            break;
        end
    end

    if ~isequal(robot,env.goal)
        success=false;
    end

    result.success=success;
    result.path=path;
    result.steps=size(path,1)-1;
    result.waitCount=waitCount;
    result.avoidanceEvents=avoidanceEvents;

    result.pathLength_m= ...
        computePathLength( ...
        path,env.cellSize_m);

    result.smoothness_deg= ...
        computeSmoothness(path);
end

function L = computePathLength( ...
        path,cellSize_m)

    if size(path,1)<2
        L=0;
        return;
    end

    d=diff(path,1,1);

    L=sum( ...
        sqrt(sum(d.^2,2))) ...
        *cellSize_m;
end

function s = computeSmoothness(path)

    if size(path,1)<3
        s=0;
        return;
    end

    d=diff(path,1,1);

    % WAIT actions do not contribute to heading change.
    d=d(any(d~=0,2),:);

    if size(d,1)<2
        s=0;
        return;
    end

    theta=atan2(d(:,2),d(:,1));

    dtheta=diff(theta);

    dtheta=atan2( ...
        sin(dtheta),cos(dtheta));

    s=mean(abs(rad2deg(dtheta)));
end

function drawSTQLScene( ...
        ax,env,context,dynPos, ...
        robot,path,step, ...
        waits,avoidanceEvents,fig)
% Visualization follows the plotting convention used in
% SGH_DijkstraPathPlanning_obs2:
%   - one-cell gray boundary around greenhouse
%   - planting beds as semi-transparent pink/red rectangles
%   - fixed object obstacles as black filled circles
%   - start cell green and goal cell cyan
% Dynamic obstacle keeps the same orange color used by the CMQL animation.

    cla(ax);
    hold(ax,'on');

    nX=env.gridSize(1);
    nY=env.gridSize(2);

    % -------------------------------------------------------------
    % 1) GREENHOUSE PLOT EXTENT AND GRID
    % -------------------------------------------------------------
    axis(ax,'equal');
    xlim(ax,[-1 nX+1]);
    ylim(ax,[-1 nY+1]);

    set(ax, ...
        'XTick',-1:1:nX+1, ...
        'YTick',-1:1:nY+1, ...
        'YDir','normal', ...
        'GridColor',[0.40 0.40 0.40], ...
        'GridAlpha',0.40, ...
        'Layer','top');

    grid(ax,'on');
    box(ax,'on');

    xlabel(ax,'Grid x');
    ylabel(ax,'Grid y');

    % -------------------------------------------------------------
    % 2) ONE-CELL GRAY GREENHOUSE BOUNDARY
    %    Same visual idea as SGH_DijkstraPathPlanning_obs2.
    % -------------------------------------------------------------
    borderGray=[0.75 0.75 0.75];

    % Bottom
    fill(ax, ...
        [-1 nX+1 nX+1 -1], ...
        [-1 -1 0 0], ...
        borderGray, ...
        'EdgeColor','none');

    % Left
    fill(ax, ...
        [-1 0 0 -1], ...
        [-1 -1 nY+1 nY+1], ...
        borderGray, ...
        'EdgeColor','none');

    % Top
    fill(ax, ...
        [-1 nX+1 nX+1 -1], ...
        [nY nY nY+1 nY+1], ...
        borderGray, ...
        'EdgeColor','none');

    % Right
    fill(ax, ...
        [nX nX+1 nX+1 nX], ...
        [-1 -1 nY+1 nY+1], ...
        borderGray, ...
        'EdgeColor','none');

    % -------------------------------------------------------------
    % 3) START / GOAL CELLS
    %    Dijkstra-reference convention:
    %    start = green cell, goal = cyan cell.
    % -------------------------------------------------------------
    rectangle(ax, ...
        'Position',[env.start(1)-1 env.start(2)-1 1 1], ...
        'FaceColor',[0.10 0.90 0.10], ...
        'EdgeColor','none');

    rectangle(ax, ...
        'Position',[env.goal(1)-1 env.goal(2)-1 1 1], ...
        'FaceColor',[0.00 0.90 0.90], ...
        'EdgeColor','none');

    % -------------------------------------------------------------
    % 4) PLANTING BEDS
    %    Pink/red semi-transparent rectangles.
    % -------------------------------------------------------------
    bedColor=[1.00 0.35 0.42];

    for k=1:size(env.bedObs,1)
        rectangle(ax, ...
            'Position',[ ...
                env.bedObs(k,1)-1, ...
                env.bedObs(k,2)-1, ...
                1,1], ...
            'FaceColor',bedColor, ...
            'FaceAlpha',0.30, ...
            'EdgeColor','none');
    end

    % -------------------------------------------------------------
    % 5) FIXED OBJECT OBSTACLES
    %    Black filled circles, as in the Dijkstra reference.
    % -------------------------------------------------------------
    for k=1:size(env.objObs,1)
        plot(ax, ...
            env.objObs(k,1)-0.5, ...
            env.objObs(k,2)-0.5, ...
            'o', ...
            'MarkerFaceColor','black', ...
            'MarkerEdgeColor','black', ...
            'MarkerSize',14);
    end

    % -------------------------------------------------------------
    % 6) ACTIVE CORRIDOR
    % -------------------------------------------------------------
    if context>0
        c=env.corridors(context);

        plot(ax, ...
            c.x*ones(size(c.yRange))-0.5, ...
            c.yRange-0.5, ...
            ':', ...
            'LineWidth',1.4, ...
            'Color',[0.45 0.45 0.45]);

        text(ax, ...
            c.x-0.32, ...
            c.yRange(end)-0.5, ...
            sprintf('C%d',context), ...
            'FontWeight','bold', ...
            'FontSize',9, ...
            'BackgroundColor','w', ...
            'Margin',1);
    end

    % -------------------------------------------------------------
    % 7) ROBOT TRAJECTORY
    % -------------------------------------------------------------
    if size(path,1)>1
        plot(ax, ...
            path(:,1)-0.5, ...
            path(:,2)-0.5, ...
            '-', ...
            'LineWidth',2.0, ...
            'Color',[0.05 0.30 0.95]);
    end

    % Small trajectory nodes, visually similar to the original reference.
    plot(ax, ...
        path(:,1)-0.5, ...
        path(:,2)-0.5, ...
        'o', ...
        'MarkerFaceColor',[0.05 0.30 0.95], ...
        'MarkerEdgeColor','none', ...
        'MarkerSize',4);

    % -------------------------------------------------------------
    % 8) DYNAMIC OBSTACLE
    %    Preserve original CMQL orange color.
    % -------------------------------------------------------------
    if ~isempty(dynPos)
        plot(ax, ...
            dynPos(1)-0.5, ...
            dynPos(2)-0.5, ...
            'o', ...
            'MarkerFaceColor',[0.95 0.55 0.10], ...
            'MarkerEdgeColor',[0.20 0.15 0.05], ...
            'MarkerSize',14, ...
            'LineWidth',1.3);

        text(ax, ...
            dynPos(1)-0.25, ...
            dynPos(2)+0.25, ...
            'DO', ...
            'FontWeight','bold', ...
            'FontSize',8);
    end

    % -------------------------------------------------------------
    % 9) CURRENT ROBOT POSITION
    % -------------------------------------------------------------
    plot(ax, ...
        robot(1)-0.5, ...
        robot(2)-0.5, ...
        'o', ...
        'MarkerFaceColor',[0.00 0.25 1.00], ...
        'MarkerEdgeColor','k', ...
        'MarkerSize',11, ...
        'LineWidth',1.1);

    % Labels
    text(ax,env.start(1)-0.45,env.start(2)-0.5,'S', ...
        'FontWeight','bold','FontSize',9);
    text(ax,env.goal(1)-0.45,env.goal(2)-0.5,'G', ...
        'FontWeight','bold','FontSize',9);

    % -------------------------------------------------------------
    % 10) TITLE / PAUSE STATE
    % -------------------------------------------------------------
    policyText='Q(x,y,a)';

    if context==0
        contextText='C0 - Nominal';
    else
        contextText=sprintf('C%d - %s', ...
            context,env.corridors(context).label);
    end

    if isgraphics(fig) && ...
            isappdata(fig,'STQLPaused') && ...
            getappdata(fig,'STQLPaused')
        runState='PAUSED';
    else
        runState='RUNNING';
    end

    title(ax,sprintf( ...
        ['STQL Case A | %s | %s | step=%d | ' ...
         'WAIT=%d | avoidance=%d | %s'], ...
        policyText,contextText,step, ...
        waits,avoidanceEvents,runState));

    hold(ax,'off');
end

function togglePauseSTQL(fig,event)
% SPACE toggles animation pause state.

    if ~isgraphics(fig)
        return;
    end

    if strcmp(event.Key,'space')
        paused=false;

        if isappdata(fig,'STQLPaused')
            paused=getappdata(fig,'STQLPaused');
        end

        setappdata(fig,'STQLPaused',~paused);

        if ~paused
            set(fig,'Name', ...
                'Standard Q-Learning - Dynamic Case A 1x4 Greenhouse [PAUSED]');
        else
            set(fig,'Name', ...
                'Standard Q-Learning - Dynamic Case A 1x4 Greenhouse [RUNNING]');
        end
    end
end

function waitWhilePaused(fig)
% Keep UI responsive while animation is paused.

    if isempty(fig) || ~isgraphics(fig)
        return;
    end

    while isgraphics(fig) && ...
            isappdata(fig,'STQLPaused') && ...
            getappdata(fig,'STQLPaused')

        drawnow;
        pause(0.05);
    end
end
